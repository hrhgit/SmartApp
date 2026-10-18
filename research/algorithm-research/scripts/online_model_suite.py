"""Online classifiers for causal next-destination replay (no dataset I/O).

``learn(record)`` consumes ONE already-matured, deduplicated outcome. ``scores``
reads only a query's current-interface features and semantic candidate labels.
The replay owner, not this module, enforces label maturity and outcome dedup.

River 0.21.2 implements the streaming classifiers. sklearn's MLPClassifier is
used as a binary candidate scorer to avoid predeclaring destination classes.
Install the explicitly pinned dependencies in an isolated directory, then put
that directory on PYTHONPATH before starting Python; imports never install or
modify dependencies automatically. See dependency_metadata for actual versions.

Official references (API versions correspond to the implementation):
https://riverml.xyz/0.21.2/api/linear-model/SoftmaxRegression/
https://riverml.xyz/0.21.2/api/linear-model/PAClassifier/
https://riverml.xyz/0.21.2/api/tree/HoeffdingTreeClassifier/
https://riverml.xyz/0.21.2/api/forest/ARFClassifier/
https://scikit-learn.org/1.6/modules/generated/sklearn.neural_network.MLPClassifier.html
"""

from __future__ import annotations

import copy
import collections
import hashlib
import importlib.metadata
import math
import random
from typing import Any


DEPENDENCIES = {
    "river": "0.21.2",
    "numpy": "1.26.4",
    "pandas": "2.2.3",
    "scipy": "1.13.1",
    "scikit-learn": "1.6.1",
}
SEED = 20260923


def dependency_metadata() -> dict:
    installed = {}
    for name in DEPENDENCIES:
        try:
            installed[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            installed[name] = None
    return {"expected": dict(DEPENDENCIES), "installed": installed}


def _numeric_x(record: dict) -> dict:
    """Never infer features from a training label or candidate identity."""
    if "x" in record:
        values = record["x"]
    else:
        values = {"origin=" + record["origin"]: 1.0}
        if record.get("app"):
            values["app=" + record["app"]] = 1.0
    output = {}
    for key, value in values.items():
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("Non-finite input feature")
        if number:
            output[str(key)] = number
    output.setdefault("__bias__", 1.0)
    return output


def _nonnegative_x(x: dict) -> dict:
    # Signed features, if a later explicit input ablation introduces them,
    # become distinct positive/negative features. Current onehots are positive.
    return {(str(key), "positive" if value > 0 else "negative"): abs(value)
            for key, value in x.items() if value}


class RiverClassifier:
    """Adapter for River models whose class vocabulary grows during learn_one."""

    def __init__(self, model: Any, feature_mode: str = "numeric"):
        self.model = model
        self.feature_mode = feature_mode
        self.seen_labels = set()

    def _features(self, record: dict) -> dict:
        x = _numeric_x(record)
        if self.feature_mode == "nonnegative":
            return _nonnegative_x(x)
        if self.feature_mode == "nominal_current":
            # These are the same current-interface facts as their onehots;
            # no history or later context is introduced for trees.
            x["__current_origin__"] = record["origin"]
            x["__current_app__"] = record.get("app", "")
        return x

    def learn(self, record: dict) -> None:
        label = record["label"]
        self.model.learn_one(self._features(record), label)
        self.seen_labels.add(label)

    def scores(self, query: dict) -> dict:
        probability = self.model.predict_proba_one(self._features(query))
        result = {}
        for label in query["labels"]:
            value = float(probability.get(label, 0.0))
            if not math.isfinite(value):
                raise ValueError("Online classifier emitted a non-finite score")
            result[label] = value
        return result


class CandidateMLP:
    """A small online MLP estimating compatibility of (current, destination).

    A single binary neural network shares parameters over destination features
    and their interactions with current features. Legacy exact-only records
    create one positive pair. A shared-input record may instead provide all
    cutoff-time useful path actions as ``positive_labels``; they are one update
    for one matured outcome, never extra visits. Up to ``negatives`` unselected
    alternatives are sampled from earlier learned labels. This is a learning
    assumption, not evidence of explicit user rejection.

    The model makes one partial_fit update per outcome; there are no offline
    epochs or fitted encoders. Hashing permits new destination names. No label
    is added to seen_labels during prediction. The legacy replay scores only
    learned labels; the shared-input replay can score every cutoff-time action.
    """

    def __init__(self, hidden: int, learning_rate: float, negatives: int = 8,
                 n_features: int = 2048, seed: int = SEED):
        from sklearn.feature_extraction import FeatureHasher
        from sklearn.neural_network import MLPClassifier

        self.hasher = FeatureHasher(n_features=n_features, input_type="dict",
                                    alternate_sign=True)
        self.model = MLPClassifier(
            hidden_layer_sizes=(hidden,), activation="relu", solver="adam",
            alpha=0.0001, batch_size="auto", learning_rate_init=learning_rate,
            max_iter=1, shuffle=False, random_state=seed,
        )
        self.negatives = negatives
        self.rng = random.Random(seed)
        self.seen_labels = set()
        self._fitted = False

    @staticmethod
    def _candidate_x(record: dict, label: str) -> dict:
        """Read declared cutoff-time candidate fields, never a future label."""
        candidate = record.get("candidate_features", {}).get(label, {})
        if not isinstance(candidate, dict):
            raise ValueError("Candidate features must be a mapping")
        result = {}
        for key, value in candidate.items():
            if key in {"action_id", "semantic_key", "layer_keys"}:
                continue
            if isinstance(value, (int, float)):
                number = float(value)
                if not math.isfinite(number):
                    raise ValueError("Non-finite candidate feature")
                if number:
                    result["value:" + str(key)] = number
            elif isinstance(value, str):
                result["category:" + str(key) + "=" + value] = 1.0
            else:
                raise ValueError("Unsupported candidate feature type")
        return result

    @staticmethod
    def _pair(x: dict, label: str, candidate_x: dict | None = None) -> dict:
        # Length prefix makes concatenation unambiguous even for opaque labels.
        label_token = str(len(label)) + ":" + label
        paired = {"destination:" + label_token: 1.0}
        for key, value in x.items():
            paired["input:" + key] = value
            paired["interaction:" + label_token + ":" + key] = value
        for key, value in (candidate_x or {}).items():
            paired["candidate:" + key] = value
        return paired

    def learn(self, record: dict) -> None:
        if "positive_labels" in record:
            positives = tuple(dict.fromkeys(record["positive_labels"]))
            if not positives or any(not isinstance(label, str) or not label for label in positives):
                raise ValueError("A matured outcome needs useful action labels")
        else:
            positives = (record["label"],)
        positive_set = set(positives)
        # A shared-input replay may draw negatives from every action that was
        # actually available at this cutoff.  The legacy replay keeps its
        # earlier seen-label pool unless it opts in explicitly.
        if record.get("sample_all_asof_negatives", False):
            others = sorted(set(record.get("allowed_negatives", ())) - positive_set)
        else:
            others = sorted(self.seen_labels - positive_set)
        if "allowed_negatives" in record:
            others = [other for other in others
                      if other in record["allowed_negatives"]]
        if len(others) > self.negatives:
            others = self.rng.sample(others, self.negatives)
        labels = [*positives, *others]
        x = _numeric_x(record)
        matrix = self.hasher.transform([
            self._pair(x, item, self._candidate_x(record, item)) for item in labels])
        targets = [1] * len(positives) + [0] * len(others)
        self.model.partial_fit(matrix, targets, classes=[0, 1])
        self._fitted = True
        self.seen_labels.update(positives)

    def scores(self, query: dict) -> dict:
        result = {label: 0.0 for label in query["labels"]}
        labels = ([label for label in query["labels"]]
                  if query.get("allow_unseen_candidates", False)
                  else [label for label in query["labels"] if label in self.seen_labels])
        if not self._fitted or not labels:
            return result
        x = _numeric_x(query)
        matrix = self.hasher.transform([
            self._pair(x, item, self._candidate_x(query, item)) for item in labels])
        probability = self.model.predict_proba(matrix)[:, 1]
        for label, value in zip(labels, probability):
            if not math.isfinite(float(value)):
                raise ValueError("Online MLP emitted a non-finite score")
            result[label] = float(value)
        return result


def _current_distance(left: dict, right: dict) -> float:
    """Current identity dominates app identity; epsilon preserves exact votes.

    River 0.21.2 KNN otherwise short-circuits to a single neighbor when distance
    is exactly zero, suppressing repeated identical-current outcomes. Adding a
    constant epsilon preserves neighbor order and allows those outcomes to vote.
    """
    return (1e-6 + float(left["origin"] != right["origin"])
            + 0.25 * float(left["app"] != right["app"]))


class StreamingKNN:
    """Exact River streaming kNN, FIFO memory, latest-first distance ties."""

    def __init__(self, n_neighbors: int, window_size: int):
        from river import neighbors

        class RecentLazySearch(neighbors.LazySearch):
            def search(self, item, n_neighbors, **kwargs):
                # Reversing an iterator leaves the learned FIFO window intact.
                points = [(*entry, self.dist_func(item, entry[0]))
                          for entry in reversed(self.window)]
                selected = sorted(points, key=lambda point: point[-1])[:n_neighbors]
                return tuple(map(list, zip(*selected)))

        engine = RecentLazySearch(window_size=window_size, dist_func=_current_distance)
        self.model = neighbors.KNNClassifier(
            n_neighbors=n_neighbors, engine=engine, weighted=True, cleanup_every=0)
        self.seen_labels = set()

    @staticmethod
    def _features(record: dict) -> dict:
        return {"origin": record["origin"], "app": record.get("app", "")}

    def learn(self, record: dict) -> None:
        self.model.learn_one(self._features(record), record["label"])
        self.seen_labels.add(record["label"])

    def scores(self, query: dict) -> dict:
        values = self.model.predict_proba_one(self._features(query))
        return {label: float(values.get(label, 0.0)) for label in query["labels"]}


class OnlineClusterFrequency:
    """Incremental River KMeans followed by per-cluster label frequencies.

    Current identities hash into 224 dimensions with magnitude 3; application
    identities hash into a disjoint 32-dimensional block with magnitude 1.
    This is a fixed, collision-prone categorical representation, not a learned
    semantic embedding. In this current-only experiment, clusters mainly group
    current destinations/applications, not rich multi-view situations.

    New centers are seeded from the first distinct learned current identities,
    up to n_clusters, using no label information. Each example is assigned to
    its nearest center BEFORE that center moves. Its destination frequency is
    permanently credited to that assigned cluster: later centroid movement does
    not relabel past outcomes. Queries never seed or update centers or counts.
    """

    def __init__(self, n_clusters: int, center_step: float, seed: int = SEED):
        from river import cluster

        self.model = cluster.KMeans(n_clusters=n_clusters, halflife=center_step,
                                    sigma=0.0, seed=seed)
        # The first distinct current inputs provide causal, label-free seeds.
        self.model.centers = {}
        self.n_clusters = n_clusters
        self.seed_origins = set()
        self.counts = collections.defaultdict(collections.Counter)
        self.seen_labels = set()
        self.last_learned_cluster = None

    @staticmethod
    def _hash(value: str, modulus: int) -> int:
        digest = hashlib.sha256(value.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big") % modulus

    @classmethod
    def _features(cls, record: dict) -> dict:
        # Dense fixed coordinates ensure that centroid updates also move absent
        # coordinates towards zero and prediction never initializes dimensions.
        x = dict.fromkeys(range(256), 0.0)
        x[cls._hash(record["origin"], 224)] = 3.0
        if record.get("app"):
            x[224 + cls._hash(record["app"], 32)] = 1.0
        return x

    def learn(self, record: dict) -> None:
        x = self._features(record)
        origin = record["origin"]
        if len(self.model.centers) < self.n_clusters and origin not in self.seed_origins:
            self.model.centers[len(self.model.centers)] = dict(x)
            self.seed_origins.add(origin)
        assigned = self.model.predict_one(x)
        self.counts[assigned][record["label"]] += 1
        self.last_learned_cluster = assigned
        self.seen_labels.add(record["label"])
        self.model.learn_one(x)

    def scores(self, query: dict) -> dict:
        if not self.model.centers:
            return {label: 0.0 for label in query["labels"]}
        assigned = self.model.predict_one(self._features(query))
        counts = self.counts.get(assigned, {})
        total = sum(counts.values())
        return {label: counts.get(label, 0) / total if total else 0.0
                for label in query["labels"]}


class LocalSoftmax:
    """One dynamic online softmax model for each exact current origin.

    The local model sees a constant bias; origin is represented by model routing.
    This isolates unrelated current interfaces without adding history, time or
    background features, and permits new destination classes at every update.
    """

    def __init__(self, learning_rate: float, l2: float):
        self.learning_rate = learning_rate
        self.l2 = l2
        self.models = {}
        self.seen_labels = set()

    def learn(self, record: dict) -> None:
        from river import linear_model, optim

        origin = record["origin"]
        if origin not in self.models:
            self.models[origin] = linear_model.SoftmaxRegression(
                optimizer=optim.SGD(lr=self.learning_rate), l2=self.l2)
        self.models[origin].learn_one({"__bias__": 1.0}, record["label"])
        self.seen_labels.add(record["label"])

    def scores(self, query: dict) -> dict:
        model = self.models.get(query["origin"])
        values = model.predict_proba_one({"__bias__": 1.0}) if model else {}
        return {label: float(values.get(label, 0.0)) for label in query["labels"]}


def supplemental_configurations() -> list:
    """32 settings added after the initial 59-setting comparison was frozen."""
    grid = []

    def add(family: str, **parameters: Any) -> None:
        suffix = "-".join(str(parameters[key]) for key in sorted(parameters))
        grid.append({"id": "online-" + family + "-" + suffix,
                     "family": family, "parameters": parameters})

    for n_neighbors in (3, 5, 10, 20):
        for window_size in (50, 200, 1000):
            add("streaming_knn", n_neighbors=n_neighbors, window_size=window_size)
    for n_clusters in (2, 4, 8, 16):
        for center_step in (0.05, 0.2, 0.5):
            add("online_cluster_frequency", n_clusters=n_clusters,
                center_step=center_step, seed=SEED)
    for learning_rate in (0.1, 0.3, 1.0, 3.0):
        for l2 in (0.0, 0.001):
            add("local_softmax", learning_rate=learning_rate, l2=l2)
    assert len(grid) == 32
    return grid


def configurations() -> list:
    """Original 59 settings followed by 32 separately identifiable additions."""
    grid = []

    def add(family: str, **parameters: Any) -> None:
        suffix = "-".join(str(parameters[key]) for key in sorted(parameters))
        grid.append({"id": "online-" + family + "-" + suffix,
                     "family": family, "parameters": parameters})

    for family in ("softmax", "logistic_ovr"):
        for learning_rate in (0.03, 0.1, 0.3, 1.0):
            for l2 in (0.0, 0.001):
                add(family, learning_rate=learning_rate, l2=l2)
    for l2 in (0.0, 0.001, 0.01):
        add("perceptron_ovr", l2=l2)
    for C in (0.01, 0.1, 1.0, 10.0):
        for mode in (1, 2):
            add("passive_aggressive_ovr", C=C, mode=mode)
    for learning_rate in (0.03, 0.3):
        for l2 in (0.001, 0.01):
            add("hinge_ovr", learning_rate=learning_rate, l2=l2)
    for family in ("multinomial_nb", "bernoulli_nb"):
        for alpha in (0.1, 1.0, 10.0):
            add(family, alpha=alpha)
    for grace_period in (5, 20, 50):
        for leaf_prediction in ("mc", "nba"):
            add("hoeffding_tree", grace_period=grace_period,
                leaf_prediction=leaf_prediction, delta=0.01)
    for grace_period in (10, 30, 100):
        for leaf_prediction in ("mc", "nba"):
            add("adaptive_hoeffding_tree", grace_period=grace_period,
                leaf_prediction=leaf_prediction, delta=0.01, seed=SEED)
    for n_models in (3, 10):
        for grace_period in (10, 50):
            add("adaptive_random_forest", n_models=n_models,
                grace_period=grace_period, seed=SEED)
    for hidden in (16, 32):
        for learning_rate in (0.001, 0.01, 0.05):
            add("candidate_mlp", hidden=hidden, learning_rate=learning_rate,
                negatives=8, n_features=2048, seed=SEED)
    assert len(grid) == 59
    assert len({config["id"] for config in grid}) == len(grid)
    return copy.deepcopy(grid + supplemental_configurations())


def build(config: dict) -> Any:
    parameters = dict(config["parameters"])
    family = config["family"]
    if family == "candidate_mlp":
        return CandidateMLP(**parameters)
    if family == "streaming_knn":
        return StreamingKNN(**parameters)
    if family == "online_cluster_frequency":
        return OnlineClusterFrequency(**parameters)
    if family == "local_softmax":
        return LocalSoftmax(**parameters)

    from river import forest, linear_model, multiclass, naive_bayes, optim, tree

    feature_mode = "numeric"
    if family in ("softmax", "logistic_ovr", "hinge_ovr"):
        learning_rate = parameters.pop("learning_rate")
        optimizer = optim.SGD(lr=learning_rate)
        if family == "softmax":
            model = linear_model.SoftmaxRegression(optimizer=optimizer, **parameters)
        else:
            loss = optim.losses.Hinge() if family == "hinge_ovr" else optim.losses.Log()
            classifier = linear_model.LogisticRegression(
                optimizer=optimizer, loss=loss, intercept_lr=learning_rate,
                **parameters)
            model = multiclass.OneVsRestClassifier(classifier)
    elif family == "perceptron_ovr":
        model = multiclass.OneVsRestClassifier(linear_model.Perceptron(**parameters))
    elif family == "passive_aggressive_ovr":
        model = multiclass.OneVsRestClassifier(linear_model.PAClassifier(**parameters))
    elif family == "multinomial_nb":
        model = naive_bayes.MultinomialNB(**parameters)
        feature_mode = "nonnegative"
    elif family == "bernoulli_nb":
        model = naive_bayes.BernoulliNB(**parameters)
        feature_mode = "nonnegative"
    elif family in ("hoeffding_tree", "adaptive_hoeffding_tree", "adaptive_random_forest"):
        parameters.update(nominal_attributes=["__current_origin__", "__current_app__"],
                          max_depth=12)
        feature_mode = "nominal_current"
        if family == "hoeffding_tree":
            model = tree.HoeffdingTreeClassifier(**parameters)
        elif family == "adaptive_hoeffding_tree":
            model = tree.HoeffdingAdaptiveTreeClassifier(**parameters)
        else:
            model = forest.ARFClassifier(max_features=None, **parameters)
    else:
        raise ValueError("Unknown online model family: " + family)
    return RiverClassifier(model, feature_mode=feature_mode)
