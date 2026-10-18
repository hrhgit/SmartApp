"""Causal hierarchical fallback policies for offline recommendation research.

The module keeps three quantities separate:

* exact arrivals are mutually exclusive outcomes;
* path support is derived from those arrivals and is never another arrival;
* fallback utility is a decision preference, not a predicted probability.

Explicit ``parentID`` links, URL and file-path ancestors derived from known
reopenable children, the approved ChatGPT service/home relation, and a confirmed
owner application are admitted. Derived parent actions exist only in this
in-memory hierarchy; they are neither visits nor persisted candidates. File
parents are parsed lexically, without scanning or resolving the filesystem.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
import hashlib
import math
import posixpath
from typing import Callable, Iterable, Mapping, Sequence
from urllib.parse import quote, unquote, urlsplit

import numpy as np

import evaluation_contract as scoring


UNKNOWN_ROOT = "hierarchy:unknown-root"
UNKNOWN_PREFIX = "hierarchy:unknown-child:"
CONTENT_EDGES = {
    "explicit_parent", "service_home", "url_parent", "path_parent",
}


@dataclass(frozen=True)
class ParentEdge:
    parent: str
    kind: str
    distance: int = 1


@dataclass(frozen=True)
class FallbackPolicy:
    """A small, inspectable family of path-utility functions."""

    name: str
    mode: str
    content_retention: float = 0.8
    app_retention: float = scoring.FALLBACK_GAIN
    floor: float = 0.4

    def __post_init__(self) -> None:
        if self.mode not in {"product", "saturating"}:
            raise ValueError("Unknown fallback policy mode")
        for value in (self.content_retention, self.app_retention, self.floor):
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Fallback values must be finite and in [0, 1]")
        if self.mode == "saturating" and self.floor >= self.content_retention:
            raise ValueError("Saturating floor must be below one-step retention")

    def path_utility(self, edge_kinds: Sequence[str]) -> float:
        if not edge_kinds:
            return 1.0
        if "owner_app" in edge_kinds:
            # The app edge is deliberately a separate coarse fallback.  An app
            # is never treated as another URL/path level.
            return self.app_retention if len(edge_kinds) == 1 else 0.0
        if any(kind not in CONTENT_EDGES for kind in edge_kinds):
            raise ValueError("Unsupported hierarchy edge")
        service_distance = sum(kind == "service_home" for kind in edge_kinds)
        explicit_distance = sum(
            kind in {"explicit_parent", "url_parent", "path_parent"}
            for kind in edge_kinds
        )
        if self.mode == "product":
            explicit_utility = self.content_retention ** explicit_distance
        elif explicit_distance:
            ratio = (self.content_retention - self.floor) / (1.0 - self.floor)
            explicit_utility = self.floor + (1.0 - self.floor) * ratio ** explicit_distance
        else:
            explicit_utility = 1.0
        # A confirmed website root keeps the agreed homepage value of 0.6.
        # Other recorded URL and directory parent levels use the content curve.
        return explicit_utility * (scoring.FALLBACK_GAIN ** service_distance)


class Hierarchy:
    """Prediction-time known actions plus non-executable unknown outcomes."""

    def __init__(self, metadata: Mapping[str, Mapping], parents: Mapping[str, Sequence[ParentEdge]],
                 executable: Iterable[str] | None = None, *, _skip_validation: bool = False) -> None:
        self.metadata = {str(key): dict(value) for key, value in metadata.items()}
        self.parents = {str(key): tuple(value) for key, value in parents.items()}
        self.executable = set(self.metadata if executable is None else executable)
        self._homepage_cache: dict[str, tuple[str, ...]] | None = None
        self._homepage_group_by_id_cache: dict[str, str] | None = None
        if not _skip_validation:
            self._validate()

    @property
    def derived_parent_ids(self) -> frozenset[str]:
        """Transient structural destinations, never independently observed targets."""
        return frozenset(
            identifier for identifier, target in self.metadata.items()
            if target.get("_derived_parent") is True
        )

    def homepage_ids(self) -> dict[str, tuple[str, ...]]:
        if self._homepage_cache is None:
            self._homepage_cache = _homepage_ids(self.metadata)
        return self._homepage_cache

    def homepage_group(self, identifier: str) -> str | None:
        if self._homepage_group_by_id_cache is None:
            self._homepage_group_by_id_cache = {
                action: group
                for group, actions in self.homepage_ids().items()
                for action in actions
            }
        return self._homepage_group_by_id_cache.get(identifier)

    def _validate(self) -> None:
        for child, edges in self.parents.items():
            if len({edge.parent for edge in edges}) != len(edges):
                raise ValueError("Duplicate parent edge")
            for edge in edges:
                if edge.parent == child:
                    raise ValueError("Self-parent hierarchy edge")
                if edge.kind not in CONTENT_EDGES | {"owner_app"}:
                    raise ValueError("Unknown hierarchy edge kind")
                if not isinstance(edge.distance, int) or edge.distance < 1:
                    raise ValueError("Hierarchy edge distance must be a positive integer")
        state: dict[str, int] = {}
        for root in self.parents:
            if state.get(root) == 2:
                continue
            stack: list[tuple[str, bool]] = [(root, False)]
            while stack:
                node, exiting = stack.pop()
                if exiting:
                    state[node] = 2
                    continue
                current = state.get(node, 0)
                if current == 1:
                    raise ValueError("Hierarchy cycle")
                if current == 2:
                    continue
                state[node] = 1
                stack.append((node, True))
                for edge in reversed(self.parents.get(node, ())):
                    parent_state = state.get(edge.parent, 0)
                    if parent_state == 1:
                        raise ValueError("Hierarchy cycle")
                    if parent_state == 0:
                        stack.append((edge.parent, False))

    def paths(self, outcome: str) -> dict[str, tuple[str, ...]]:
        """Return the shortest typed path from every ancestor to ``outcome``."""
        result: dict[str, tuple[str, ...]] = {outcome: ()}
        queue = deque([(outcome, (), frozenset({outcome}))])
        while queue:
            child, path, seen = queue.popleft()
            for edge in self.parents.get(child, ()):
                if edge.parent in seen:
                    raise ValueError("Hierarchy cycle")
                candidate = (edge.kind,) * edge.distance + path
                previous = result.get(edge.parent)
                if previous is None or len(candidate) < len(previous):
                    result[edge.parent] = candidate
                    queue.append((edge.parent, candidate, seen | {edge.parent}))
        return result

    def ancestors(self, outcome: str) -> set[str]:
        return set(self.paths(outcome)) - {outcome}

    def specificity(self, action: str) -> int:
        """Stable tie-break only; never used as probability or utility."""
        return max(
            (sum(kind in CONTENT_EDGES for kind in path) for path in self.paths(action).values()),
            default=0,
        )

    def with_unknown_children(self, parents: Mapping[str, float]) -> tuple["Hierarchy", dict[str, float]]:
        metadata = dict(self.metadata)
        edges = {key: list(value) for key, value in self.parents.items()}
        masses: dict[str, float] = {}
        for index, (parent, mass) in enumerate(sorted(parents.items())):
            if parent not in self.executable or not math.isfinite(mass) or mass <= 0:
                continue
            key = f"{UNKNOWN_PREFIX}{index}"
            metadata[key] = {"id": key, "kind": "unknown"}
            edge_kind = "owner_app" if metadata.get(parent, {}).get("kind") == "app" else "explicit_parent"
            edges[key] = [ParentEdge(parent, edge_kind)]
            masses[key] = float(mass)
        return Hierarchy(metadata, edges, self.executable), masses


def _homepage_ids(metadata: Mapping[str, Mapping]) -> dict[str, tuple[str, ...]]:
    candidates: dict[str, list[str]] = {}
    for identifier, target in metadata.items():
        location = _web_location(target)
        if location is None or location[1]:
            continue
        origin, _, query, fragment = location
        group = scoring.service(target)
        if not group and (query or fragment):
            continue
        group = group or _web_home_group(origin)
        candidates.setdefault(group, []).append(identifier)
    return {group: tuple(sorted(identifiers)) for group, identifiers in candidates.items()}


def is_service_home(target: Mapping) -> bool:
    try:
        return bool(scoring.service(target)) and urlsplit(target.get("locator", "")).path in {"", "/"}
    except (TypeError, ValueError):
        return False


def _web_home_group(origin: tuple[str, str, int | None]) -> str:
    scheme, host, port = origin
    suffix = f":{port}" if port is not None else ""
    return f"web-home:{scheme}://{host}{suffix}/"


def _web_location(target: Mapping) -> tuple[
    tuple[str, str, int | None], tuple[str, ...], str, str,
] | None:
    """Parse a web target without changing its exact stored locator.

    Paths are split only on literal URL slashes. Ambiguous dot, empty, or
    backslash segments are rejected instead of being normalized into a guessed
    parent. Query and fragment remain available to the caller but do not add
    path levels.
    """
    if target.get("kind") != "page":
        return None
    locator = target.get("locator")
    if not isinstance(locator, str) or not locator:
        return None
    try:
        parsed = urlsplit(locator)
        scheme = parsed.scheme.lower()
        host = parsed.hostname
        port = parsed.port
        if (scheme not in {"http", "https"} or not host
                or parsed.username is not None or parsed.password is not None
                or "\\" in parsed.path):
            return None
        if port == (80 if scheme == "http" else 443):
            port = None
        path = parsed.path
        if path in {"", "/"}:
            segments: tuple[str, ...] = ()
        else:
            if not path.startswith("/") or "//" in path:
                return None
            raw_segments = path[1:].split("/")
            if raw_segments and raw_segments[-1] == "":
                raw_segments.pop()
            if any(segment in {"", ".", ".."} for segment in raw_segments):
                return None
            decoded_segments = tuple(unquote(segment, errors="strict") for segment in raw_segments)
            if any(segment in {".", ".."} or "/" in segment or "\\" in segment
                   for segment in decoded_segments):
                return None
            segments = tuple(raw_segments)
        origin = (scheme, host.lower().rstrip("."), port)
        return origin, segments, parsed.query, parsed.fragment
    except (UnicodeDecodeError, TypeError, ValueError):
        return None


def _file_location(target: Mapping) -> tuple[str, str] | None:
    """Return a lexical file-URL location; never resolve or inspect the path."""
    if target.get("kind") not in {"directory", "file"}:
        return None
    locator = target.get("locator")
    if not isinstance(locator, str) or not locator:
        return None
    try:
        parsed = urlsplit(locator)
        if (parsed.scheme.lower() != "file" or parsed.query or parsed.fragment
                or parsed.username is not None or parsed.password is not None
                or parsed.port is not None or not parsed.path.startswith("/")
                or "%2f" in parsed.path.lower() or "\\" in parsed.path):
            return None
        path = unquote(parsed.path, errors="strict")
        if "\x00" in path:
            return None
        normalized = posixpath.normpath(path)
        if not normalized.startswith("/"):
            return None
        host = (parsed.hostname or "").lower().rstrip(".")
        if host == "localhost":
            host = ""
        return host, normalized
    except (UnicodeDecodeError, TypeError, ValueError):
        return None


def _locator_parent_index(
    metadata: Mapping[str, Mapping],
) -> tuple[dict[tuple[tuple[str, str, int | None], tuple[str, ...]], list[str]],
           dict[tuple[str, str], list[str]]]:
    """Index only known, clean URL paths and known directory locators."""
    web_index: dict[tuple[tuple[str, str, int | None], tuple[str, ...]], list[str]] = {}
    directory_index: dict[tuple[str, str], list[str]] = {}
    for identifier, target in metadata.items():
        web = _web_location(target)
        if web is not None:
            origin, segments, query, fragment = web
            if not query and not fragment:
                web_index.setdefault((origin, segments), []).append(identifier)
        if target.get("kind") == "directory":
            location = _file_location(target)
            if location is not None:
                directory_index.setdefault(location, []).append(identifier)
    return web_index, directory_index


def _derived_parent_id(kind: str, locator: str, metadata: Mapping[str, Mapping]) -> str:
    digest = hashlib.sha256(f"{kind}\0{locator}".encode("utf-8")).hexdigest()
    for size in (24, 32, 48, 64):
        identifier = f"derived-parent:{kind}:{digest[:size]}"
        existing = metadata.get(identifier)
        if existing is None or (
            existing.get("_derived_parent") is True
            and existing.get("kind") == kind
            and existing.get("locator") == locator
        ):
            return identifier
    raise ValueError("Could not allocate a stable derived-parent identity")


def _web_ancestor_locator(origin: tuple[str, str, int | None],
                          segments: Sequence[str]) -> str:
    scheme, host, port = origin
    authority_host = f"[{host}]" if ":" in host and not host.startswith("[") else host
    authority = authority_host + (f":{port}" if port is not None else "")
    path = "/" if not segments else "/" + "/".join(segments) + "/"
    return f"{scheme}://{authority}{path}"


def _file_ancestor_locator(host: str, path: str) -> str:
    authority_host = f"[{host}]" if ":" in host and not host.startswith("[") else host
    encoded_path = quote(path, safe="/:@")
    return f"file://{authority_host}{encoded_path}"


def _add_derived_locator_parents(
    metadata: dict[str, dict], executable: set[str], source_ids: Iterable[str], *,
    make_executable: bool,
) -> None:
    """Materialize path ancestors of known children as transient destinations.

    A recorded parent locator wins. If no target at that exact structural
    location has ever been recorded, create an in-memory parent action from the
    child's locator. If a parent is recorded but unavailable, do not evade
    that decision by manufacturing a duplicate action.
    """
    web_index, directory_index = _locator_parent_index(metadata)
    homepage_groups = set(_homepage_ids(metadata))

    def add_parent(kind: str, locator: str,
                   index: dict, index_key: object) -> None:
        recorded = index.get(index_key, ())
        if recorded:
            return
        identifier = _derived_parent_id(kind, locator, metadata)
        if identifier not in metadata:
            metadata[identifier] = {
                "id": identifier,
                "kind": kind,
                "locator": locator,
                "_derived_parent": True,
            }
        index.setdefault(index_key, []).append(identifier)
        if make_executable:
            executable.add(identifier)

    for identifier in sorted(set(map(str, source_ids))):
        target = metadata.get(identifier)
        if target is None or (make_executable and identifier not in executable):
            continue
        web = _web_location(target)
        if web is not None:
            origin, child_segments, _, _ = web
            service = scoring.service(target)
            home_group = service or _web_home_group(origin)
            for depth in range(len(child_segments)):
                if depth == 0 and home_group in homepage_groups:
                    continue
                parent_segments = child_segments[:depth]
                locator = _web_ancestor_locator(origin, parent_segments)
                add_parent(
                    "page", locator, web_index, (origin, parent_segments),
                )
                if depth == 0:
                    homepage_groups.add(home_group)

        file_location = _file_location(target)
        if file_location is not None:
            host, child_path = file_location
            parent_path = posixpath.dirname(child_path)
            while parent_path != child_path:
                add_parent(
                    "directory", _file_ancestor_locator(host, parent_path),
                    directory_index, (host, parent_path),
                )
                if parent_path == "/":
                    break
                child_path = parent_path
                parent_path = posixpath.dirname(parent_path)


def _inferred_locator_edges(identifier: str, target: Mapping,
                            locator_index: tuple[
                                Mapping[tuple[tuple[str, str, int | None], tuple[str, ...]], Sequence[str]],
                                Mapping[tuple[str, str], Sequence[str]],
                            ]) -> list[ParentEdge]:
    edges: list[ParentEdge] = []
    web_index, directory_index = locator_index
    web = _web_location(target)
    if web is not None:
        origin, child_segments, _, _ = web
        for depth in range(len(child_segments)):
            kind = "service_home" if depth == 0 else "url_parent"
            distance = 1 if depth == 0 else len(child_segments) - depth
            for parent_id in web_index.get((origin, child_segments[:depth]), ()):
                if parent_id != identifier:
                    edges.append(ParentEdge(parent_id, kind, distance))

    file_location = _file_location(target)
    if file_location is not None:
        host, child_path = file_location
        path_below_parent = child_path
        parent_path = posixpath.dirname(path_below_parent)
        distance = 1
        while parent_path != path_below_parent:
            for parent_id in directory_index.get((host, parent_path), ()):
                if parent_id != identifier:
                    edges.append(ParentEdge(parent_id, "path_parent", distance))
            if parent_path == "/":
                break
            path_below_parent = parent_path
            parent_path = posixpath.dirname(parent_path)
            distance += 1
    return edges


def _parent_edges(identifier: str, target: Mapping, metadata: Mapping[str, Mapping],
                  homes: Mapping[str, Sequence[str]], include_owner_apps: bool,
                  locator_index: tuple[
                      Mapping[tuple[tuple[str, str, int | None], tuple[str, ...]], Sequence[str]],
                      Mapping[tuple[str, str], Sequence[str]],
                  ]) -> list[ParentEdge]:
    candidates: dict[str, ParentEdge] = {}

    def add(edge: ParentEdge) -> None:
        if edge.parent != identifier and edge.parent in metadata:
            candidates.setdefault(edge.parent, edge)

    explicit = target.get("parentID")
    if isinstance(explicit, str) and explicit in metadata:
        add(ParentEdge(explicit, "explicit_parent"))
    group = scoring.service(target)
    if not is_service_home(target):
        for home in homes.get(group, ()) if group else ():
            add(ParentEdge(home, "service_home"))
    for edge in _inferred_locator_edges(identifier, target, locator_index):
        prior = candidates.get(edge.parent)
        if prior is None or prior.kind not in {"explicit_parent", "service_home"}:
            if prior is None or edge.distance < prior.distance:
                candidates[edge.parent] = edge
    bundle = target.get("bundleID")
    owner = "app:" + bundle if isinstance(bundle, str) and bundle else None
    owner_meta = metadata.get(owner, {}) if owner else {}
    if (include_owner_apps and owner != identifier
            and owner_meta.get("kind") == "app"
            and owner_meta.get("id") == owner
            and owner_meta.get("bundleID") == bundle):
        add(ParentEdge(owner, "owner_app"))
    return sorted(
        candidates.values(),
        key=lambda edge: (edge.kind, edge.parent, edge.distance),
    )


def build_hierarchy(targets: Iterable[Mapping], *, executable: Iterable[str] | None = None,
                    include_owner_apps: bool = True) -> Hierarchy:
    """Build as-of actions and transient path parents derived from them."""
    metadata = {str(target["id"]): dict(target) for target in targets if target.get("id")}
    allowed = set(metadata if executable is None else executable)
    _add_derived_locator_parents(
        metadata, allowed, tuple(allowed), make_executable=True,
    )
    homes = _homepage_ids(metadata)
    locator_index = _locator_parent_index(metadata)
    parents: dict[str, list[ParentEdge]] = {}
    for identifier, target in metadata.items():
        candidates = _parent_edges(
            identifier, target, metadata, homes, include_owner_apps, locator_index,
        )
        if candidates:
            # A target may have a content parent and an owner app.  They are
            # alternative fallback routes, not a manufactured single chain.
            parents[identifier] = candidates
    return Hierarchy(metadata, parents, allowed)


def add_outcome(hierarchy: Hierarchy, target: Mapping) -> Hierarchy:
    """Attach a future truth without making its target or parents executable."""
    identifier = target.get("id")
    if not identifier:
        return hierarchy
    identifier = str(identifier)
    if identifier in hierarchy.metadata:
        return hierarchy
    metadata = dict(hierarchy.metadata)
    metadata[identifier] = dict(target)
    executable = set(hierarchy.executable)
    _add_derived_locator_parents(
        metadata, executable, (identifier,), make_executable=False,
    )
    parents = {key: list(value) for key, value in hierarchy.parents.items()}
    edges = _parent_edges(
        identifier, target, metadata, _homepage_ids(metadata), True,
        _locator_parent_index(metadata),
    )
    if edges:
        parents[identifier] = edges
    # The outcome is a new non-executable leaf.  Existing as-of edges are kept
    # exactly as they were, so no full-graph revalidation or future backfill is
    # needed here.
    return Hierarchy(metadata, parents, hierarchy.executable, _skip_validation=True)


def path_support(arrivals: Iterable[tuple[str, str, float]], hierarchy: Hierarchy) -> dict[str, Counter]:
    """Count one exact arrival and at most one derived support per ancestor."""
    exact: Counter[str] = Counter()
    derived: Counter[str] = Counter()
    content: Counter[str] = Counter()
    owner_apps: Counter[str] = Counter()
    seen: dict[str, tuple[str, float]] = {}
    for outcome_id, target, weight in arrivals:
        previous = seen.get(outcome_id)
        if previous is not None:
            if previous != (target, weight):
                raise ValueError("One arrival ID cannot describe different targets or weights")
            continue
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("Arrival weight must be finite and positive")
        seen[outcome_id] = (target, weight)
        exact[target] += weight
        for ancestor, path in unique_ancestor_paths(hierarchy, target).items():
            derived[ancestor] += weight
            if "owner_app" in path:
                owner_apps[ancestor] += weight
            else:
                content[ancestor] += weight
    return {"exact": exact, "path": derived, "combined": exact + derived,
            "content_path": content, "owner_app_support": owner_apps,
            "unique_outcomes": Counter({"n": len(seen)})}


def unique_ancestor_paths(hierarchy: Hierarchy, outcome: str) -> dict[str, tuple[str, ...]]:
    """Collapse duplicate IDs for the same service homepage into one support."""
    result: dict[str, tuple[str, ...]] = {}
    seen_home_groups: set[str] = set()
    paths = hierarchy.paths(outcome)
    for ancestor, path in sorted(paths.items(), key=lambda item: (len(item[1]), item[0])):
        if ancestor == outcome:
            continue
        group = hierarchy.homepage_group(ancestor)
        if group:
            if group in seen_home_groups:
                continue
            seen_home_groups.add(group)
            representatives = hierarchy.homepage_ids().get(group, ())
            representative = representatives[0] if representatives else ancestor
            result[representative] = path
        else:
            result[ancestor] = path
    return result


def posterior_from_scores(scores: Mapping[str, float], semantics: str, *, temperature: float = 1.0) -> dict[str, float]:
    """Convert declared probabilities or logits; reject bare ranking values."""
    if not scores:
        return {}
    values = np.asarray(list(scores.values()), dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Scores must be finite")
    labels = list(scores)
    if semantics == "probabilities":
        if np.any(values < 0) or not math.isclose(float(values.sum()), 1.0, abs_tol=1e-9):
            raise ValueError("Declared probabilities must be nonnegative and sum to one")
        return dict(zip(labels, map(float, values)))
    if semantics == "ranking":
        raise ValueError("Ranking values are not probabilities; use rank_proxy_posterior explicitly")
    if semantics != "logits" or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Expected logits with a finite positive temperature")
    scaled = values / temperature
    probabilities = np.exp(scaled - scaled.max())
    probabilities /= probabilities.sum()
    return dict(zip(labels, map(float, probabilities)))


def rank_proxy_posterior(scores: Mapping[str, float], *, power: float = 1.0) -> dict[str, float]:
    """Scale-invariant rank proxy for models that do not claim probabilities."""
    if not math.isfinite(power) or power <= 0:
        raise ValueError("Rank power must be finite and positive")
    if not scores or any(not math.isfinite(float(value)) for value in scores.values()):
        raise ValueError("Rank scores must be nonempty and finite")
    order = sorted(scores, key=lambda key: (-scores[key], key))
    weights = np.asarray([1.0 / ((index + 1) ** power) for index in range(len(order))])
    weights /= weights.sum()
    return dict(zip(order, map(float, weights)))


def attach_open_mass(posterior: Mapping[str, float], hierarchy: Hierarchy, open_mass: float,
                     parent_weights: Mapping[str, float]) -> tuple[dict[str, float], Hierarchy]:
    """Keep unknown outcomes; never renormalize them into known children."""
    if (not 0 <= open_mass < 1
            or any(not math.isfinite(float(value)) or value < 0 for value in posterior.values())
            or not math.isclose(math.fsum(posterior.values()), 1.0, abs_tol=1e-9)):
        raise ValueError("Invalid posterior or open-set mass")
    eligible = {key: float(value) for key, value in parent_weights.items()
                if key in hierarchy.executable and math.isfinite(value) and value > 0}
    total = math.fsum(eligible.values())
    parent_mass: dict[str, float] = {}
    if total:
        ordered = sorted(eligible.items())
        allocated = 0.0
        for key, value in ordered[:-1]:
            share = open_mass * value / total
            parent_mass[key] = share
            allocated += share
        parent_mass[ordered[-1][0]] = max(0.0, open_mass - allocated)
    expanded, unknowns = hierarchy.with_unknown_children(parent_mass)
    result = {key: (1.0 - open_mass) * float(value) for key, value in posterior.items()}
    result.update(unknowns)
    result[UNKNOWN_ROOT] = max(0.0, open_mass - math.fsum(unknowns.values()))
    if not math.isclose(math.fsum(result.values()), 1.0, abs_tol=1e-9):
        raise AssertionError("Open-set expansion did not preserve probability mass")
    return result, expanded


def blocked_current_actions(
    hierarchy: Hierarchy,
    *,
    current_action: str | None = None,
    current_app: str | None = None,
) -> set[str]:
    """Return exact and semantically equivalent actions already reached now."""
    blocked = {value for value in (current_action, current_app) if value is not None}
    if current_action is None:
        return blocked
    group = hierarchy.homepage_group(current_action)
    if not group:
        return blocked
    blocked.update(set(hierarchy.homepage_ids().get(group, ())) & hierarchy.executable)
    return blocked


def deduplicate_equivalent_service_homes(
    ranked: Sequence[str], hierarchy: Hierarchy,
) -> list[str]:
    """Keep one model-ranked ID for each confirmed equivalent site homepage."""
    result: list[str] = []
    seen_groups: set[str] = set()
    for action in ranked:
        group = hierarchy.homepage_group(action)
        if group:
            if group in seen_groups:
                continue
            seen_groups.add(group)
        result.append(action)
    return result


def action_utility(hierarchy: Hierarchy, action: str, outcome: str, policy: FallbackPolicy,
                   *, current_action: str | None = None, current_app: str | None = None) -> float:
    if action not in hierarchy.executable or action in blocked_current_actions(
        hierarchy, current_action=current_action, current_app=current_app,
    ):
        return 0.0
    action_group = hierarchy.homepage_group(action)
    if action_group and action_group == hierarchy.homepage_group(outcome):
        return 1.0
    paths = hierarchy.paths(outcome)
    if action not in paths:
        return 0.0
    return policy.path_utility(paths[action])


def useful_fallback_actions(
    hierarchy: Hierarchy,
    outcome: str,
    policy: FallbackPolicy,
    *,
    current_action: str | None = None,
    current_app: str | None = None,
) -> dict[str, tuple[str, ...]]:
    """Return as-of executable non-exact actions that can help this outcome.

    Equivalent homepage records are represented as an audit route rather
    than a parent edge: they satisfy the same homepage intent, but neither one
    is the other's parent.
    """
    result: dict[str, tuple[str, ...]] = {}
    paths = hierarchy.paths(outcome)
    for action, path in paths.items():
        if action == outcome:
            continue
        if action_utility(
            hierarchy, action, outcome, policy,
            current_action=current_action, current_app=current_app,
        ) > 0:
            result[action] = path

    group = hierarchy.homepage_group(outcome)
    if group:
        for action in hierarchy.executable:
            if action == outcome or action in result:
                continue
            if (hierarchy.homepage_group(action) == group
                    and action_utility(
                        hierarchy, action, outcome, policy,
                        current_action=current_action, current_app=current_app,
                    ) > 0):
                result[action] = ("equivalent_service_home",)
    return result


def menu_value(menu: Sequence[str], posterior: Mapping[str, float], hierarchy: Hierarchy,
               policy: FallbackPolicy, *, current_action: str | None = None,
               current_app: str | None = None) -> float:
    contributions = []
    for outcome in sorted(posterior):
        probability = posterior[outcome]
        best = max((action_utility(hierarchy, action, outcome, policy,
                                   current_action=current_action, current_app=current_app)
                    for action in menu), default=0.0)
        contributions.append(probability * best)
    return math.fsum(contributions)


def greedy_menu(posterior: Mapping[str, float], hierarchy: Hierarchy, policy: FallbackPolicy,
                *, size: int = 4, current_action: str | None = None,
                current_app: str | None = None,
                candidate_order: Sequence[str] | None = None) -> tuple[list[str], list[float]]:
    """Greedily add the action with the largest non-duplicated menu value."""
    if size < 1:
        return [], []
    if any(not math.isfinite(value) or value < 0 for value in posterior.values()):
        raise ValueError("Posterior must be finite and nonnegative")
    if not math.isclose(sum(posterior.values()), 1.0, abs_tol=1e-9):
        raise ValueError("Posterior must sum to one")
    candidate_pool = hierarchy.executable - blocked_current_actions(
        hierarchy, current_action=current_action, current_app=current_app,
    )
    supplied_order = list(posterior) if candidate_order is None else list(candidate_order)
    if len(supplied_order) != len(set(supplied_order)):
        raise ValueError("Candidate order cannot contain duplicates")
    candidates = [key for key in supplied_order if key in candidate_pool]
    candidates.extend(sorted(candidate_pool - set(candidates)))
    base_index = {action: index for index, action in enumerate(candidates)}
    outcomes = sorted(posterior)
    paths = {outcome: hierarchy.paths(outcome) for outcome in outcomes}
    specificity = {
        action: max(
            (sum(kind in CONTENT_EDGES for kind in path)
             for path in (paths.get(action) or hierarchy.paths(action)).values()),
            default=0,
        )
        for action in candidates
    }
    candidate_set = set(candidates)
    home_groups = {
        identifier: hierarchy.homepage_group(identifier)
        for identifier in candidate_set | set(outcomes)
    }
    home_candidates: dict[str, list[str]] = {}
    for action in candidates:
        group = home_groups[action]
        if group:
            home_candidates.setdefault(group, []).append(action)
    path_items = {
        outcome: tuple(sorted(outcome_paths.items()))
        for outcome, outcome_paths in paths.items()
    }
    selected: list[str] = []
    values: list[float] = []
    current_best = {outcome: 0.0 for outcome in outcomes}
    current_value = 0.0
    while len(selected) < min(size, len(candidates)):
        contributions: dict[str, list[float]] = {action: [] for action in candidates if action not in selected}
        for outcome in outcomes:
            probability = posterior[outcome]
            previous = current_best[outcome]
            for action, path in path_items[outcome]:
                if action not in candidate_set or action in selected:
                    continue
                utility = policy.path_utility(path)
                if utility > previous:
                    contributions[action].append(probability * (utility - previous))
            group = home_groups[outcome]
            if group:
                for action in home_candidates.get(group, ()):
                    if action in selected or action in paths[outcome]:
                        continue
                    if 1.0 > previous:
                        contributions[action].append(probability * (1.0 - previous))
        scored = [
            (current_value + math.fsum(contributions[action]), specificity[action],
             -base_index[action], action)
            for action in candidates if action not in selected
        ]
        if not scored:
            break
        value, _, _, action = max(scored, key=lambda item: (item[0], item[1], item[2]))
        if value <= current_value + 1e-15:
            break
        selected.append(action)
        values.append(value)
        current_value = value
        for outcome in outcomes:
            if home_groups.get(action) and home_groups[action] == home_groups.get(outcome):
                utility = 1.0
            else:
                path = paths[outcome].get(action)
                utility = policy.path_utility(path) if path is not None else 0.0
            current_best[outcome] = max(
                current_best[outcome],
                utility,
            )
    return selected, values


def experimental_rank_score(predictions: Sequence[str], truth: Mapping, hierarchy: Hierarchy,
                            policy: FallbackPolicy, *, current_action: str | None = None,
                            current_app: str | None = None, cutoff: int = 4) -> dict[str, float]:
    """Research-only graded score; it never replaces the shared 0.6 scorer."""
    if cutoff != 4:
        raise ValueError("The shared report contract requires cutoff=4")
    expanded = add_outcome(hierarchy, truth)
    truth_id = str(truth.get("id", ""))
    best = 0.0
    prefixes = []
    for index in range(cutoff):
        if index < len(predictions):
            best = max(best, action_utility(expanded, predictions[index], truth_id, policy,
                                            current_action=current_action, current_app=current_app))
        prefixes.append(best)
    return {"top1": prefixes[0], "top2": prefixes[1], "top4": prefixes[3],
            "rank_score": math.fsum(prefixes) / cutoff}
