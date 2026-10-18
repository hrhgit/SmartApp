"""Small auditable online multiclass baselines with optional metadata semantics.

These standard-library implementations support fractional sample weights and
classes/features first observed in mature training records. Four independent
heads separate cleaned first-target supervision (0) from fixed future steps
(1..3). The replay caller owns maturity, outcome deduplication, and annotations.
No candidate list, target inventory, future class vocabulary, or query truth is
read. predict() returns a normalized distribution over that head's learned labels.
"""
from collections import Counter
import math


VERSION = 'semantic-online-classifier-v1'
FEATURE_SPEC = {'current_origin': 1., 'current_app': .25, 'current_kind': .1,
                'bias': 1., 'semantic_total_l1_budget': .5,
                'semantic_tag_value': 'strength * confidence; duplicate tags use max',
                'semantic_budget_normalization': 'divide by max(1, sum raw strengths)'}


def _number(value, name, minimum=0., strict=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < minimum or (strict and value == minimum)):
        raise ValueError(name + ' must be finite and ' + ('greater than ' if strict else 'at least ') + str(minimum))
    return float(value)


def _step(step):
    if isinstance(step, bool) or not isinstance(step, int) or step not in (0, 1, 2, 3):
        raise ValueError('step must be 0, 1, 2, or 3')
    return step


def features(q, semantic=False, semantic_budget=.5):
    """Origin has weight 1; ALL semantic tags together have weight at most .5.

    Generic kind/site/extension tags keep their supplied weak strengths. A lone
    weak tag is never normalized upward to full semantic strength. The intercept
    is not a metadata similarity signal. Exact destination identities stay labels.
    """
    origin, app = q.get('origin'), q.get('app')
    if not isinstance(origin, str) or not origin:
        raise ValueError('A known current origin is required')
    budget = _number(semantic_budget, 'semantic_budget')
    if budget > .5:
        raise ValueError('semantic_budget cannot exceed the current-interface .5 L1 cap')
    kind = 'window_class' if origin.startswith('window_class:') else origin.split(':', 1)[0]
    x = {'origin=' + origin: 1., 'kind=' + kind: .1, 'bias': 1.}
    if isinstance(app, str) and app:
        x['app=' + app] = .25
    if not semantic or not budget:
        return x
    annotation = q.get('annotation') or {}
    if not isinstance(annotation, dict):
        raise ValueError('annotation must be an observed metadata annotation dictionary')
    raw = {}
    for tag in annotation.get('tags', ()):
        if not isinstance(tag, dict) or not isinstance(tag.get('key'), str) or not tag['key']:
            raise ValueError('Semantic tags require a nonempty key')
        strength = _number(tag.get('strength', 0.), 'tag strength')
        confidence = _number(tag.get('confidence', 0.), 'tag confidence')
        if strength > 1 or confidence > 1:
            raise ValueError('Semantic strength/confidence must not exceed one')
        raw[tag['key']] = max(raw.get(tag['key'], 0.), strength * confidence)
    divisor = max(1., sum(raw.values()))
    x.update({'semantic=' + key: budget * value / divisor for key, value in raw.items() if value > 0})
    return x


def _probabilities(logits):
    if not logits:
        return {}
    anchor = max(logits.values())
    if not math.isfinite(anchor):
        raise ValueError('Nonfinite classifier score')
    exponential = {key: math.exp(value - anchor) for key, value in logits.items()}
    total = sum(exponential.values())
    return {key: value / total for key, value in exponential.items()}


class OnlineSoftmax:
    """Weighted one-example SGD on multiclass cross-entropy, optional L2.

    For each learned class c: W_c += lr * weight * (1[c=y]-P(c|x))*x.
    L2 adds -lr*weight*l2*W to non-intercept coefficients. New classes begin at
    zero; old records are not replayed when they appear. Consequently an initial
    one-class period has zero cross-entropy gradient, an explicit limitation of
    this dynamic-vocabulary baseline rather than hidden retrospective fitting.
    """
    def __init__(self, lr=.3, l2=0., semantic=False, semantic_budget=.5):
        self.lr = _number(lr, 'lr', strict=True)
        self.l2 = _number(l2, 'l2')
        self.semantic, self.semantic_budget = bool(semantic), semantic_budget
        self.heads = {step: {} for step in range(4)}
        self.training_mass = {step: Counter() for step in range(4)}

    def learn(self, q, label, weight=1., step=1):
        head = self.heads[_step(step)]
        weight = _number(weight, 'weight')
        if not isinstance(label, str) or not label:
            raise ValueError('Training label must be a nonempty observed semantic target')
        if weight == 0:
            return
        x = features(q, self.semantic, self.semantic_budget)
        head.setdefault(label, {})
        probabilities = _probabilities({key: sum(row.get(k, 0.) * v for k, v in x.items())
                                       for key, row in sorted(head.items())})
        rate = self.lr * weight
        for key, row in head.items():
            gradient = (1. if key == label else 0.) - probabilities[key]
            if self.l2:
                for feature in row:
                    if feature != 'bias':
                        row[feature] *= 1. - rate * self.l2
            for feature, value in x.items():
                updated = row.get(feature, 0.) + rate * gradient * value
                if not math.isfinite(updated):
                    raise ValueError('Nonfinite online Softmax coefficient')
                row[feature] = updated
        self.training_mass[step][label] += weight

    def predict(self, q, step=1):
        head = self.heads[_step(step)]
        x = features(q, self.semantic, self.semantic_budget)
        return _probabilities({key: sum(row.get(k, 0.) * v for k, v in x.items())
                               for key, row in sorted(head.items())})


class OnlineMultinomialNB:
    """Fractional class/feature counts with additive smoothing and dynamic vocab.

    Each update adds weight to the observed class and weight*x_j to its feature
    counts. Prediction ignores never-trained feature keys. Alpha smooths both
    class priors and feature likelihoods, using only this head's learned classes
    and feature vocabulary. These are generative comparison scores, not an
    assertion that correlated metadata tags are conditionally independent.
    """
    def __init__(self, alpha=1., semantic=False, semantic_budget=.5):
        self.alpha = _number(alpha, 'alpha', strict=True)
        self.semantic, self.semantic_budget = bool(semantic), semantic_budget
        self.heads = {step: {'mass': Counter(), 'features': {}, 'tokens': Counter(), 'vocab': set()}
                      for step in range(4)}

    def learn(self, q, label, weight=1., step=1):
        head = self.heads[_step(step)]
        weight = _number(weight, 'weight')
        if not isinstance(label, str) or not label:
            raise ValueError('Training label must be a nonempty observed semantic target')
        if weight == 0:
            return
        x = features(q, self.semantic, self.semantic_budget)
        head['mass'][label] += weight
        row = head['features'].setdefault(label, Counter())
        head['vocab'].update(x)
        for feature, value in x.items():
            row[feature] += weight * value
            head['tokens'][label] += weight * value

    def predict(self, q, step=1):
        head = self.heads[_step(step)]
        x = features(q, self.semantic, self.semantic_budget)
        if not head['mass']:
            return {}
        known = {key: value for key, value in x.items() if key in head['vocab']}
        total = sum(head['mass'].values()) + self.alpha * len(head['mass'])
        vocabulary = len(head['vocab'])
        scores = {}
        for label, mass in sorted(head['mass'].items()):
            denominator = head['tokens'][label] + self.alpha * vocabulary
            row = head['features'][label]
            scores[label] = math.log((mass + self.alpha) / total) + sum(
                value * math.log((row[key] + self.alpha) / denominator)
                for key, value in known.items())
        return _probabilities(scores)


def build(config):
    family = config['family']
    options = {'semantic': config.get('semantic', False),
               'semantic_budget': config.get('semantic_budget', .5)}
    if family == 'softmax':
        return OnlineSoftmax(lr=config.get('lr', .3), l2=config.get('l2', 0.), **options)
    if family == 'multinomial_nb':
        return OnlineMultinomialNB(alpha=config.get('alpha', 1.), **options)
    raise ValueError('Unknown online classifier family: ' + str(family))
