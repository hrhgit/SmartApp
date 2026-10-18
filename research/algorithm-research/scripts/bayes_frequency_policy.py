"""Offline hierarchical decisions and frequent-destination miss-cost hypotheses.

Frequencies are causal sufficient statistics, not observed recovery effort.
The supplied ranking model probabilities remain estimates, not a calibrated
posterior. The empirical-Bayes child distribution preserves unknown mass.
"""
from collections import Counter
import math
import numpy as np

import evaluation_contract as scoring

UNKNOWN = 'unobserved:other_destination'
UNKNOWN_CHILD = 'unobserved:chatgpt_child'
PRIOR_STRENGTH = 2.
MODES = ('fixed', 'global_frequency', 'context_frequency')


def frequency_costs(outcomes, global_counts, local_counts, mode):
    """Add-one global smoothing; one global pseudo-arrival for sparse context.

    Positive weights have arithmetic mean one over this same outcome space.
    Scaling has no effect on decisions. No loss weights are fitted to test hits.
    """
    outcomes = list(outcomes)
    if mode not in MODES: raise ValueError('Unknown decision hypothesis')
    if not outcomes: return {}
    if mode == 'fixed': return {y: 1. for y in outcomes}
    global_total = sum(global_counts.get(y, 0.)+1 for y in outcomes)
    base = {y: (global_counts.get(y, 0.)+1)/global_total for y in outcomes}
    if mode == 'context_frequency':
        total = sum(local_counts.get(y, 0.) for y in outcomes)
        base = {y: (local_counts.get(y, 0.)+base[y])/(total+1.) for y in outcomes}
    return {y: value*len(outcomes) for y, value in base.items()}


def expand_service_probabilities(scores, aliases, entries, global_counts, local_counts, unseen_probability):
    """Preserve root model probabilities; expand only the agreed service group.

    Concrete child frequencies are never read from the evaluation labels.
    Unknown child has one pseudo-arrival in the empirical global prior. Local
    context counts update a Dirichlet prior of fixed total concentration two.
    """
    if not 0 < unseen_probability < 1: raise ValueError('Expected open-set mass in (0,1)')
    roots = Counter()
    for label, value in scores.items():
        roots[aliases.get(label, label)] += value
    total = sum(roots.values())
    if not total: return {UNKNOWN: 1.}, []
    posterior = {UNKNOWN: unseen_probability}; actions = []
    for root, mass in roots.items():
        probability = mass/total*(1-unseen_probability)
        if root != scoring.SERVICE:
            posterior[root] = posterior.get(root, 0.)+probability; actions.append(root)
            continue
        children = sorted(y for y in global_counts if aliases.get(y) == root)
        global_total = sum(global_counts[y] for y in children)+1.
        local_total = sum(local_counts.get(y, 0.) for y in children)
        for child in children:
            prior = global_counts[child]/global_total
            conditional = (local_counts.get(child, 0.)+PRIOR_STRENGTH*prior)/(local_total+PRIOR_STRENGTH)
            posterior[child] = probability*conditional
            # Root is one executable homepage action, not another outcome count.
            if child not in entries: actions.append(child)
        posterior[UNKNOWN_CHILD] = probability*(PRIOR_STRENGTH/global_total)/(local_total+PRIOR_STRENGTH)
        actions.append(scoring.SERVICE)
    assert math.isclose(sum(posterior.values()), 1., abs_tol=1e-10)
    return posterior, list(dict.fromkeys(actions))


def risk_menu(posterior, actions, aliases, entries, costs, size=4):
    """Exact minimum-risk first action, then greedy marginal menu improvement.

    Set loss assumes an ideal user can choose the least costly offered action.
    Later slots are an approximation, not a globally optimal TopK proof.
    """
    if size < 1 or not actions: return [], []
    first = scoring.minimum_risk_action(posterior, actions, aliases, entries, costs)
    outcomes = list(posterior); actions = list(dict.fromkeys(actions))
    matrix = np.array([[scoring.decision_loss(a, y, aliases, entries) for y in outcomes] for a in actions])
    weights = np.array([posterior[y]*costs[y] for y in outcomes])
    chosen = actions.index(first['action']); selected = [chosen]
    current = matrix[chosen]; risks = [float(current @ weights)]
    while len(selected) < min(size, len(actions)):
        risks_if_added = np.minimum(matrix, current).dot(weights)
        remaining = [i for i in range(len(actions)) if i not in selected]
        chosen = min(remaining, key=lambda i: (risks_if_added[i], actions[i] in entries, i))
        selected.append(chosen); current = np.minimum(current, matrix[chosen])
        risks.append(float(current @ weights))
    return [actions[i] for i in selected], risks


def calibration_summary(rows, distributions, aliases_by_position, entries_by_position):
    """Probability diagnostics on the declared partition, including unknown bins.

    Mapping an unseen truth to its unknown bucket is evaluation only. Exact
    missing-target counts are retained; this never grants a menu hit.
    """
    brier = nll = missing = 0.; bins = {i: [] for i in range(5)}; n = 0
    for row in rows:
        if not scoring.eligible(row): continue
        pos = row['position']; p = distributions[pos]; truth = row['truth_action']
        missing += truth not in p
        bucket = truth if truth in p else (UNKNOWN_CHILD if UNKNOWN_CHILD in p and
            row['_scoring']['aliases'].get(truth) == scoring.SERVICE else UNKNOWN)
        target_p = p.get(bucket, 0.)
        nll -= math.log(max(target_p, 1e-15))
        brier += sum(v*v for v in p.values())+1-2*target_p
        best = max(p, key=p.get); confidence = p[best]
        bins[min(4, int(confidence*5))].append((confidence, best == bucket)); n += 1
    details = [{'n': len(v), 'mean_confidence': sum(x for x, _ in v)/len(v),
                'accuracy': sum(y for _, y in v)/len(v)} for v in bins.values() if v]
    return {'n': n, 'multiclass_brier': brier/n if n else None, 'nll_floor_1e15': nll/n if n else None,
            'truth_missing_n': int(missing), 'confidence_bins': details,
            'scored_partition': 'known exact targets plus unknown-root and unknown-service-child buckets',
            'ece_5bins': sum(r['n']*abs(r['mean_confidence']-r['accuracy']) for r in details)/n if n else None,
            'probabilities_calibrated': False}
