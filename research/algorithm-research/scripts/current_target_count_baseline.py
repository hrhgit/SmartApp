"""Parameter-free current-target transition counts and an explicit global tie-break.

No semantic smoothing, recency, time decay, inventory, context features, gradient
updates, service-label folding, or tuned mixing weights. All mature labels remain
eligible; global independent-arrival counts order conditional-count ties, including
the zero-support/cold-start case. Scores are ranking keys, not calibrated posteriors.
"""
from collections import Counter, defaultdict
from copy import deepcopy

from review_historical_models import causal_stream, navigation_label
import evaluation_contract as scoring


NAME = 'baseline/current_target_counts_global_ties'
CONFIG = {'version': 'current-target-counts-v1',
    'ranking': ['conditional_count_desc', 'global_count_desc', 'stable_label_asc'],
    'conditional_count_unit': 'unique (current origin, independent outcome id)',
    'global_count_unit': 'unique independent outcome id',
    'cold_start': 'global counts when no conditional votes',
    'output_pool': 'all mature learned targets, excluding current destination and its unspecified owner',
    'features': 'current origin only', 'label_scope': 'original concrete actions and agreed window classes',
    'semantic_smoothing': False, 'time_decay': False, 'recency': False, 'gradient_updates': False,
    'hyperparameter_tuning': False, 'calibrated_posterior': False}


class CurrentTargetCounts:
    def __init__(self):
        self.conditional = defaultdict(Counter)
        self.global_counts = Counter()
        self.seen_conditional = {}
        self.seen_global = {}

    def learn(self, origin, label, outcome_id):
        key = (origin, outcome_id)
        if key in self.seen_conditional and self.seen_conditional[key] != label:
            raise ValueError('Conflicting conditional outcome identity')
        if outcome_id in self.seen_global and self.seen_global[outcome_id] != label:
            raise ValueError('Conflicting global outcome identity')
        if key in self.seen_conditional:
            return False
        self.seen_conditional[key] = label
        self.conditional[origin][label] += 1
        if outcome_id not in self.seen_global:
            self.seen_global[outcome_id] = label
            self.global_counts[label] += 1
        return True

    def ranked(self, origin):
        conditional = self.conditional.get(origin, {})
        return sorted(self.global_counts, key=lambda label:
                      (-conditional.get(label, 0), -self.global_counts[label], label))

    def state(self):
        return {'conditional': {k: dict(v) for k, v in self.conditional.items()},
                'global_counts': dict(self.global_counts),
                'seen_conditional': [[origin, outcome, label] for (origin, outcome), label in self.seen_conditional.items()],
                'seen_global': dict(self.seen_global)}

    @classmethod
    def restore(cls, state):
        model = cls()
        model.conditional = defaultdict(Counter, {k: Counter(v) for k, v in state['conditional'].items()})
        model.global_counts = Counter(state['global_counts'])
        model.seen_conditional = {(origin, outcome): label for origin, outcome, label in state['seen_conditional']}
        model.seen_global = dict(state['seen_global'])
        return model


def replay(data, rows, ready_limit=None):
    stream, excluded = causal_stream(data, 'hard_kind')
    if ready_limit is not None:
        stream = [r for r in stream if r['ready_index'] <= ready_limit]
    model = CurrentTargetCounts()
    outputs, audit, cursor = {}, [], 0
    for row in sorted(rows, key=lambda r: r['input_index']):
        pos = row['position']
        q = data['observations'][pos]
        while cursor < len(stream) and stream[cursor]['ready_index'] < q['index']:
            item = stream[cursor]
            record = item['record']
            if record['outcome_timestamp'] >= q['timestamp']:
                raise ValueError('Future outcome entered counts')
            model.learn(record['origin'], record['label'], record['outcome_id'])
            cursor += 1
        state_before = model.state()
        ranked = [label for label in model.ranked(q['origin']) if navigation_label(label, q)]
        outputs[pos] = {'predictions': ranked[:4], 'rankable': ranked}
        if state_before != model.state():
            raise ValueError('Prediction mutated learned state')
        ready = stream[cursor-1]['ready_index'] if cursor else -1
        if ready >= q['index'] or (ready_limit is not None and ready > ready_limit):
            raise ValueError('Maturity boundary violated')
        audit.append({'position': pos, 'input_index': q['index'], 'latest_training_ready_index': ready,
            'mature_relations': cursor, 'distinct_outcomes': len(model.seen_global),
            'vocabulary_n': len(model.global_counts), 'learned_state_sha256': scoring.digest(state_before),
            'origin_supported': bool(model.conditional.get(q['origin']))})
    restored = CurrentTargetCounts.restore(deepcopy(model.state()))
    for q in data['observations']:
        if model.ranked(q['origin']) != restored.ranked(q['origin']):
            raise ValueError('Reload parity failed')
    if ready_limit is not None and len({r['learned_state_sha256'] for r in audit}) != 1:
        raise ValueError('Frozen state changed after warmup')
    return outputs, model.state(), {'cutoffs': audit, 'excluded_training': excluded,
        'reload_parity': True, 'pure_read_only_prediction': True,
        'post_warmup_frozen': ready_limit is not None}
