"""Two-head incremental supervision from the same matured destination arrival.

The primary head learns the exact actionable destination. The application head
learns its explicitly observed owner, including owners of concrete pages/paths.
These are non-exclusive labels of the SAME outcome, not separate arrivals.
Prediction exposes only current candidate inputs, never destination_app.
"""
from collections import Counter
from copy import deepcopy
import gzip
import json
import math
from pathlib import Path

from evaluate_evidence_model import DENIED_BUNDLES
from fused_online_models import _rank_scores
from summarize_tuned_history import verify_run


QUERY_FIELDS = {'origin', 'app', 'timestamp', 'x', 'labels',
                'candidate_last_seen', 'candidate_key_order'}


def _canonical_owner(value):
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError('destination_app must be an observed bundle ID or app key')
    bundle = value[4:] if value.startswith('app:') else value
    if not bundle or bundle in DENIED_BUNDLES:
        return None
    return 'app:' + bundle


class HierarchicalActionModel:
    """RRF of a primary destination scorer and an independent application head.

    Both heads use the same current-interface features. Each consumes a matured
    (origin, outcome_id) at most once, preserving the existing replay identity.
    Different origins can supervise the same physical outcome without treating
    its window/page/app descriptions as additional arrivals within one origin.

    app_weight=0 returns primary scores exactly. Pages remain page scores; an
    application score does not inherit or replace any child's exact identity.
    """

    def __init__(self, base, app_model, app_weight, rank_offset=0):
        if base is app_model:
            raise ValueError('Heads must be independent model instances')
        self.app_weight = float(app_weight)
        self.rank_offset = float(rank_offset)
        if not math.isfinite(self.app_weight) or not 0 <= self.app_weight <= 1:
            raise ValueError('app_weight must be finite and between 0 and 1')
        if not math.isfinite(self.rank_offset) or self.rank_offset < 0:
            raise ValueError('rank_offset must be finite and nonnegative')
        self.base, self.app_model = base, app_model
        self.primary_outcomes, self.app_outcomes = {}, {}

    def learn(self, record):
        identity = record['origin'], record['outcome_id']
        owner = _canonical_owner(record.get('destination_app'))
        if identity in self.primary_outcomes and self.primary_outcomes[identity] != record['label']:
            raise ValueError('Conflicting primary labels for the same origin/outcome')
        if owner and identity in self.app_outcomes and self.app_outcomes[identity] != owner:
            raise ValueError('Conflicting application labels for the same origin/outcome')
        clean = {key: deepcopy(value) for key, value in record.items() if key != 'destination_app'}
        learned_primary = learned_app = False
        if identity not in self.primary_outcomes:
            self.base.learn(clean)
            self.primary_outcomes[identity] = record['label']
            learned_primary = True
        if owner and identity not in self.app_outcomes:
            auxiliary = {**deepcopy(clean), 'label': owner}
            self.app_model.learn(auxiliary)
            self.app_outcomes[identity] = owner
            learned_app = True
        return {'primary_learned': learned_primary, 'app_learned': learned_app}

    def scores(self, query):
        clean = {key: deepcopy(value) for key, value in query.items() if key in QUERY_FIELDS}
        primary = self.base.scores(clean)
        if self.app_weight == 0:
            return primary
        app_labels = [key for key in clean['labels'] if key.startswith('app:') and len(key) > 4]
        auxiliary_query = {**deepcopy(clean), 'labels': app_labels}
        for field in ('candidate_last_seen', 'candidate_key_order'):
            if field in auxiliary_query:
                auxiliary_query[field] = {key: value for key, value in auxiliary_query[field].items()
                                          if key in app_labels}
        auxiliary = {key: value for key, value in self.app_model.scores(auxiliary_query).items()
                     if key in app_labels}
        tie_keys = clean.get('candidate_key_order', {})
        left, right = _rank_scores(primary, tie_keys), _rank_scores(auxiliary, tie_keys)
        scores = {}
        for key in sorted(left.keys() | right.keys()):
            value = (1 - self.app_weight) / (self.rank_offset + left[key]) if key in left else 0.0
            value += self.app_weight / (self.rank_offset + right[key]) if key in right else 0.0
            scores[key] = value
        return scores


wrapper = HierarchicalActionModel


def augment_training_stream(stream, ledger):
    """Add destination_app from this exact cleaned arrival, never a catalog.

    Input positions, actual outcome IDs and maturity indices must agree with the
    frozen decision ledger. Missing/conflicting owners are explicitly unknown;
    neither the source application nor a later target record fills that gap.
    """
    result, counts = [], Counter()
    for row in stream:
        position = row['position']
        if not isinstance(position, int) or not 0 <= position < len(ledger):
            raise ValueError('Training position outside cleaning ledger')
        decision = ledger[position]
        if decision.get('position') != position:
            raise ValueError('Cleaning ledger position mismatch')
        record = row['record']
        if decision.get('target_event_id') != record['outcome_id']:
            raise ValueError('Cleaned actual outcome ID mismatch')
        if decision.get('label_ready_event_index') != row['ready_index']:
            raise ValueError('Cleaned label maturity mismatch')
        if (decision.get('input_through_event_index') is not None
                and decision['input_through_event_index'] != row['input_index']):
            raise ValueError('Cleaned input index mismatch')
        if not row['input_index'] < row['outcome_index'] <= row['ready_index']:
            raise ValueError('Invalid input/outcome/maturity order')
        if decision.get('training_included_after_dedup') is False:
            raise ValueError('Training row was excluded by the frozen cleaning ledger')
        target = decision.get('target')
        if not isinstance(target, dict):
            raise ValueError('Cleaned arrival has no target metadata')
        explicit, observed = target.get('bundleID'), target.get('_bundle')
        if explicit and observed and explicit != observed:
            owner, owner_source = None, 'conflicting_observed_owners'
        elif observed:
            owner, owner_source = _canonical_owner(observed), 'observed_bundle'
        elif explicit:
            owner, owner_source = _canonical_owner(explicit), 'target_bundle'
        else:
            owner, owner_source = None, 'missing_owner'
        kind = target.get('kind', 'unknown')
        expected_label = owner if kind == 'window' else target.get('id')
        if expected_label is not None and expected_label != record['label']:
            raise ValueError('Action training label does not match its cleaned target')
        augmented = deepcopy(row)
        augmented['record']['destination_app'] = owner
        result.append(augmented)
        counts['records'] += 1
        counts['with_owner' if owner else 'without_owner'] += 1
        counts['owner_source:' + owner_source] += 1
        counts['kind:' + kind] += 1
        if owner:
            counts['owner_kind:' + kind] += 1
    return result, dict(counts)


def load_augmented_streams(run, cleanings=None):
    """Read only a completed immutable run and verify its artifact manifest."""
    run = Path(run)
    provenance, hashes = verify_run(run)
    required = {'cleaning-audit-private.json.gz', 'training-streams-private.json.gz'}
    if not required <= hashes.keys():
        raise ValueError('Run does not freeze both cleaning ledger and training streams')
    with gzip.open(run / 'cleaning-audit-private.json.gz', 'rt', encoding='utf-8') as handle:
        ledgers = json.load(handle)['ledger_private']
    with gzip.open(run / 'training-streams-private.json.gz', 'rt', encoding='utf-8') as handle:
        streams = json.load(handle)
    chosen = list(streams) if cleanings is None else list(cleanings)
    output, audits = {}, {}
    for name in chosen:
        output[name], audits[name] = augment_training_stream(streams[name], ledgers[name])
    return output, {'source_run': str(run), 'source_artifact_hashes': hashes,
                    'source_version': provenance.get('version'), 'cleanings': audits,
                    'owner_source': 'only matching cleaned target at original maturity',
                    'query_features_changed': False}
