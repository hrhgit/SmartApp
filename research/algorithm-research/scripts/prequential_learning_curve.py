"""Warm up on independent mature outcomes, then replay all historical models.

This is retrospective prequential development evaluation, not a new holdout.
Frozen controls receive the same causal observed input, but no post-warmup
training labels, learned vocabulary, statistical memory, or parameter updates.
"""
import argparse
from bisect import bisect_left
from collections import Counter
from copy import copy, deepcopy
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil

import describe_dwell as dwell
import evaluate_context_knn as core
import evaluation_contract as scoring
from evaluate_dwell_cleaning import derive_label
from evaluate_evidence_model import DENIED_BUNDLES
from prepare_multibackground_dataset import evidence_for_new
from trajectory_supervision import TrajectoryEvents, action_key
from compare_semantic_trajectory import training_rows
from compare_fresh_models import Experiment, log
from complete_fresh_comparison import complete
from review_historical_models import ROOT, Sources, write, sha, navigation_label


def warmup_boundary(stream, count):
    if count < 1:
        raise ValueError('Warmup must contain independent outcomes')
    seen = set()
    for row in sorted(stream, key=lambda r: (r['ready_index'], r['position'])):
        seen.add(row['outcome_id'])
        if len(seen) >= count:
            return row['ready_index']
    raise ValueError('Not enough independent mature outcomes')


def head_zero(data):
    return [r for r in training_rows(data, 'hard_kind') if r['head'] == 0
            and navigation_label(r['label'], data['observations'][r['position']])]


def cap_training(data, ready_limit):
    # Leave observed inputs and evaluator-only labels untouched. All learner
    # entry points consume streams, never evaluation or reference truth.
    return {**data, 'streams': {name: [r for r in rows if r['ready_index'] <= ready_limit]
                              for name, rows in data['streams'].items()},
            'reference_stream': [r for r in data['reference_stream'] if r['ready_index'] <= ready_limit],
            'original_cleaning_stream': []}


def new_directory(path):
    path = path.resolve()
    if path.exists() or path == ROOT.resolve() or not path.is_relative_to(ROOT.resolve()):
        raise ValueError('Use a new private training run: '+str(path))
    os.umask(0o077)
    path.mkdir(mode=0o700)
    return path


def seal(out):
    for p in out.iterdir():
        if p.is_file():
            p.chmod(0o600)
    write(out/'artifact-hashes.json', {p.name: sha(p) for p in out.iterdir()
                                     if p.is_file() and p.name != 'artifact-hashes.json'})


def prepare(source, out, warmup=100):
    sources = Sources()
    data = sources.read(source/'dataset-private.json.gz')
    meta = sources.read(source/'run.json')
    objects = sources.read(source/'observation-objects-private.json.gz')
    events = core.load_lines(sources.check(Path(meta['snapshot'])/'events.jsonl'))
    samples = core.load_lines(sources.check(Path(meta['snapshot'])/'navigation-samples.jsonl'))
    visits, _, _ = dwell.extract_visits(events, samples, core.KNOWN_TEST_SESSIONS)
    lookup = TrajectoryEvents({e['id']: e for e in events}, {e['id']: i for i, e in enumerate(events)})
    by_visit = {v['arrival_event_id']: v for v in visits}
    stream = head_zero(data)
    boundary = warmup_boundary(stream, warmup)
    warm = [r for r in stream if r['ready_index'] <= boundary]
    evaluation, statuses = [], Counter()
    for pos, obj in enumerate(objects):
        q = core.Observation(**obj)
        scored, why = derive_label(q, by_visit, lookup, 5)
        statuses[why['status']] += 1
        if scored is None or scored.truth_meta.get('bundleID') in DENIED_BUNDLES:
            continue
        truth = action_key(scored.truth_meta)
        if truth is None or q.index <= boundary:
            continue
        evaluation.append({'position': pos, 'sample_id': q.sample_id, 'split': 'new_test',
            'truth_action': truth, 'truth_metadata': scored.truth_meta, 'input_index': q.index,
            'label_ready_index': q.ready_index, 'evidence_grade': 'unsupported', 'strict': q.strict})
    evidence_for_new(evaluation, data['observations'], data['reference_stream'])
    data = {**data, 'evaluation': evaluation}
    out = new_directory(out)
    rows = scoring.prepare_evaluation(data, out)
    parts = scoring.cohorts(rows)
    sizes = scoring.cohort_sizes({k: parts[k] for k in ('all', 'evidence_total')})
    plan = {'protocol': 'mature-prequential-warm100-v1', 'evaluation_scope': 'complete_two_cohorts',
        'warmup_requested_outcomes': warmup, 'warmup_actual_outcomes': len({r['outcome_id'] for r in warm}),
        'warmup_relations': len(warm), 'warmup_ready_index': boundary, 'boundary_event_index': boundary+1,
        'warmup_includes_entire_maturity_batch': True, 'sizes': sizes, 'candidate_n': len(rows),
        'rolling_window': 100, 'rolling_stride': 10, 'evidence_min_n': 10,
        'window_members': 'last 100 legal cutoffs; evidence is a subset of the same interval',
        'final_endpoint': 'append last full 100-row window if stride does not land on last row',
        'score': 'common rank_score.mean times 100; exact Top1 and Top4 separately',
        'training': 'all natural hard_kind mature feedback, strictly ready_index < input_index',
        'frozen': 'cap every supervised training head at warmup_ready_index; freeze learned priors and vocabulary too',
        'observed_context': 'same evolving causal inputs in both arms, including optional unordered time backgrounds',
        'evidence': 'same live-history diagnostic subset for both arms; not frozen-arm support or an online confidence gate',
        'historical_parameters_fixed': True, 'new_independent_holdout': False,
        'fixed_270_94_benchmark_unchanged': True, 'all_anchor_label_statuses': dict(statuses),
        'validation_range_utc': [core.utc(data['observations'][r['position']]['timestamp']) for r in (rows[0], rows[-1])],
        'scoring_contract': scoring.contract()}
    write(out/'dataset-private.json.gz', data)
    shutil.copyfile(sources.check(source/'observation-objects-private.json.gz'), out/'observation-objects-private.json.gz')
    write(out/'plan.json', plan)
    write(out/'summary.json', plan)
    write(out/'run.json', {**meta, **plan, 'created_at': datetime.now(timezone.utc).isoformat(),
        'source_dataset': str(source), 'source_files': sources.accessed})
    sources.verify_unchanged()
    seal(out)
    log(phase='prequential_dataset_frozen', **plan)


class Replay(Experiment):
    def __init__(self, dataset, output, arm):
        super().__init__(dataset, output)
        self.arm = arm
        if arm == 'frozen':
            self.data = cap_training(self.data, self.meta['warmup_ready_index'])
            self.stream = head_zero(self.data)
        self.plan.update(protocol=self.meta['protocol'], arm=arm, new_dataset_only=False,
            learning_ready_limit=self.meta['warmup_ready_index'] if arm == 'frozen' else None,
            training_relations=len(self.stream), training_stream_digest=scoring.digest(self.stream),
            warmup=self.scope, reuse_old_forecasts=False)
        write(self.out/'plan.json', self.plan)
        # A common explicit event-order ledger, independent of correctness.
        ready = [r['ready_index'] for r in self.stream]
        audit = []
        for row in self.rows:
            n = bisect_left(ready, row['input_index'])
            last = ready[n-1] if n else -1
            if last >= row['input_index'] or (arm == 'frozen' and last > self.meta['warmup_ready_index']):
                raise ValueError('Future or post-freeze learning')
            audit.append({'position': row['position'], 'input_index': row['input_index'],
                'score_ready_index': row['label_ready_index'], 'mature_head0_relations': n,
                'latest_training_ready_index': last})
        write(self.out/'chronology-audit-private.json', audit)

    def add(self, name, outputs, family, config=None, state=None, **extra):
        # Compact immutable per-model forecast checkpoints, written before any
        # aggregate scoring. Avoid repeatedly compressing every old rankable set.
        if name in self.outputs or set(map(int, outputs)) != self.positions:
            raise ValueError('Duplicate or incomplete model forecasts: '+name)
        values = {int(p): {'predictions': list(v['predictions'])} for p, v in outputs.items()}
        if any(len(v['predictions']) > 4 for v in values.values()):
            raise ValueError('Invalid menu')
        self.outputs[name] = values
        self.models[name] = {'family': family, 'config': config, 'state': state, 'arm': self.arm,
            'common_training_cleaning': 'hard_kind', 'full_common_validation': True, **extra}
        write(self.out/f'forecast-{len(self.models):03d}-private.json.gz', {'model': name, 'outputs': values})
        write(self.out/'models.json', self.models)

    def report(self, metrics, cross):
        # The old fixed-validation narrative is deliberately not used here.
        write(self.out/'protocol-summary.json', {'arm': self.arm, 'models': len(metrics),
            'scope': self.scope, 'fixed_benchmark_not_replaced': True})

    def finish(self):
        write(self.out/'predictions-private.json.gz', self.outputs)
        super().finish()


def run(dataset, output, arm):
    base = new_directory(output.with_name(output.name+'-base'))
    exp = Replay(dataset, base, arm)
    for method in ('legacy', 'statistics_and_examples', 'train_rankers', 'multi_background',
                   'temporal_models', 'early_context', 'semantics', 'decisions'):
        getattr(exp, method)()
    # Save before importing the additional 35 historical configurations.
    exp.finish()
    final = copy(exp)
    final.out = new_directory(output)
    final.sources = Sources()
    final.plan = deepcopy(exp.plan)
    complete(final, base)
    if len(final.outputs) != 182:
        raise ValueError('Historical model registry changed')
    final.evaluate()
    final.finish()


def rolling_intervals(n, width=100, stride=10):
    if width < 1 or stride < 1:
        raise ValueError('Window and stride must be positive')
    ends = list(range(width, n+1, stride))
    if n >= width and (not ends or ends[-1] != n):
        ends.append(n)
    return [(end-width, end) for end in ends]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('run')
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--arm', choices=('online', 'frozen'), required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.source.resolve(), args.output)
    else:
        run(args.dataset.resolve(), args.output.resolve(), args.arm)


if __name__ == '__main__':
    main()
