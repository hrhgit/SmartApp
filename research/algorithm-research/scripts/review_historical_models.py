"""Re-score historical forecasts and replay legacy families on one audited dataset.

This is a development-data audit, not a fresh holdout or deployment command.
Saved forecasts are never silently filled, re-ranked, or extended to new cutoffs.
Legacy learners are explicitly adapted to current-only, unrestricted destinations.
"""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import sys
import time

import evaluation_contract as scoring

ROOT = Path.home() / 'Library/Application Support/SmartApp/training'
DATASET = 'multibackground-v1-20260923-dataset-02'
REFERENCE = 'bayes-frequency-20260923-experiment-03'
LEGACY = 'reopenable-action-cleaning-v1-20260923-sweep-03'
FULL_RUNS = (
    'context-statistics-v1-20260923-tuning-02',
    'multibackground-v1-20260923-experiment-01',
    'temporal-fusion-v1-20260923-experiment-02',
    'statistical-residual-top2-v1-20260923-experiment-01',
    'joint-ranking-alternatives-v1-20260923-experiment-01', REFERENCE,
)
OLD_RUNS = (
    'navigation-hierarchy-v2-20260923-evaluation-01',
    'temporal-context-v1-20260923-experiment-01',
    'no-candidate-restriction-v1-20260923-control-01',
)
EARLY_RUNS = (
    'context-clusters-v1-20260923-experiment-01',
    'context-clusters-v2-20260923-experiment-01',
    'context-rules-v1-20260923-experiment-01',
    'context-rules-v1-20260923-experiment-02',
    'evidence-model-v1-20260923-experiment-01',
    'ordered-current-v1-20260923-evaluation-01',
    'dwell-tuning-v1-20260923-search-01',
    'dwell-cleaning-v1-20260923-comparison-01',
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with (gzip.open(path, 'wt') if str(path).endswith('.gz') else path.open('w')) as f:
        json.dump(value, f, ensure_ascii=False, indent=None if str(path).endswith('.gz') else 2)
    path.chmod(0o600)


class Sources:
    def __init__(self):
        self.accessed = {}
        self.manifests = {}

    def check(self, path):
        path = path.resolve()
        digest = sha(path)
        manifest = path.parent/'artifact-hashes.json'
        if manifest.exists():
            if manifest not in self.manifests:
                self.manifests[manifest] = json.loads(manifest.read_text())
                self.accessed[str(manifest)] = sha(manifest)
            expected = self.manifests[manifest].get(path.name)
            if expected is not None and digest != expected:
                raise ValueError('Historical artifact changed: '+str(path))
        self.accessed[str(path)] = digest
        return path

    def read(self, path):
        path = self.check(path)
        with (gzip.open(path, 'rt') if path.suffix == '.gz' else path.open()) as f:
            return json.load(f)

    def verify_unchanged(self):
        for path, expected in self.accessed.items():
            if sha(Path(path)) != expected:
                raise ValueError('Source changed during review: '+path)


def prediction_file(run):
    return next(p for p in (run/'predictions-private.json.gz', run/'predictions-private.json') if p.exists())


def normalized_output(value):
    if isinstance(value, list):
        return {'predictions': value}
    if 'predictions' in value:
        return {k: value[k] for k in ('predictions', 'rankable') if k in value}
    if 'single' in value:
        return {'predictions': value['single']}
    raise ValueError('Unknown prediction schema')


def base_query(observation):
    """No evaluation metadata, background inventory, or ordered past features."""
    origin, app = observation['origin'], observation['app']
    kind = 'window_class' if origin.startswith('window_class:') else origin.split(':', 1)[0]
    return {'origin': origin, 'app': app, 'timestamp': observation['timestamp'],
            'x': {'origin='+origin: 1., 'app='+(app or ''): 1., 'kind='+kind: 1., 'bias': 1.}}


def navigation_label(label, observation):
    """The same action definition applies in supervision and inference."""
    identity = observation['annotation']['identity']
    return label != identity['id'] and label not in {
        'app:'+identity.get('bundleID', ''), 'window_class:'+identity.get('bundleID', '')}


def causal_stream(data, policy):
    from compare_semantic_trajectory import training_rows
    observations = data['observations']
    records, excluded = [], Counter()
    for row in training_rows(data, policy):
        if row['head'] != 0:
            continue
        q = observations[row['position']]
        if not navigation_label(row['label'], q):
            excluded['non_navigation'] += 1
            continue
        if q['index'] > row['ready_index'] or row['record']['target_evidence_index'] > row['ready_index']:
            raise ValueError('Training evidence not mature')
        if row['weight'] != 1.:
            raise ValueError('Legacy learner cannot silently ignore non-unit training weight')
        record = base_query(q)
        record.update(label=row['label'], outcome_id=row['outcome_id'],
                      outcome_timestamp=row['outcome_timestamp'])
        records.append({'position': row['position'], 'ready_index': row['ready_index'], 'record': record})
    return records, dict(excluded)


def build_model(config):
    import stat_model_suite as stat
    import online_model_suite as online
    from fused_online_models import FusedModel
    if config['family'] == 'rank_fusion_ensemble':
        p = config['parameters']
        return FusedModel(build_model(p['left']), build_model(p['right']), p['right_weight'], p['rank_offset'])
    if config['id'].startswith('stat_'):
        return stat.build(config)
    return online.build(config)


def replay_legacy(data, rows, config, policy):
    stream, excluded = causal_stream(data, policy)
    model = build_model(config)
    cursor, vocabulary, last_seen = 0, set(), {}
    outputs, per_cutoff, timing = {}, {}, []
    for row in sorted(rows, key=lambda r: r['position']):
        pos = row['position']; observation = data['observations'][pos]
        while cursor < len(stream) and stream[cursor]['ready_index'] < observation['index']:
            item = stream[cursor]; record = item['record']
            if record['outcome_timestamp'] >= observation['timestamp']:
                raise ValueError('Future outcome entered learner')
            model.learn(record)
            vocabulary.add(record['label'])
            last_seen[record['label']] = max(last_seen.get(record['label'], -math.inf), record['outcome_timestamp'])
            cursor += 1
        query = base_query(observation)
        query.update(labels=sorted(vocabulary), candidate_last_seen=dict(last_seen))
        started = time.perf_counter()
        scores = model.scores(query)
        timing.append((time.perf_counter()-started)*1000)
        if not set(scores) <= vocabulary or any(not math.isfinite(v) for v in scores.values()):
            raise ValueError('Invalid output or future vocabulary')
        ordered = sorted(scores, key=lambda k: (-scores[k], k))
        # This is target-change semantics, never an inventory or same-origin gate.
        ordered = [label for label in ordered if navigation_label(label, observation)]
        outputs[pos] = {'predictions': ordered[:4], 'rankable': ordered}
        per_cutoff[pos] = {'mature_updates': cursor, 'vocabulary_n': len(vocabulary),
                           'input_index': observation['index'],
                           'last_ready_index': stream[cursor-1]['ready_index'] if cursor else None}
    # Stored state is AFTER the last evaluated cutoff; never score earlier rows with it.
    # The historical River KNN owns a locally defined search class. Cloudpickle
    # available with the sklearn/joblib stack preserves that class as well.
    try:
        import cloudpickle
    except ImportError:
        from joblib.externals import cloudpickle
    payload = cloudpickle.dumps({'model': model, 'vocabulary': vocabulary, 'last_seen': last_seen,
                            'training_cursor': cursor, 'config': config, 'policy': policy}, protocol=5)
    restored = pickle.loads(payload)['model']
    if restored.scores(query) != model.scores(query):
        raise ValueError('Model reload parity failed')
    if cursor < len(stream):
        probe = stream[cursor]['record']
        model.learn(probe); restored.learn(probe)
        future_query = {**query, 'timestamp': max(query['timestamp'], probe['outcome_timestamp'])+1,
                        'labels': sorted(vocabulary | {probe['label']})}
        if model.scores(future_query) != restored.scores(future_query):
            raise ValueError('Incremental resume parity failed')
    return outputs, payload, {'policy': policy, 'eligible_training_relations': len(stream),
        'learned_through_last_prediction': cursor, 'excluded_training': excluded,
        'reload_and_resume_parity': True, 'prediction_ms_max': max(timing, default=0),
        'cutoffs': per_cutoff}


class Review:
    def __init__(self, output):
        self.out = output
        self.sources = Sources()
        self.data = self.sources.read(ROOT/DATASET/'dataset-private.json.gz')
        self.rows = scoring.prepare_evaluation(self.data, output)
        self.by_sample = {r['sample_id']: r for r in self.rows}
        self.by_position = {r['position']: r for r in self.rows}
        self.models, self.predictions, self.exceptions = {}, {}, []
        self.window_actions = {}
        # Evaluation-only identity normalization. Never passed to a learner.
        for q in self.data['observations']:
            self.remember_window(q['target'], q.get('session'))
        for stream in self.data['streams'].values():
            for item in stream:
                self.remember_window(item['record']['target'], self.data['observations'][item['position']].get('session'))
        # Legacy forecasts may include a window never selected as a foreground
        # evaluation anchor. Resolve immutable ownership from the frozen event
        # catalog FOR SCORING ONLY, not to expand a predictor's as-of vocabulary.
        lineage = self.sources.read(ROOT/'multibackground-v1-20260923-experiment-01'/'run.json')
        events = self.sources.check(Path(lineage['snapshot'])/'events.jsonl')
        if self.sources.accessed[str(events)] != lineage['events_sha256']:
            raise ValueError('Frozen event snapshot mismatch')
        with events.open() as f:
            for line in f:
                event = json.loads(line)
                target = event.get('data', {}).get('target')
                if isinstance(target, dict):
                    self.remember_window(target, event.get('sessionID'))

    def remember_window(self, meta, session=None, key=None):
        if meta.get('kind') != 'window':
            return
        bundle = meta.get('bundleID') or meta.get('_bundle')
        if not bundle:
            return
        keys = [meta['id']]
        if session and not meta['id'].startswith('window@'):
            keys.append('window@'+session+':'+meta['id'])
        if key:
            keys.append(key)
        for k in keys:
            if k in self.window_actions and self.window_actions[k] != 'app:'+bundle:
                raise ValueError('Ambiguous window ownership')
            self.window_actions[k] = 'app:'+bundle

    def add(self, name, outputs, source, category, adapt_windows=False, extra=None):
        adapted, converted, unresolved = {}, 0, 0
        for key, raw in outputs.items():
            pos = int(key)
            if pos not in self.by_position:
                continue
            value = normalized_output(raw)
            if isinstance(raw, dict) and raw.get('sample_id') and raw['sample_id'] != self.by_position[pos]['sample_id']:
                raise ValueError('Cutoff identity mismatch')
            if adapt_windows:
                transformed = []
                for target in value['predictions']:
                    action = self.window_actions.get(target, target)
                    converted += action != target
                    unresolved += target.startswith('window@') and action == target
                    transformed.append(action)
                # Keep all actual slots, including duplicate parent actions; no refill.
                value['predictions'] = transformed
                value.pop('rankable', None)
            if len(value['predictions']) > 4:
                raise ValueError('Expected actual four-slot output')
            adapted[pos] = value
        if not adapted:
            self.exceptions.append({'name': name, 'reason': 'No common evaluation cutoff'})
            return
        if name in self.models:
            raise ValueError('Duplicate model name')
        missing = sorted(set(self.by_position)-adapted.keys())
        self.models[name] = {'source': source, 'category': category, 'forecast_n': len(adapted),
            'full_common_dataset': not missing, 'missing_positions': missing,
            'window_slots_mapped_to_app_action': converted, 'unresolved_window_slots': unresolved,
            **(extra or {})}
        self.predictions[name] = adapted

    def load_frozen(self, additional_runs=()):
        for run_name in FULL_RUNS+OLD_RUNS+tuple(additional_runs):
            run = ROOT/run_name
            values = self.sources.read(prediction_file(run))
            if all(str(k).isdigit() for k in values):
                values = {'original': values}
            for name, outputs in values.items():
                if name.endswith('_service'):
                    self.exceptions.append({'name': run_name+'/'+name,
                        'reason': 'Evaluation-only service-collapsed labels; score actual exact forecasts instead'})
                    continue
                self.add(run_name+'/'+name, outputs, run_name, 'frozen', adapt_windows=False)
        run_name = 'background-replacement-v1-20260923-diagnosis-01'
        for path in sorted((ROOT/run_name).glob('predictions-*-private.json')):
            for weight, outputs in self.sources.read(path).items():
                self.add(run_name+'/'+path.name.removeprefix('predictions-').removesuffix('-private.json')+'/'+weight,
                         outputs, run_name, 'frozen_background_ablation')
        run_name = 'semantic-trajectory-v1-20260923-comparison-01'
        for name, outputs in self.sources.read(ROOT/run_name/'selected-predictions-private.json').items():
            for head in ('single', 'future'):
                self.add(run_name+'/'+name+'_selected/'+head+'_menu',
                    {pos: {'predictions': record[head]} for pos, record in outputs.items()}, run_name, 'frozen_old',
                    extra={'historical_objective': name, 'evaluated_menu': head,
                           'current_evaluation': 'common next stable destination; not old multi-step coverage'})
        for path in sorted((ROOT/LEGACY).glob('best-predictions-*-private.json')):
            self.add(LEGACY+'/'+path.name.removeprefix('best-predictions-').removesuffix('-private.json'),
                     self.sources.read(path), LEGACY, 'historical_restricted')
        for run_name in EARLY_RUNS:
            values = self.sources.read(prediction_file(ROOT/run_name))['5']
            if isinstance(values, list):
                values = {'selected': values}
            for variant, records in values.items():
                outputs = {}
                for record in records:
                    pos = record['position']
                    if pos not in self.by_position:
                        continue
                    ref = self.by_position[pos]
                    if record.get('sample_id') and record['sample_id'] != ref['sample_id']:
                        raise ValueError('Historical sample identity mismatch')
                    if record.get('truth') and record['truth'] != ref['observed_outcome']['key']:
                        raise ValueError('Different historical scoring label; cannot pool')
                    meta = record.get('prediction_meta', record.get('predicted_meta', []))
                    if isinstance(meta, list):
                        for target, identity in zip(record['predictions'], meta):
                            if identity:
                                self.remember_window(identity, key=target)
                    outputs[pos] = record
                self.add(run_name+'/'+variant, outputs, run_name, 'historical_restricted', adapt_windows=True,
                         extra={'original_scoring_threshold_seconds': 5})
        path = ROOT/'context-knn-v1-20260923-evaluation-02/predictions-private.jsonl'
        models = defaultdict(dict)
        with self.sources.check(path).open() as f:
            for line in f:
                record = json.loads(line)
                ref = self.by_sample.get(record['sample_id'])
                if ref:
                    models[record['model']][ref['position']] = record
        for model, outputs in models.items():
            self.add('context-knn-v1-20260923-evaluation-02/'+model, outputs, path.parent.name,
                     'historical_different_training_labels', adapt_windows=True,
                     extra={'note': 'Original first-arrival predictor scored against common stable-destination labels; partial time coverage'})

    def replay(self):
        selection = self.sources.read(ROOT/LEGACY/'selection.json')
        configs = [deepcopy(v['config']) for v in selection['base_family_winners'].values()]
        for scope in ('conditional', 'global'):
            for family in ('frequency', 'recency'):
                c = {'id': 'stat_'+family+'_'+scope, 'family': family, 'parameters': {'scope': scope}}
                if not any(x['id'] == c['id'] for x in configs):
                    configs.append(c)
        fusion = deepcopy(selection['best_current_only']['config']['parameters']['base'])
        configs.append(fusion)
        plan = {'configs': configs, 'cleaning': 'hard_kind', 'current_sequence_length': 1,
                'output_vocabulary': 'all mature learned destinations; no inventory or same-origin gate',
                'observed_recency_adapter': 'last mature independent destination timestamp, not candidate inventory',
                'navigation_semantics': 'exclude current destination/app-only non-navigation in training and inference',
                'parameters': 'historical family representatives frozen before replay; no new search',
                'additional_cleaning': ['resolved'], 'fresh_holdout': False,
                'historical_selection_bias': True, 'scoring': scoring.contract()}
        write(self.out/'replay-plan.json', plan)
        jobs = [(c, 'hard_kind') for c in configs]
        jobs += [(c, 'resolved') for c in configs if c['family'] in (
            'count_candidate_recency', 'global_mixture', 'bayesian_smoothing', 'recency',
            'adaptive_hoeffding_tree', 'rank_fusion_ensemble')]
        for i, (config, policy) in enumerate(jobs):
            outputs, payload, audit = replay_legacy(self.data, self.rows, config, policy)
            name = 'modern_replay/'+policy+'/'+config['id']
            stem = 'replay-'+str(i).zfill(3)
            state = self.out/(stem+'-model-private.pkl')
            state.write_bytes(payload); state.chmod(0o600)
            write(self.out/(stem+'-causal-audit-private.json'), audit)
            self.add(name, outputs, LEGACY, 'modern_replay', extra={'config': config, 'cleaning': policy,
                'adapted_model_not_identical_old_pipeline': True, 'state_file': state.name,
                'state_trust': 'local artifact only; never load untrusted pickle',
                'reload_and_resume_parity': audit['reload_and_resume_parity']})
            print(json.dumps({'phase': 'causal_replay', 'completed': i+1, 'total': len(jobs),
                              'family': config['family'], 'cleaning': policy}), flush=True)

    def evaluate(self):
        metrics, paired = {}, {}
        reference = REFERENCE+'/existing_order'
        reference_predictions = self.predictions[reference]
        for name, outputs in self.predictions.items():
            rows = [r for r in self.rows if r['position'] in outputs]
            parts = scoring.cohorts(rows)
            parts['test_evidence'] = [r for r in parts['evidence_total'] if r['split'] != 'validation']
            parts['validation_evidence'] = [r for r in parts['evidence_total'] if r['split'] == 'validation']
            for grade in ('strong', 'repeated', 'single'):
                parts['old_strength_'+grade] = [r for r in parts['old_evidence']
                    if r.get('diagnosis_after_outcome', {}).get('grade') == grade]
            metrics[name] = {group: scoring.measure(members, outputs) for group, members in parts.items()}
            ref_metrics = scoring.measure(parts['evidence_total'], reference_predictions)
            scoring.require_comparable(metrics[name]['evidence_total'], ref_metrics)
            pair = {'accepted': scoring.paired_topk(parts['evidence_total'], reference_predictions, outputs),
                    'reference_on_same_rows': ref_metrics, 'exact': {}}
            for k in (1, 2, 4):
                delta = Counter()
                for row in parts['evidence_total']:
                    if not scoring.eligible(row):
                        continue
                    before = scoring.hits(row, scoring.prediction_at(reference_predictions, row['position']))['exact'][k]
                    after = scoring.hits(row, outputs[row['position']])['exact'][k]
                    delta['helped'] += after and not before; delta['hurt'] += before and not after
                pair['exact'][str(k)] = {**delta, 'net': delta['helped']-delta['hurt']}
            paired[name] = pair
        # Models in each cohort share the exact same member/label/scoring hash.
        full = [name for name, m in self.models.items() if m['full_common_dataset']]
        for group in metrics[full[0]]:
            scoring.require_comparable(*[metrics[name][group] for name in full])
        write(self.out/'metrics-private.json', metrics)
        write(self.out/'paired-private.json', paired)
        write(self.out/'models.json', self.models)
        write(self.out/'predictions-private.json.gz', self.predictions)
        write(self.out/'exceptions.json', self.exceptions)
        self.report(metrics, paired, full)
        return metrics

    def report(self, metrics, paired, full):
        def cell(m, k, exact=True):
            x = m['exact'] if exact else m
            return f"{x['top'+str(k)+'_hits']}/{x['n']}"
        def score(m):
            return f"{100*m['rank_score']['mean']:.2f}" if m['n'] else '—'
        def table(names, group='evidence_total'):
            lines = ['| 模型 | 精确Top1 | 精确Top2 | 精确Top4 | 接受Top4 | 综合分 | 返回覆盖 |',
                     '|---|---:|---:|---:|---:|---:|---:|']
            for name in names:
                m = metrics[name][group]
                coverage = f"{m['prediction_coverage']:.1%}" if m['n'] else '—'
                lines.append('| '+name+' | '+' | '.join([cell(m,1),cell(m,2),cell(m,4),cell(m,4,False),score(m),coverage])+' |')
            return lines
        def ordering(name):
            m = metrics[name]['evidence_total']
            return (-m['exact']['top4_hits'], -m['rank_score']['mean'], name)
        ordered = sorted(full, key=ordering)
        lines = ['# 历史模型统一重评与因果回放', '', scoring.report_scope(self.rows), '',
            '本次全部指标经当前 evaluation_contract.py 重新计算。精确命中收益1，正确首页回退0.6，错误0；综合分为Top1至Top4前缀最佳收益均值×100。服务接受命中不等于具体会话命中。普通窗口按既定应用动作标签评分，不宣称窗口实例命中。', '',
            f'恢复 {len(self.models)} 组模型/配置输出，其中 {len(full)} 组覆盖全部固定评测截点；其余只在明确共有子集比较，未补造缺失预测。', '',
            '表格按精确Top4降序、综合分次序排列，属于已看过的开发数据描述，不是重新选参或部署决定。相同输出的重复基线保留来源，不算独立支持证据。', '',
            '## 全部有证据样本：完整覆盖模型', '']
        lines += table(ordered)
        lines += ['', '## 新旧时段与较晚时段', '',
            '| 模型 | 旧证据精确Top4 | 新增证据精确Top4 | 较晚精确Top4 | 较晚综合分 | 全部合法综合分 |',
            '|---|---:|---:|---:|---:|---:|']
        for name in ordered:
            m = metrics[name]
            lines.append('| '+name+' | '+' | '.join([cell(m['old_evidence'],4), cell(m['new_evidence'],4),
                cell(m['test_evidence'],4), score(m['test_evidence']), score(m['all'])])+' |')
        lines += ['', '## 早期模型：固定旧样本或共有子集', '',
            '旧模型保留原先真正输出的槽位。窗口身份只在评分适配时映射到已知所属应用动作，不去重、不补位、不重排。缺失预测不会当成弃权；只在明确的共有样本上给分。旧KNN第一到达预测与现在稳定目的地标签不同，单独标记。', '']
        partial = [name for name in self.models if name not in full]
        lines += table(sorted(partial))
        lines += ['', '### 每个早期模型的同样本参考', '',
            '| 历史模型 | 共有合法样本 | 共有证据样本 | 模型精确Top4 | 参考精确Top4 | 模型综合分 | 参考综合分 |',
            '|---|---:|---:|---:|---:|---:|---:|']
        for name in sorted(partial):
            m = metrics[name]['evidence_total']; ref = paired[name]['reference_on_same_rows']
            lines.append('| '+name+f" | {metrics[name]['all']['n']} | {m['n']} | "+
                         ' | '.join([cell(m,4),cell(ref,4),score(m),score(ref)])+' |')
        lines += ['', '## 边界与复现', '',
            '- 冻结预测重评分与重新训练回放分别标记；没有用最终训练权重倒推历史预测。',
            '- 重新回放的旧家族共用 hard_kind 标签、成熟边界、独立行为去重和长度1当前输入；少量 resolved 清洗对照单列。标签词表来自成熟历史，不受库存或同起点支持集限制。',
            '- 旧 observed-recency 类的输入改为成熟历史独立到达的最近时间；属于明确适配版本，不冒称逐位复现旧候选模型。',
            '- 老家族参数来自既有历史家族代表，不在本次全量结果上搜索；其历史选择偏差仍存在。联合模型各随机种子全部保留，不只汇报最好种子。',
            '- early/late/new均为既有冻结数据；统一重评不会让它们成为未见测试。证据分组依赖真实后续目标，不能作为线上置信阈值。',
            '- 每模型详细新旧与证据强弱分组见 metrics-private.json；同样本修正/损失见 paired-private.json；预测、模型状态与成熟边界审计均保存在本目录。',
            '- 本轮不改变原始数据、历史run、采集设置、安装包或运行时模型。', '']
        (self.out/'report-private.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--skip-replay', action='store_true')
    parser.add_argument('--additional-run', action='append', default=[],
                        help='Additional frozen prediction run under the private training root')
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists() or out == ROOT.resolve() or not out.is_relative_to(ROOT.resolve()):
        parser.error('Use a new private training run')
    os.umask(0o077); sys.dont_write_bytecode = True; out.mkdir(mode=0o700)
    review = Review(out)
    if any(Path(name).name != name for name in args.additional_run):
        parser.error('Additional runs must be directory names, not paths')
    review.load_frozen(args.additional_run)
    print(json.dumps({'phase': 'frozen_loaded', 'models': len(review.models)}), flush=True)
    if not args.skip_replay:
        review.replay()
    review.evaluate()
    review.sources.verify_unchanged()
    for source in Path(__file__).parent.glob('*.py'):
        # Freeze transitive local research sources for independent replay.
        (out/('source-'+source.name)).write_bytes(source.read_bytes())
    import online_model_suite
    import importlib.metadata
    write(out/'run.json', {'complete': True, 'created_at': datetime.now(timezone.utc).isoformat(),
        'source_files': review.sources.accessed, 'dataset': str(ROOT/DATASET),
        'scoring_contract': scoring.contract(), 'models': len(review.models),
        'full_dataset_models': sum(m['full_common_dataset'] for m in review.models.values()),
        'runtime_changed': False, 'fresh_holdout': False, 'dependencies': online_model_suite.dependency_metadata(),
        'serialization_dependencies': {name: importlib.metadata.version(name) for name in ('joblib', 'cloudpickle')}})
    write(out/'artifact-hashes.json', {p.name: sha(p) for p in out.iterdir() if p.is_file()})
    assert all(p.stat().st_mode & 0o777 == 0o600 for p in out.iterdir() if p.is_file())
    print(json.dumps({'phase': 'complete', 'output': str(out), 'models': len(review.models)}), flush=True)


if __name__ == '__main__':
    main()
