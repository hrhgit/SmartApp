#!/usr/bin/env python3
"""Causal shadow replay for a small online recurrent foreground sequence model.

This is research-only code.  It rebuilds each input sequence from confirmed
foreground arrivals, learns only after a label's maturity index, and writes
private artefacts only under SmartApp's training directory.  It neither changes
the collector nor installs a runtime recommendation model.

The comparison is intentionally narrow:

* ``CurrentOnlyFrequency`` is the single-step, unordered statistical control.
  It can see the current foreground origin and mature labels, but never an
  earlier ordered suffix.
* ``OnlineSequenceRNN`` is a small Elman RNN trained one mature example at a
  time by truncated back-propagation through the foreground-only suffix.

No multi-step statistical model is trained or selected by this runner.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import time

import evaluate_context_knn as io
import evaluation_contract as scoring
from compare_semantic_trajectory import training_rows
from summarize_tuned_history import verify_run
from temporal_context import build_contexts


VERSION = 'online-sequence-rnn-shadow-v1'
PRIVATE_ROOT = io.PRIVATE_ROOT
TRAINING_POLICY = 'hard_kind'
SEQUENCE_LIMIT = 5
ROLLING_WIDTH = 25
ROLLING_STRIDE = 25

# These are fixed before the private dataset is read.  The order is also the
# deterministic final tiebreaker after the documented validation metrics.
RNN_CONFIGS = [
    {'id': 'rnn_h8_w3_lr0.05', 'hidden': 8, 'window': 3, 'input_dim': 32,
     'learning_rate': .05, 'l2': 1e-5, 'seed': 4101},
    {'id': 'rnn_h12_w3_lr0.03', 'hidden': 12, 'window': 3, 'input_dim': 32,
     'learning_rate': .03, 'l2': 1e-5, 'seed': 4102},
    {'id': 'rnn_h12_w5_lr0.03', 'hidden': 12, 'window': 5, 'input_dim': 32,
     'learning_rate': .03, 'l2': 1e-5, 'seed': 4103},
    {'id': 'rnn_h16_w5_lr0.02', 'hidden': 16, 'window': 5, 'input_dim': 32,
     'learning_rate': .02, 'l2': 1e-5, 'seed': 4104},
]

SOURCE_FILES = (
    'online_sequence_rnn_shadow.py',
    'evaluation_contract.py',
    'temporal_context.py',
    'known_inactivity.py',
    'compare_semantic_trajectory.py',
    'summarize_tuned_history.py',
    'evaluate_context_knn.py',
)

SAFE_OBSERVATION_KEYS = (
    'position', 'index', 'ready_index', 'sample_id', 'anchor_id', 'session',
    'origin', 'timestamp', 'target',
)
SAFE_EVALUATION_KEYS = (
    'position', 'sample_id', 'truth_action', 'truth_metadata',
    'evidence_grade', 'split',
)
SAFE_STREAM_RECORD_KEYS = (
    'observed_step', 'action_key', 'weight', 'event_id', 'arrival_timestamp_ms', 'target',
)


def _require(mapping, keys, name):
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise ValueError(f'{name} missing required fields: {missing}')


def _finite(value, name, positive=False):
    try:
        value = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f'{name} must be finite') from error
    if not math.isfinite(value) or (positive and value <= 0):
        raise ValueError(f'{name} must be finite' + (' and positive' if positive else ''))
    return value


def _private_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    path.chmod(0o600)


def _private_gzip_json(path, value):
    with gzip.open(path, 'wt') as handle:
        json.dump(value, handle, ensure_ascii=False)
    path.chmod(0o600)


def _source_bytes():
    directory = Path(__file__).resolve().parent
    return {name: (directory / name).read_bytes() for name in SOURCE_FILES}


def _safe_data(frozen):
    """Project a frozen dataset onto the only fields this experiment may read.

    In particular, this does not propagate ``background_state``, candidate
    inventories, annotations, future-label bundles, or evaluation diagnosis to
    either predictor.  Targets and truth metadata remain only for the common
    scorer, never for a model input.
    """
    _require(frozen, ('observations', 'evaluation', 'streams'), 'dataset')
    if TRAINING_POLICY not in frozen['streams']:
        raise ValueError(f'Missing {TRAINING_POLICY} training stream')

    observations = []
    for row in frozen['observations']:
        _require(row, SAFE_OBSERVATION_KEYS, 'observation')
        observations.append({key: deepcopy(row[key]) for key in SAFE_OBSERVATION_KEYS})
    observations.sort(key=lambda row: row['position'])
    if [row['position'] for row in observations] != list(range(len(observations))):
        raise ValueError('Observation positions must be contiguous')
    if any(left['index'] >= right['index'] for left, right in zip(observations, observations[1:])):
        raise ValueError('Observation indices must be strictly increasing')

    evaluation = []
    for row in frozen['evaluation']:
        _require(row, SAFE_EVALUATION_KEYS, 'evaluation row')
        evaluation.append({key: deepcopy(row[key]) for key in SAFE_EVALUATION_KEYS})
    evaluation.sort(key=lambda row: row['position'])

    stream = []
    for row in frozen['streams'][TRAINING_POLICY]:
        _require(row, ('position', 'input_index', 'ready_index', 'record'), 'training row')
        _require(row['record'], SAFE_STREAM_RECORD_KEYS, 'training record')
        stream.append({
            'position': row['position'], 'input_index': row['input_index'],
            'ready_index': row['ready_index'],
            'record': {key: deepcopy(row['record'][key]) for key in SAFE_STREAM_RECORD_KEYS},
        })

    result = {'observations': observations, 'evaluation': evaluation,
              'streams': {TRAINING_POLICY: stream}}
    for row in result['observations']:
        unexpected = set(row) - set(SAFE_OBSERVATION_KEYS)
        if unexpected:
            raise ValueError('Unsafe observation projection: ' + repr(unexpected))
    return result


def sequence_inputs(observations, contexts):
    """Return a predictor allowlist built from the pre-cutoff temporal suffix."""
    result = {}
    for observation in observations:
        position = observation['position']
        context = contexts.get(position)
        if context is None:
            raise ValueError('Missing reconstructed sequence context')
        if context.get('through_index') != observation['index']:
            raise ValueError('Sequence context is not frozen at its prediction cutoff')
        sequence = tuple(context.get('sequence', ()))
        if not sequence or sequence[-1] != observation['origin']:
            raise ValueError('Foreground sequence must end at the current origin')
        if len(sequence) > SEQUENCE_LIMIT:
            raise ValueError('Temporal sequence exceeded its declared bound')
        if any(not isinstance(token, str) or not token for token in sequence):
            raise ValueError('Invalid foreground target key in sequence')
        result[position] = {
            'position': position,
            'index': observation['index'],
            'timestamp': observation['timestamp'],
            'origin': observation['origin'],
            'sequence': sequence,
            'sequence_length': len(sequence),
        }
    return result


def supervision_stream(data, inputs):
    """Use one mature, cleaned head-zero label per existing trajectory rule."""
    rows = [row for row in training_rows(data, TRAINING_POLICY) if row['head'] == 0]
    result, seen = [], set()
    weights = []
    for row in rows:
        if row['position'] not in inputs:
            raise ValueError('Training row does not have a safe input')
        query = inputs[row['position']]
        label = row['label']
        if not isinstance(label, str) or not label:
            raise ValueError('Training label must be a nonempty action key')
        if row['input_index'] != query['index'] or not row['input_index'] < row['ready_index']:
            raise ValueError('Invalid label maturity chronology')
        weight = _finite(row['weight'], 'training weight', positive=True)
        outcome_timestamp = _finite(row['outcome_timestamp'], 'outcome timestamp')
        identity = (query['origin'], row['outcome_id'])
        if identity in seen:
            raise ValueError('Duplicate mature origin/outcome reached the shared stream')
        seen.add(identity)
        result.append({
            'position': row['position'], 'input_index': row['input_index'],
            'ready_index': row['ready_index'], 'origin': query['origin'],
            'sequence': query['sequence'], 'label': label, 'outcome_id': row['outcome_id'],
            'timestamp': query['timestamp'], 'outcome_timestamp': outcome_timestamp,
            'weight': weight,
        })
        weights.append(weight)
    result.sort(key=lambda row: (row['ready_index'], row['input_index'], row['position'],
                                 str(row['outcome_id'])))
    if any(left['ready_index'] > right['ready_index'] for left, right in zip(result, result[1:])):
        raise ValueError('Training stream is not maturity ordered')
    return result, {
        'policy': TRAINING_POLICY,
        'head': 0,
        'mature_relations': len(result),
        'unique_origin_outcomes': len(seen),
        'weight_min': min(weights) if weights else None,
        'weight_max': max(weights) if weights else None,
        'weight_sum': math.fsum(weights),
        'stream_sha256': scoring.digest([
            [row['position'], row['input_index'], row['ready_index'], row['origin'],
             row['label'], row['outcome_id'], row['weight']]
            for row in result
        ]),
    }


class CurrentOnlyFrequency:
    """The explicit single-step baseline: current origin -> mature label count.

    Global counts only fill unused menu slots.  The model never receives an
    ordered earlier target, sequence length, prior target, time feature, or
    candidate inventory.
    """

    family = 'single_step_unordered_frequency'

    def __init__(self):
        self.conditional = defaultdict(Counter)
        self.global_counts = Counter()
        self.seen = set()
        self.updates = 0

    def learn(self, record):
        identity = (record['origin'], record['outcome_id'])
        if identity in self.seen:
            return False
        weight = _finite(record['weight'], 'training weight', positive=True)
        self.seen.add(identity)
        self.conditional[record['origin']][record['label']] += weight
        self.global_counts[record['label']] += weight
        self.updates += 1
        return True

    @staticmethod
    def _order(counts):
        return sorted(counts, key=lambda label: (-counts[label], label))

    def ranked(self, query):
        local = self._order(self.conditional.get(query['origin'], Counter()))
        global_ranked = self._order(self.global_counts)
        return local + [label for label in global_ranked if label not in set(local)]

    def export_state(self):
        return {
            'family': self.family,
            'updates': self.updates,
            'conditional_counts': {key: dict(value) for key, value in self.conditional.items()},
            'global_counts': dict(self.global_counts),
        }


class OnlineSequenceRNN:
    """Small dynamically-labelled Elman RNN updated with one mature example.

    It uses a deterministic hash encoder for foreground action keys.  Output
    classes are created only when their first mature label is learned.  The
    softmax values are ranking scores for this experiment, not calibrated
    posterior probabilities.
    """

    family = 'online_elman_rnn'

    def __init__(self, config):
        self.config = dict(config)
        self.hidden = int(config['hidden'])
        self.window = int(config['window'])
        self.input_dim = int(config['input_dim'])
        self.learning_rate = _finite(config['learning_rate'], 'learning rate', positive=True)
        self.l2 = _finite(config['l2'], 'l2')
        if self.hidden < 1 or self.window < 1 or self.window > SEQUENCE_LIMIT or self.input_dim < 4:
            raise ValueError('Invalid RNN dimensions')
        random_source = random.Random(config['seed'])
        input_scale = 1.0 / math.sqrt(self.input_dim)
        recurrent_scale = 1.0 / math.sqrt(self.hidden)
        self.wx = [[random_source.uniform(-input_scale, input_scale) for _ in range(self.input_dim)]
                   for _ in range(self.hidden)]
        self.wh = [[random_source.uniform(-recurrent_scale, recurrent_scale) for _ in range(self.hidden)]
                   for _ in range(self.hidden)]
        self.bh = [0.0] * self.hidden
        self.wy = {}
        self.by = {}
        self.labels = []
        self.seen = set()
        self.updates = 0

    def _encode(self, token):
        """Two signed hash features; no token inventory is retained by the model."""
        digest = hashlib.sha256(token.encode('utf-8')).digest()
        vector = [0.0] * self.input_dim
        for offset in (0, 4):
            value = int.from_bytes(digest[offset:offset + 4], 'big')
            index = value % self.input_dim
            vector[index] += .5 if (value >> 31) else -.5
        return vector

    def _forward(self, sequence):
        sequence = tuple(sequence)[-self.window:]
        if not sequence:
            raise ValueError('RNN needs a nonempty foreground sequence')
        hidden = [0.0] * self.hidden
        cache = []
        for token in sequence:
            x = self._encode(token)
            pre = [self.bh[i]
                   + math.fsum(self.wx[i][j] * x[j] for j in range(self.input_dim))
                   + math.fsum(self.wh[i][j] * hidden[j] for j in range(self.hidden))
                   for i in range(self.hidden)]
            next_hidden = [math.tanh(value) for value in pre]
            cache.append((x, hidden, next_hidden))
            hidden = next_hidden
        return hidden, cache

    def _ensure_label(self, label):
        if label not in self.wy:
            self.labels.append(label)
            self.wy[label] = [0.0] * self.hidden
            self.by[label] = 0.0

    def _scores(self, sequence):
        if not self.labels:
            return {}
        hidden, _ = self._forward(sequence)
        return {label: math.fsum(weight * value for weight, value in zip(self.wy[label], hidden))
                + self.by[label] for label in self.labels}

    def ranked(self, query):
        scores = self._scores(query['sequence'])
        return sorted(scores, key=lambda label: (-scores[label], label))

    def learn(self, record):
        identity = (record['origin'], record['outcome_id'])
        if identity in self.seen:
            return False
        label = record['label']
        weight = _finite(record['weight'], 'training weight', positive=True)
        self._ensure_label(label)
        hidden, cache = self._forward(record['sequence'])
        logits = [math.fsum(a * b for a, b in zip(self.wy[key], hidden)) + self.by[key]
                  for key in self.labels]
        maximum = max(logits)
        exponentials = [math.exp(max(-700., value - maximum)) for value in logits]
        denominator = math.fsum(exponentials)
        if denominator <= 0 or not math.isfinite(denominator):
            raise ValueError('Invalid RNN softmax denominator')
        probabilities = [value / denominator for value in exponentials]
        output_gradients = {
            key: weight * (probability - float(key == label))
            for key, probability in zip(self.labels, probabilities)
        }

        # Compute the hidden gradient with the pre-update output layer.
        gradient = [math.fsum(output_gradients[key] * self.wy[key][i] for key in self.labels)
                    for i in range(self.hidden)]
        for key in self.labels:
            output = self.wy[key]
            delta = output_gradients[key]
            for i in range(self.hidden):
                output[i] -= self.learning_rate * (delta * hidden[i] + self.l2 * output[i])
            self.by[key] -= self.learning_rate * delta

        # Truncated BPTT is naturally bounded by the configured foreground suffix.
        for x, previous, current in reversed(cache):
            pre_gradient = [max(-5., min(5., gradient[i] * (1. - current[i] * current[i])))
                            for i in range(self.hidden)]
            previous_gradient = [math.fsum(self.wh[i][j] * pre_gradient[i]
                                            for i in range(self.hidden))
                                 for j in range(self.hidden)]
            for i in range(self.hidden):
                for j in range(self.input_dim):
                    self.wx[i][j] -= self.learning_rate * (pre_gradient[i] * x[j]
                                                            + self.l2 * self.wx[i][j])
                for j in range(self.hidden):
                    self.wh[i][j] -= self.learning_rate * (pre_gradient[i] * previous[j]
                                                            + self.l2 * self.wh[i][j])
                self.bh[i] -= self.learning_rate * pre_gradient[i]
            gradient = previous_gradient

        self.seen.add(identity)
        self.updates += 1
        return True

    def export_state(self):
        return {
            'version': VERSION,
            'family': self.family,
            'config': self.config,
            'updates': self.updates,
            'labels': list(self.labels),
            'wx': self.wx,
            'wh': self.wh,
            'bh': self.bh,
            'wy': self.wy,
            'by': self.by,
        }


def timing_summary(values_ns):
    values = [value / 1_000_000 for value in values_ns]
    if not values:
        return {'n': 0, 'min_ms': None, 'median_ms': None, 'p95_ms': None,
                'max_ms': None, 'total_ms': 0.0}
    ordered = sorted(values)
    p95 = ordered[math.ceil(.95 * len(ordered)) - 1]
    return {
        'n': len(values), 'min_ms': min(values), 'median_ms': statistics.median(values),
        'p95_ms': p95, 'max_ms': max(values), 'total_ms': math.fsum(values),
    }


def causal_replay(inputs, stream, positions, model):
    """Predict first, then only learn feedback mature before each cutoff."""
    ordered_positions = sorted(positions, key=lambda position: inputs[position]['index'])
    cursor, outputs, updates_ns, predicts_ns, ledger = 0, {}, [], [], []
    for position in ordered_positions:
        query = inputs[position]
        latest_ready, matured = None, 0
        while cursor < len(stream) and stream[cursor]['ready_index'] < query['index']:
            record = stream[cursor]
            if not (record['position'] < position and record['input_index'] < query['index']
                    and record['ready_index'] < query['index']):
                raise ValueError('Future or same-cutoff feedback entered online learning')
            started = time.perf_counter_ns()
            learned = model.learn(record)
            updates_ns.append(time.perf_counter_ns() - started)
            if not learned:
                raise ValueError('Shared stream unexpectedly contains a duplicate update')
            latest_ready = record['ready_index']
            matured += 1
            cursor += 1
        started = time.perf_counter_ns()
        ranked = model.ranked(query)
        predicts_ns.append(time.perf_counter_ns() - started)
        if len(ranked) != len(set(ranked)):
            raise ValueError('Model produced duplicate actions')
        outputs[position] = {'predictions': ranked[:4], 'rankable': ranked}
        ledger.append({
            'position': position, 'input_index': query['index'],
            'mature_updates_at_cutoff': cursor, 'new_updates_at_cutoff': matured,
            'latest_training_ready_index': latest_ready,
        })
    if set(outputs) != set(positions):
        raise ValueError('Replay did not emit every evaluation prediction')
    return outputs, {'update': timing_summary(updates_ns), 'predict': timing_summary(predicts_ns)}, ledger


def cohorts(rows):
    return {
        'validation_all': [row for row in rows if row['split'] == 'validation'],
        'validation_original': [row for row in rows if row['split'] == 'validation'
                                and row['evidence_grade'] == 'original'],
        'test_all': [row for row in rows if row['split'] == 'test'],
        'test_original': [row for row in rows if row['split'] == 'test'
                          and row['evidence_grade'] == 'original'],
        'full_all': list(rows),
        'full_original': [row for row in rows if row['evidence_grade'] == 'original'],
    }


def score_cohorts(rows, outputs):
    return {name: scoring.measure(part, outputs) for name, part in cohorts(rows).items()}


def selection_key(candidate):
    """Top4 remains primary, with service and exact Top2 explicit tiebreakers."""
    metric = candidate['validation']['validation_original']
    return (
        metric['top4_hits'], metric['top2_hits'], metric['top1_hits'],
        metric['exact']['top4_hits'], metric['exact']['top2_hits'], metric['exact']['top1_hits'],
        metric['rank_score']['total'], -candidate.get('order', 0),
    )


def choose_rnn(candidates):
    if not candidates or not all(candidate['validation']['validation_original']['n'] for candidate in candidates):
        raise ValueError('No legal original-evidence validation cohort for RNN selection')
    return max(candidates, key=selection_key)


def rolling_intervals(count, width=ROLLING_WIDTH, stride=ROLLING_STRIDE):
    if width < 1 or stride < 1:
        raise ValueError('Rolling window width and stride must be positive')
    if count < width:
        return []
    ends = list(range(width, count + 1, stride))
    if ends[-1] != count:
        ends.append(count)
    return [(end - width, end) for end in ends]


def rolling_metrics(rows, outputs):
    legal = [row for row in rows if scoring.eligible(row)]
    result = []
    for start, end in rolling_intervals(len(legal)):
        window = legal[start:end]
        result.append({
            'ordinal_start': start, 'ordinal_end': end,
            'first_position': window[0]['position'], 'last_position': window[-1]['position'],
            'metrics': scoring.measure(window, outputs),
        })
    return {'legal_n': len(legal), 'width': ROLLING_WIDTH, 'stride': ROLLING_STRIDE,
            'windows': result}


def _change(before, after):
    if after and not before:
        return 'helped'
    if before and not after:
        return 'hurt'
    if after:
        return 'both_correct'
    return 'both_wrong'


def paired_help_harm(rows, reference, alternative):
    """Paired, sample-level help/harm for accepted and exact Top1/2/4."""
    counts = {level: {k: Counter() for k in (1, 2, 4)} for level in ('accepted', 'exact')}
    rank_counts = Counter()
    cases = []
    for row in rows:
        if not scoring.eligible(row):
            continue
        before = scoring.prediction_at(reference, row['position'])
        after = scoring.prediction_at(alternative, row['position'])
        before_hits, after_hits = scoring.hits(row, before), scoring.hits(row, after)
        detail = {'position': row['position'], 'sample_id': row['sample_id'],
                  'split': row['split'], 'evidence_grade': row['evidence_grade'],
                  'truth_action': row['truth_action'],
                  'reference_predictions': before['predictions'],
                  'alternative_predictions': after['predictions'],
                  'accepted': {}, 'exact': {}}
        for level in counts:
            for k in (1, 2, 4):
                change = _change(before_hits[level][k], after_hits[level][k])
                counts[level][k][change] += 1
                detail[level][f'top{k}'] = {
                    'reference': before_hits[level][k], 'alternative': after_hits[level][k],
                    'change': change,
                }
        before_rank = scoring.ranked_gain(row, before)['score']
        after_rank = scoring.ranked_gain(row, after)['score']
        if after_rank > before_rank:
            rank_change = 'helped'
        elif after_rank < before_rank:
            rank_change = 'hurt'
        else:
            rank_change = 'unchanged'
        rank_counts[rank_change] += 1
        detail['rank_score'] = {'reference': before_rank, 'alternative': after_rank,
                                'delta': after_rank - before_rank, 'change': rank_change}
        detail['decision_loss'] = {
            'reference': scoring.observed_decision_loss(row, before),
            'alternative': scoring.observed_decision_loss(row, after),
        }
        cases.append(detail)
    summary = {
        level: {f'top{k}': {**counts[level][k],
                            'net': counts[level][k]['helped'] - counts[level][k]['hurt']}
                 for k in (1, 2, 4)}
        for level in counts
    }
    summary['rank_score'] = {**rank_counts,
                             'net': rank_counts['helped'] - rank_counts['hurt']}
    summary['n'] = len(cases)
    return summary, cases


def _pct(value):
    return '—' if value is None else f'{value:.1%}'


def _metric_cells(metric):
    return [
        f"{_pct(metric['top1'])} ({metric['top1_hits']})",
        f"{_pct(metric['top2'])} ({metric['top2_hits']})",
        f"{_pct(metric['top4'])} ({metric['top4_hits']})",
        f"{_pct(metric['exact']['top1'])} ({metric['exact']['top1_hits']})",
        f"{_pct(metric['exact']['top2'])} ({metric['exact']['top2_hits']})",
        f"{_pct(metric['exact']['top4'])} ({metric['exact']['top4_hits']})",
        _pct(metric['rank_score']['mean']), _pct(metric['prediction_coverage']),
    ]


def _ms(value):
    return '—' if value is None else f'{value:.3f}'


def render_report(summary):
    selected = summary['selected']
    lines = [
        '# 在线前台序列 RNN shadow 实验', '',
        '本 run 只比较一个小型、逐条更新的 Elman RNN 与单步无序列统计基线；不训练、不选择多步统计模型。',
        'RNN 的输入是截点前最多 5 个已确认前台到达，按会话、采集边界、来源失效和粒度丢失边界切断。单步基线仅见当前前台来源和成熟反馈频数。',
        '两者均不读取后台应用/窗口/标签/窗格、打开库存、候选清单、`background_state`、评测真值或未来标签；输出只来自此前已成熟的历史标签。', '',
        '## 协议', '',
        f"- 固定数据集中有 {summary['scope']['candidate_n']} 个评测截点，统一合法性审查后 {summary['scope']['eligible_n']} 个；"
        f"验证／测试划分固定，测试段在选参前不评分。",
        f"- 训练使用 `{TRAINING_POLICY}` 的第一个保留后续到达标签，共 {summary['training']['mature_relations']} 条成熟关系；"
        '每次预测只吸收 `ready_index < input_index` 的关系。',
        '- RNN 配置固定为 4 个小模型；验证原始有证据样本按服务接受 Top4、Top2、Top1，精确 Top4、Top2、Top1，RankScore 和预设顺序选择。',
        '- 所有结果由同一 `evaluation_contract` 评分：服务首页正确回退收益为 0.6，精确目标另列；RankScore 为 Top1 到 Top4 的前缀收益平均。',
        '- 固定数据此前已被查看，因此这是可复现实验性 shadow 回放，不是全新独立留出集。', '',
        '## 验证选参（仅验证原始有证据样本）', '',
        '| 配置 | 样本 | 服务 Top1 | 服务 Top2 | 服务 Top4 | 精确 Top1 | 精确 Top2 | 精确 Top4 | RankScore | 覆盖率 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for candidate in summary['candidates']:
        metric = candidate['validation']['validation_original']
        cells = _metric_cells(metric)
        lines.append('| ' + ' | '.join([candidate['config']['id'], str(metric['n']), *cells]) + ' |')
    lines += [
        '', f"选择：`{selected['config']['id']}`（验证选择键：`{list(selection_key(selected))}`）。", '',
        '## 固定评测与全程 prequential 回放', '',
        '| 范围 | 模型 | 样本 | 服务 Top1 | 服务 Top2 | 服务 Top4 | 精确 Top1 | 精确 Top2 | 精确 Top4 | RankScore | 覆盖率 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    labels = {
        'validation_original': '验证／原始有证据',
        'test_original': '固定测试／原始有证据',
        'test_all': '固定测试／全部合法',
        'full_original': '全程／原始有证据',
        'full_all': '全程 prequential／全部合法',
    }
    for cohort in ('validation_original', 'test_original', 'test_all', 'full_original', 'full_all'):
        for name, label in (('baseline', '单步统计基线'), ('selected', '在线 RNN')):
            metric = summary['models'][name]['metrics'][cohort]
            cells = _metric_cells(metric)
            lines.append('| ' + ' | '.join([labels[cohort], label, str(metric['n']), *cells]) + ' |')
    lines += [
        '', '## 滚动 prequential 窗口（每窗 25 个合法截点）', '',
        '| 序列范围 | 窗口 | 模型 | 服务 Top1 | 服务 Top2 | 服务 Top4 | 精确 Top1 | 精确 Top2 | 精确 Top4 | RankScore |',
        '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for scope in ('all', 'test'):
        for name, label in (('baseline', '单步统计基线'), ('selected', '在线 RNN')):
            curve = summary['prequential'][scope][name]
            for item in curve['windows']:
                metric = item['metrics']
                cells = _metric_cells(metric)
                lines.append('| ' + ' | '.join([
                    '全程' if scope == 'all' else '固定测试',
                    f"{item['first_position']}–{item['last_position']}", label,
                    *cells[:7],
                ]) + ' |')
    lines += [
        '', '## 更新与推理耗时', '',
        '范围为本机 Python shadow replay：更新是单条成熟监督关系的模型更新；推理包括对当时已学习的完整标签词表排序，不含数据读取和统一评分。', '',
        '| 模型 | 操作 | 次数 | 最小 ms | 中位 ms | P95 ms | 最大 ms | 总 ms |',
        '|---|---|---:|---:|---:|---:|---:|---:|',
    ]
    for name, label in (('baseline', '单步统计基线'), ('selected', '在线 RNN')):
        for operation in ('update', 'predict'):
            timing = summary['models'][name]['timing'][operation]
            lines.append('| ' + ' | '.join([
                label, operation, str(timing['n']), _ms(timing['min_ms']), _ms(timing['median_ms']),
                _ms(timing['p95_ms']), _ms(timing['max_ms']), _ms(timing['total_ms']),
            ]) + ' |')
    lines += [
        '', '## RNN 相对单步基线的配对样本 help/harm', '',
        '| 口径 | TopK | 帮助 | 伤害 | 净变化 | 双正确 | 双错误 |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for level, title in (('accepted', '服务接受'), ('exact', '精确目标')):
        for k in (1, 2, 4):
            value = summary['help_harm'][level][f'top{k}']
            lines.append(f"| {title} | {k} | {value['helped']} | {value['hurt']} | {value['net']} | "
                         f"{value['both_correct']} | {value['both_wrong']} |")
    rank = summary['help_harm']['rank_score']
    lines += [
        f"RankScore：帮助 {rank['helped']}，伤害 {rank['hurt']}，不变 {rank['unchanged']}，净变化 {rank['net']}。",
        '逐样本的 Top1/Top2/Top4、服务接受/精确命中、RankScore 与实际损失均保存在 `help-harm-private.json`；该文件未进入源码。', '',
        '## 边界与限制', '',
        '- RNN 原始分数只用于排序，未声称为校准后验概率；它没有调用菜单候选、活动库存或任何未来标签。',
        '- 评测中的原始有证据分组及真值只用于评分/报告，未传给任一模型；该分组也不是线上置信门。',
        '- 本实验没有修改采集数据库、采集范围、运行时模型、安装包或菜单；最终模型状态仅是私有 shadow 产物，不能视为已部署能力。',
    ]
    return '\n'.join(lines) + '\n'


def _check_private_output(path):
    path = path.expanduser().resolve()
    root = PRIVATE_ROOT.resolve()
    if path == root or not path.is_relative_to(root) or path.exists():
        raise ValueError('Output must be a new directory under SmartApp private training storage')
    return path


def _seal(output):
    for path in output.iterdir():
        if path.is_file():
            path.chmod(0o600)
    hashes = {path.name: io.sha(path) for path in output.iterdir()
              if path.is_file() and path.name != 'artifact-hashes.json'}
    _private_json(output / 'artifact-hashes.json', hashes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True,
                        help='Frozen private semantic-trajectory dataset run')
    parser.add_argument('--output', type=Path, required=True,
                        help='New directory under SmartApp private training storage')
    args = parser.parse_args()

    dataset = args.dataset.expanduser().resolve()
    output = _check_private_output(args.output)
    provenance, dataset_hashes = verify_run(dataset)
    snapshot = Path(provenance['snapshot']).expanduser().resolve()
    expected_events_hash = provenance['migration']['current']['events.jsonl']
    if io.sha(snapshot / 'events.jsonl') != expected_events_hash:
        raise ValueError('Frozen raw event snapshot does not match dataset provenance')
    source_bytes = _source_bytes()

    with gzip.open(dataset / 'dataset-private.json.gz', 'rt') as handle:
        frozen = json.load(handle)
    data = _safe_data(frozen)
    # Drop the original container so model code cannot accidentally receive its
    # historical background fields or evaluator-only bundles later in the run.
    del frozen

    events = io.load_lines(snapshot / 'events.jsonl')
    contexts, _homes, context_audit = build_contexts(events, data['observations'], io.KNOWN_TEST_SESSIONS)
    inputs = sequence_inputs(data['observations'], contexts)
    stream, training_audit = supervision_stream(data, inputs)
    sequence_lengths = Counter(value['sequence_length'] for value in inputs.values())
    if any(len(value['sequence']) > SEQUENCE_LIMIT for value in inputs.values()):
        raise ValueError('Unsafe sequence length')

    os.umask(0o077)
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    rows = scoring.prepare_evaluation(data, output)
    rows.sort(key=lambda row: row['position'])
    if any(left['position'] >= right['position'] for left, right in zip(rows, rows[1:])):
        raise ValueError('Evaluation positions must be strictly ordered')
    scope = scoring.audit_summary(rows)
    split_counts = Counter(row['split'] for row in rows)
    evidence_counts = Counter(row['evidence_grade'] for row in rows)

    plan = {
        'version': VERSION,
        'protocol': 'strict-causal-prequential-shadow-replay',
        'model_route': 'online_elman_rnn_only_no_multistep_statistical_ablation',
        'baseline': {
            'family': CurrentOnlyFrequency.family,
            'input': 'current foreground origin only',
            'ranking': 'conditional mature frequency; global mature frequency only fills remaining slots',
            'forbidden_inputs': ['prior sequence', 'time feature', 'background state', 'inventory', 'candidate list'],
        },
        'sequence_model': {
            'family': OnlineSequenceRNN.family,
            'input': f'last 1..{SEQUENCE_LIMIT} confirmed foreground target keys reconstructed as of cutoff',
            'update': 'one mature hard_kind head-zero relationship; dynamic label vocabulary; online SGD/BPTT',
            'scores_are_calibrated_probabilities': False,
        },
        'input_allowlist': list(SAFE_OBSERVATION_KEYS),
        'predictor_fields': ['origin', 'sequence', 'sequence_length', 'index', 'timestamp'],
        'excluded_model_inputs': ['background_state', 'candidateTargetIDs', 'candidate inventory',
                                  'open inventory', 'evaluation truth', 'future labels', 'annotations'],
        'sequence_boundary_source': 'TemporalHistory build_contexts; session/capture/source/granularity boundaries reset suffixes',
        'training_policy': TRAINING_POLICY,
        'maturity_rule': 'ready_index < prediction input_index',
        'rnn_configs_fixed_before_dataset_read': RNN_CONFIGS,
        'selection': 'validation original-evidence: accepted Top4, Top2, Top1; exact Top4, Top2, Top1; RankScore; fixed config order',
        'top2_required_everywhere': True,
        'fixed_test_scored_after_selection': True,
        'rolling_prequential': {'width': ROLLING_WIDTH, 'stride': ROLLING_STRIDE},
        'dataset': str(dataset), 'dataset_hashes': dataset_hashes,
        'snapshot_events_sha256': expected_events_hash,
        'source_hashes': {name: hashlib.sha256(value).hexdigest() for name, value in source_bytes.items()},
        'scoring_contract': scoring.contract(),
        'test_previously_seen': provenance.get('test_previously_seen'),
        'collection_changed': False, 'runtime_model_changed': False,
        'runtime_installation_changed': False, 'deployment': False,
    }
    _private_json(output / 'plan.json', plan)

    positions = [row['position'] for row in rows]
    baseline_model = CurrentOnlyFrequency()
    baseline_outputs, baseline_timing, baseline_ledger = causal_replay(inputs, stream, positions, baseline_model)

    candidates = []
    for order, config in enumerate(RNN_CONFIGS):
        model = OnlineSequenceRNN(config)
        outputs, timing, ledger = causal_replay(inputs, stream, positions, model)
        if ledger != baseline_ledger:
            raise ValueError('Models did not use the identical causal update ledger')
        validation = {
            'validation_all': scoring.measure(cohorts(rows)['validation_all'], outputs),
            'validation_original': scoring.measure(cohorts(rows)['validation_original'], outputs),
        }
        candidates.append({'order': order, 'config': config, 'validation': validation,
                           'outputs': outputs, 'timing': timing, 'model': model})

    selected = choose_rnn(candidates)
    # This immutable checkpoint contains no test metrics, and is written before
    # the fixed-test scores are constructed.
    _private_json(output / 'selection.json', {
        'selected_config': selected['config'],
        'selection_key': list(selection_key(selected)),
        'selection_uses_test_labels': False,
        'candidates': [{'config': candidate['config'], 'validation': candidate['validation']}
                       for candidate in candidates],
    })

    baseline_metrics = score_cohorts(rows, baseline_outputs)
    selected_metrics = score_cohorts(rows, selected['outputs'])
    for cohort in baseline_metrics:
        scoring.require_comparable(baseline_metrics[cohort], selected_metrics[cohort])
    help_harm, cases = paired_help_harm(rows, baseline_outputs, selected['outputs'])

    prequential = {
        'all': {
            'baseline': rolling_metrics(rows, baseline_outputs),
            'selected': rolling_metrics(rows, selected['outputs']),
        },
        'test': {
            'baseline': rolling_metrics(cohorts(rows)['test_all'], baseline_outputs),
            'selected': rolling_metrics(cohorts(rows)['test_all'], selected['outputs']),
        },
    }
    summary = {
        'version': VERSION,
        'scope': {**scope, 'splits': dict(split_counts), 'evidence_grades': dict(evidence_counts)},
        'training': training_audit,
        'sequence_audit': {
            'input_n': len(inputs), 'sequence_length_counts': {str(key): value for key, value in sorted(sequence_lengths.items())},
            'all_sequences_end_at_current_foreground_origin': True,
            'maximum_sequence_length': SEQUENCE_LIMIT,
            'temporal_history_audit': dict(context_audit),
            'known_test_sessions_excluded_from_history': len(io.KNOWN_TEST_SESSIONS),
        },
        'candidates': [{'order': candidate['order'], 'config': candidate['config'],
                        'validation': candidate['validation'], 'timing': candidate['timing']}
                       for candidate in candidates],
        'selected': {'order': selected['order'], 'config': selected['config'],
                     'validation': selected['validation']},
        'models': {
            'baseline': {'family': baseline_model.family, 'metrics': baseline_metrics,
                         'timing': baseline_timing, 'state': {'updates': baseline_model.updates,
                                                               'labels': len(baseline_model.global_counts)}},
            'selected': {'family': selected['model'].family, 'metrics': selected_metrics,
                         'timing': selected['timing'], 'state': {'updates': selected['model'].updates,
                                                                   'labels': len(selected['model'].labels)}},
        },
        'help_harm': help_harm,
        'prequential': prequential,
        'selection_before_test_score': True,
        'collection_changed': False, 'runtime_model_changed': False, 'deployment': False,
    }
    _private_json(output / 'results.json', summary)
    _private_json(output / 'causal-ledger-private.json', baseline_ledger)
    _private_json(output / 'help-harm-private.json', {'summary': help_harm, 'cases': cases})
    _private_json(output / 'prequential-private.json', prequential)
    _private_gzip_json(output / 'predictions-private.json.gz', {
        'baseline': baseline_outputs, 'selected_rnn': selected['outputs'],
    })
    _private_gzip_json(output / 'model-final-private.json.gz', {
        'as_of_last_evaluation_cutoff_only': True,
        'deployment': False,
        'baseline': baseline_model.export_state(),
        'selected_rnn': selected['model'].export_state(),
    })
    (output / 'report-private.md').write_text(render_report(summary))
    (output / 'report-private.md').chmod(0o600)
    _private_json(output / 'run.json', {
        'complete': True, 'version': VERSION, 'dataset': str(dataset),
        'dataset_hashes': dataset_hashes, 'snapshot': str(snapshot),
        'snapshot_events_sha256': expected_events_hash,
        'selected_config': selected['config'], 'test_previously_seen': provenance.get('test_previously_seen'),
        'collection_changed': False, 'runtime_model_changed': False,
        'runtime_installation_changed': False, 'deployment': False,
    })
    for name, value in source_bytes.items():
        path = output / ('source-' + name)
        path.write_bytes(value)
        path.chmod(0o600)

    if verify_run(dataset)[1] != dataset_hashes:
        raise ValueError('Frozen dataset changed during replay')
    if io.sha(snapshot / 'events.jsonl') != expected_events_hash:
        raise ValueError('Frozen raw event snapshot changed during replay')
    if any((Path(__file__).resolve().parent / name).read_bytes() != value
           for name, value in source_bytes.items()):
        raise ValueError('Experiment source changed during replay')
    _seal(output)
    compact = {
        'output': str(output), 'selected_config': selected['config']['id'],
        'baseline_test_all_top4': baseline_metrics['test_all']['top4'],
        'rnn_test_all_top4': selected_metrics['test_all']['top4'],
        'baseline_test_all_top2': baseline_metrics['test_all']['top2'],
        'rnn_test_all_top2': selected_metrics['test_all']['top2'],
    }
    print(json.dumps(compact, ensure_ascii=False))


if __name__ == '__main__':
    main()
