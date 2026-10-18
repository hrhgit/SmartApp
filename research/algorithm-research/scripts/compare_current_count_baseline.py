"""Replay the plain-count baseline and compare identical historical cutoffs."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import evaluation_contract as scoring
import evaluate_context_knn as core
import context_statistics as cs
from navigation_hierarchy import ENTRY, learning_record
from current_target_count_baseline import NAME, CONFIG, replay
from prequential_learning_curve import new_directory, seal, cap_training, rolling_intervals, head_zero
from review_historical_models import Sources, write, replay_legacy
from summarize_prequential_learning import GROUPS, curve_members, values, window_curve


REFERENCE = 'statistics/none'
DISPLAY = [('无情景统计（主基准）', REFERENCE),
    ('纯当前目标频数＋全局兜底（补充）', NAME),
    ('线性交叉熵', 'joint/linear_ce_lr0.1_seed20260923'),
    ('平滑Top2', 'joint/linear_smooth_top2_lr0.1_seed20260923'),
    ('固定损失决策', 'risk/fixed'),
    ('旧条件频数（无兜底，次数并列按近期）', 'legacy/stat_frequency_conditional')]


class NoBackground:
    def factors(self, *args, **kwargs):
        raise AssertionError('No-context baseline must not read background similarities')


def replay_no_context(data, rows, config, events, limit=None):
    if config['mode'] != 'none' or config['profile'] is not None:
        raise ValueError('This replay is only for the fixed no-context statistical baseline')
    by_index = {q['index']: q['position'] for q in data['observations']}
    homes, catalog = {}, {}
    # Exactly the original as-of execution-home resolver, without constructing
    # any temporal or inventory features. Future catalog entries never enter.
    for index, event in enumerate(events):
        if event['kind'] == 'target.update':
            target = event['data']['target']
            if core.admitted(target) and not core.synthetic(target):
                catalog[target['id']] = target
            else:
                catalog.pop(target['id'], None)
        if index in by_index:
            homes[by_index[index]] = next((key for key, t in sorted(catalog.items())
                                          if t['kind'] == 'page' and t.get('locator') == ENTRY), None)
    inputs = [{**{k: q[k] for k in ('position', 'index', 'timestamp', 'origin', 'target', 'annotation')},
               'current_sequence': [q['origin']], 'background_through_index': q['index'],
               'inventory_through_index': q['index']} for q in data['observations']]
    stream = head_zero(cap_training(data, limit) if limit is not None else data)
    memory = cs.StatisticalMemory(); cursor = 0; forecasts = {}; audit = []
    for row in rows:
        pos = row['position']; q = inputs[pos]
        while cursor < len(stream) and stream[cursor]['ready_index'] < q['index']:
            item = stream[cursor]; before = inputs[item['position']]
            memory.learn(before, learning_record(item, before), item['ready_index'])
            cursor += 1
        forecasts[pos] = memory.prepare(q, homes[pos], NoBackground()).predict(config)
        audit.append({'position': pos, 'input_index': q['index'], 'learned_relations': cursor,
            'latest_ready_index': stream[cursor-1]['ready_index'] if cursor else -1})
    if limit is not None and len({r['learned_relations'] for r in audit}) != 1:
        raise ValueError('Frozen no-context baseline updated')
    return forecasts, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sources = Sources()
    previous = sources.read(args.summary/'verification.json')
    paths = {k: Path(v) for k, v in previous['source_runs'].items()}
    meta = sources.read(paths['online']/'run.json')
    if meta['scoring_contract'] != scoring.contract():
        raise ValueError('Scorer changed; previous curves cannot be silently reused')
    dataset = Path(meta['dataset'])
    data = sources.read(dataset/'dataset-private.json.gz')
    plan = sources.read(dataset/'plan.json')
    out = new_directory(args.output)
    candidates = scoring.prepare_evaluation(data, out)
    parts = {k: v for k, v in scoring.cohorts(candidates).items() if k in GROUPS}
    rows, evidence = curve_members(parts)
    masks = {'all': np.ones(len(rows), dtype=bool), 'evidence_total': np.array([r['position'] in evidence for r in rows])}
    intervals = rolling_intervals(len(rows), plan['rolling_window'], plan['rolling_stride'])
    windows = sources.read(args.summary/'rolling-metrics-private.json.gz')['windows']
    if [(w['start']-1, w['end']) for w in windows] != intervals:
        raise ValueError('Changed window membership')
    protocol = {'baseline': REFERENCE, 'supplemental_count_baseline': NAME, 'count_config': CONFIG, 'dataset': str(dataset), 'scope': plan,
        'training': 'replayed from scratch for online and frozen count baselines',
        'comparators': 'saved as-of forecasts from the same dataset, rescored through the same entry point',
        'comparison_parameters_changed': False, 'model_selection': False,
        'old_statistical_names_audited': True, 'scoring_contract': scoring.contract()}
    write(out/'plan.json', protocol)
    models = sources.read(paths['online']/'models.json')
    events = core.load_lines(sources.check(Path(meta['snapshot'])/'events.jsonl'))
    write(out/'statistics-definitions.json', {n: models[n] for n in ('statistics/none', 'navigation/v2', 'legacy/stat_frequency_conditional')})
    metrics, curves, audits, new_outputs, per_rows = {}, {}, {}, {}, {}
    for arm, path in paths.items():
        run_meta = sources.read(path/'run.json')
        if run_meta['dataset'] != str(dataset) or run_meta['scoring_contract'] != scoring.contract():
            raise ValueError('Comparator mismatch')
        outputs = sources.read(path/'predictions-private.json.gz')
        old_metrics = sources.read(path/'metrics-private.json')
        limit = plan['warmup_ready_index'] if arm == 'frozen' else None
        no_context, no_context_audit = replay_no_context(data, candidates, models[REFERENCE]['config'], events, limit)
        if any(v['predictions'] != outputs[REFERENCE][str(p)]['predictions'] for p, v in no_context.items()):
            raise ValueError('Replayed no-context baseline differs from the original')
        write(out/f'{arm}-no-context-forecasts-private.json.gz', no_context)
        write(out/f'{arm}-no-context-audit-private.json', no_context_audit)
        print(json.dumps({'phase': 'no_context_replayed_with_exact_forecast_parity', 'arm': arm}), flush=True)
        predictions, state, audit = replay(data, candidates, limit)
        outputs[NAME] = predictions
        new_outputs[arm], audits[arm] = predictions, audit
        write(out/f'{arm}-count-state-private.json', state)
        write(out/f'{arm}-chronology-audit-private.json', audit)
        write(out/f'{arm}-count-forecasts-private.json.gz', predictions)
        # Re-run the historical raw conditional baseline separately and require
        # exact forecast parity. Its empty menus and recency ties are not hidden.
        legacy_config = deepcopy(models['legacy/stat_frequency_conditional']['config'])
        legacy, _, legacy_audit = replay_legacy(cap_training(data, limit) if limit is not None else data,
                                               candidates, legacy_config, 'hard_kind')
        if any(v['predictions'] != outputs['legacy/stat_frequency_conditional'][str(p)]['predictions'] for p, v in legacy.items()):
            raise ValueError('Historical count baseline parity failed')
        write(out/f'{arm}-legacy-count-audit-private.json', legacy_audit)
        metrics[arm], curves[arm], per_rows[arm] = {}, {}, {}
        for name, forecasts in outputs.items():
            if set(map(int, forecasts)) != {r['position'] for r in candidates}:
                raise ValueError('Missing cutoff')
            metrics[arm][name], curves[arm][name] = {}, {}
            v = values(rows, forecasts)
            per_rows[arm][name] = v
            for group in GROUPS:
                result = scoring.measure(parts[group], forecasts)
                if name != NAME and result != old_metrics[name][group]:
                    raise ValueError('Comparator score changed')
                metrics[arm][name][group] = result
                curves[arm][name][group] = window_curve(v, masks[group], intervals)
                if name == NAME:
                    for i, (a, b) in enumerate(intervals):
                        subset = [r for j, r in enumerate(rows[a:b], a) if masks[group][j]]
                        check = scoring.measure(subset, forecasts)
                        got = curves[arm][name][group][i]
                        for key, expected in [('score', check['rank_score']['mean']),
                                              ('top1', check['exact']['top1']), ('top4', check['exact']['top4'])]:
                            if not np.isclose(got[key], 100*expected, atol=1e-9):
                                raise ValueError('Count curve differs from sole scoring entry point')
        print(json.dumps({'phase': 'baseline_and_comparators_scored', 'arm': arm, 'models': len(outputs)}), flush=True)
    first = candidates[0]['position']
    if new_outputs['online'][first] != new_outputs['frozen'][first]:
        raise ValueError('Warmup fork parity failed')
    for group in GROUPS:
        scoring.require_comparable(*[m[group] for arm in metrics.values() for m in arm.values()])
    deltas = {}
    for name in metrics['online']:
        deltas[name] = {}
        for group in GROUPS:
            delta = per_rows['online'][name][masks[group]]-per_rows['online'][REFERENCE][masks[group]]
            rolling = np.array([c['score'] for c in curves['online'][name][group]]) - np.array([c['score'] for c in curves['online'][REFERENCE][group]])
            deltas[name][group] = {'composite_points': float(delta[:, 0].mean()*100),
                'exact_top1_net_hits': int(delta[:, 1].sum()), 'exact_top4_net_hits': int(delta[:, 2].sum()),
                'better_samples': int((delta[:, 0] > 1e-10).sum()), 'worse_samples': int((delta[:, 0] < -1e-10).sum()),
                'tied_samples': int((np.abs(delta[:, 0]) <= 1e-10).sum()),
                'positive_windows': int((rolling > 1e-8).sum()), 'negative_windows': int((rolling < -1e-8).sum()),
                'tied_windows': int((np.abs(rolling) <= 1e-8).sum())}
    write(out/'metrics-private.json', metrics)
    write(out/'paired-gains-private.json', deltas)
    write(out/'rolling-metrics-private.json.gz', {'windows': windows, 'curves': curves})
    order = [REFERENCE, NAME]+sorted(n for n in metrics['online'] if n not in (REFERENCE, NAME))
    compact = {'plan': plan, 'windows': windows, 'order': order, 'models': {}, 'baseline_name': REFERENCE, 'preferred_model': REFERENCE}
    for name in order:
        compact['models'][name] = {arm: {g: [[round(v[k], 4) for k in ('score', 'top1', 'top4')]
                                           for v in curves[arm][name][g]] for g in GROUPS} for arm in curves}
    template = Path(__file__).with_name('prequential_dashboard_template.html')
    html = template.read_text().replace('/*__CURVE_DATA__*/ null', json.dumps(compact, ensure_ascii=False, separators=(',', ':')).replace('</', '<\\/'))
    (out/'learning-curves-private.html').write_text(html)
    plot(out, curves, windows)
    report(out, metrics, deltas, plan)
    sources.verify_unchanged()
    write(out/'verification.json', {'complete': True, 'models_with_new_baseline': len(order), 'sizes': plan['sizes'],
        'new_count_baseline_replayed': True, 'legacy_count_baseline_rerun_parity': True,
        'no_context_baseline_replayed_with_forecast_parity': True,
        'frozen_count_state_constant': True, 'warmup_fork_parity': True, 'every_count_window_independently_scored': True,
        'all_182_comparators_recomputed_with_original_score_parity': True,
        'scoring_contract': scoring.contract(), 'sources': sources.accessed,
        'baseline_config': CONFIG, 'collector_or_runtime_changed': False})
    for p in [Path(__file__), template, Path(__file__).with_name('current_target_count_baseline.py'),
              Path(__file__).with_name('summarize_prequential_learning.py')]:
        (out/('source-'+p.name)).write_bytes(p.read_bytes())
    seal(out)
    print(json.dumps({n: {g: metrics['online'][n][g]['rank_score']['mean']*100 for g in GROUPS} for _, n in DISPLAY}), flush=True)


def report(out, metrics, deltas, plan):
    def line(label, name):
        cols = []
        for g in GROUPS:
            cols += [f"{metrics['online'][name][g]['rank_score']['mean']*100:.2f}", f"{deltas[name][g]['composite_points']:+.2f}"]
        return '| '+label+' | '+' | '.join(cols)+' |'
    header = ['| 方案 | 完整集综合分 | 相对无情景统计 | 有证据综合分 | 相对无情景统计 |', '|---|---:|---:|---:|---:|']
    lines = ['# 无情景统计主基准与新增方法的净提升', '',
        f"同一100次独立成熟行为预热；完整合法集{plan['sizes']['all']}条，有证据集{plan['sizes']['evidence_total']}条（27.6%）。最近100条合法样本、步长10，92个窗口。两集合、标签、评分代码与前轮一致。", '',
        '按用户明确指定，主基准是上一轮 statistics/none 无情景统计：不输入时间背景相似度或后台库存，不叠加额外梯度学习器。保留它已有的语义平滑、全局统计、近期性及时间衰减，不能用更弱的朴素频数替换主基准。', '',
        '主基准沿用冻结配置：语义强度64、全局强度1、近期性混合0.5、24小时半衰期，并采用既有服务层处理。本轮没有调参。所有统计都按成熟新行为递增更新，“无额外学习器”并不表示不更新计数。', '',
        '补充的朴素频数只累计当前目标到下一目标的次数；条件次数优先，全局次数仅用于并列（包括条件次数为0的已知目标），仍并列按稳定ID。无同起点记录时自然退回全局频数。它不含语义平滑、时间衰减、近期性或服务标签归并；不以后台库存或同起点支持集限定输出。旧条件频数无兜底、并列按近期，单列为历史诊断。', '',
        f'[全部模型与统计基准的交互曲线](<{out}/learning-curves-private.html>)', '',
        '## 全程综合分（0–100；差值为分，不是相对百分比）', '', *header]
    lines += [line(label, name) for label, name in DISPLAY]
    lines += ['', '## 精确目标命中数与净新增命中', '',
        '| 方案 | 全集Top1 | 全集Top4 | 有证据Top1 | 有证据Top4 |', '|---|---:|---:|---:|---:|']
    for label, name in DISPLAY:
        cols = []
        for g in GROUPS:
            for k in (1, 4):
                m = metrics['online'][name][g]
                net = deltas[name][g]['exact_top'+str(k)+'_net_hits']
                cols.append(f"{m['exact']['top'+str(k)+'_hits']}/{m['n']}（净{net:+d}）")
        lines.append('| '+label+' | '+' | '.join(cols)+' |')
    lines += ['', '## 滚动增益稳定性', '',
        '| 方案 | 全集胜／平／负窗口 | 有证据胜／平／负窗口 |', '|---|---:|---:|']
    for label, name in DISPLAY:
        fields = ['／'.join(str(deltas[name][g][k]) for k in ('positive_windows', 'tied_windows', 'negative_windows')) for g in GROUPS]
        lines.append('| '+label+' | '+' | '.join(fields)+' |')
    lines += ['', '## 全部182组旧配置对无情景统计的配对增益', '', *header]
    lines += [line(name, name) for name in sorted(metrics['online']) if name != NAME]
    lines += ['', '## 边界与复现', '',
        '- 朴素频数在线／预热后冻结两臂均从头回放；旧条件频数两臂亦复跑并逐条核对。无情景统计主基准另从头重放并验证逐条预测一致。其他配置沿用相同截点已保存的因果预测，并统一重算，不重新挑参数。',
        '- 同一原始结果在同一当前目标下只计一次，全局只计一次，多个锚点不增加独立行为票数。预测前必须标签已成熟；冻结臂词表及全部计数状态哈希全程不变。',
        '- 综合分不是正确率：精确收益1、正确首页回退0.6、错误0，对Top1至Top4前缀最佳收益取均值。额外净命中数按精确目标计算。',
        '- 增益是完整方案相对基线的净增益，包含服务决策、语义和先验差异；不是隔离所有因素后单独归因于神经网络或梯度学习的因果效果。',
        '- 曲线窗口重叠，胜窗数量不构成独立显著性检验；有证据成员是同一事后诊断子集，支持必须在预测前已可用。历史开发数据不是新独立测试。',
        '- 所有旧run、原始数据和运行模型未修改。', '']
    (out/'report-private.md').write_text('\n'.join(lines))


def plot(out, curves, windows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'Arial Unicode MS', 'font.size': 10, 'axes.unicode_minus': False})
    selected = [('纯当前目标频数（补充）', NAME, '#242424'), ('无情景统计（主基准）', REFERENCE, '#009e73'),
                ('线性交叉熵', 'joint/linear_ce_lr0.1_seed20260923', '#0072b2'), ('固定损失决策', 'risk/fixed', '#d55e00')]
    x = [w['end'] for w in windows]
    for metric, title in [('score', '综合分'), ('top1', '精确Top1正确率（%）'), ('top4', '精确Top4正确率（%）')]:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, layout='constrained')
        for j, group in enumerate(GROUPS):
            base = np.array([c[metric] for c in curves['online'][REFERENCE][group]])
            for label, name, color in selected:
                y = np.array([c[metric] for c in curves['online'][name][group]])
                axes[0, j].plot(x, y, label=label, color=color, linewidth=1.6)
                if name != REFERENCE:
                    axes[1, j].plot(x, y-base, label=label, color=color, linewidth=1.6)
            axes[0, j].set(title='完整合法集' if group == 'all' else '同窗口有证据子集', ylabel=title, ylim=(-2, 102))
            axes[1, j].axhline(0, color='#242424', linewidth=1)
            axes[1, j].set(ylabel='相对无情景统计增益（分）' if metric == 'score' else '增益（百分点）', xlabel='预热后预测序号')
            for ax in axes[:, j]:
                ax.grid(alpha=.18); ax.set_xlim(x[0], x[-1])
                if group == 'evidence_total':
                    for w in windows:
                        if w['evidence_n'] < 10:
                            ax.axvspan(w['end']-5, w['end']+5, color='#999999', alpha=.18)
        axes[0, 0].legend(ncol=2, fontsize=9, loc='upper left')
        fig.suptitle('无情景统计主基准 · '+title+'\n预热100次独立成熟行为；窗口100／步长10；灰带为有证据样本不足10条', fontsize=13)
        fig.savefig(out/f'baseline-comparison-{metric}.png', dpi=160)
        plt.close(fig)


if __name__ == '__main__':
    main()
