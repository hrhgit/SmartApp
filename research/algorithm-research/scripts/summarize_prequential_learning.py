"""Independently score both replay arms and build common-window learning curves."""
import argparse
from datetime import datetime
import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import evaluation_contract as scoring
from prequential_learning_curve import new_directory, seal, rolling_intervals, head_zero
from review_historical_models import Sources, write


GROUPS = ('all', 'evidence_total')
REFERENCE = 'joint/linear_smooth_top2_lr0.1_seed20260923'
REPRESENTATIVES = [
    ('平滑Top2', REFERENCE),
    ('固定损失决策', 'risk/fixed'),
    ('非线性Top2（0.03）', 'joint/nonlinear_top2_lr0.03_seed20260923'),
    ('线性Top2', 'joint/linear_top2_lr0.1_seed20260923'),
    ('无情景统计', 'statistics/none'),
    ('时间背景统计', 'statistics/time'),
    ('早期状态KNN', 'early/knn_state'),
    ('早期KMeans', 'early/kmeans_state'),
]


def curve_members(parts):
    """Exclude illegal labels BEFORE assigning rolling-window positions."""
    rows = [r for r in parts['all'] if scoring.eligible(r)]
    evidence = {r['position'] for r in parts['evidence_total'] if scoring.eligible(r)}
    return rows, evidence


def values(rows, outputs):
    # These are the sole scoring implementations; no independent hit or gain
    # formula is allowed in the curve builder.
    result = []
    for row in rows:
        output = scoring.prediction_at(outputs, row['position'])
        hit = scoring.hits(row, output)
        result.append([scoring.ranked_gain(row, output)['score'],
                       float(hit['exact'][1]), float(hit['exact'][4])])
    return np.asarray(result, dtype=float)


def window_curve(per_row, mask, intervals):
    weighted = per_row * np.asarray(mask)[:, None]
    prefix = np.vstack([np.zeros((1, 3)), np.cumsum(weighted, axis=0)])
    counts = np.r_[0, np.cumsum(mask)]
    result = []
    for left, right in intervals:
        n = int(counts[right]-counts[left])
        scores = (100*(prefix[right]-prefix[left])/n).tolist() if n else [None]*3
        result.append({'n': n, 'score': scores[0], 'top1': scores[1], 'top4': scores[2],
                       'insufficient_evidence': n < 10})
    return result


def local_time(milliseconds):
    return datetime.fromtimestamp(milliseconds/1000, ZoneInfo('America/Los_Angeles')).isoformat(timespec='seconds')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--online', type=Path, required=True)
    parser.add_argument('--frozen', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    sources = Sources()
    paths = {'online': args.online.resolve(), 'frozen': args.frozen.resolve()}
    metas = {arm: sources.read(path/'run.json') for arm, path in paths.items()}
    if any(not m['complete'] or m['scoring_contract'] != scoring.contract() for m in metas.values()):
        raise ValueError('Both arms must be complete with the same scorer')
    if metas['online']['dataset'] != metas['frozen']['dataset']:
        raise ValueError('Different cohorts')
    for path in paths.values():
        for name in sources.read(path/'artifact-hashes.json'):
            sources.check(path/name)
    dataset = Path(metas['online']['dataset'])
    data = sources.read(dataset/'dataset-private.json.gz')
    plan = sources.read(dataset/'plan.json')
    out = new_directory(args.output)
    candidates = scoring.prepare_evaluation(data, out)
    parts = {k: v for k, v in scoring.cohorts(candidates).items() if k in GROUPS}
    rows, evidence_positions = curve_members(parts)
    positions = {r['position'] for r in candidates}
    masks = {'all': [True]*len(rows), 'evidence_total': [r['position'] in evidence_positions for r in rows]}
    intervals = rolling_intervals(len(rows), plan['rolling_window'], plan['rolling_stride'])
    windows = [{'start': a+1, 'end': b, 'all_n': b-a, 'evidence_n': sum(masks['evidence_total'][a:b]),
                'start_time': local_time(data['observations'][rows[a]['position']]['timestamp']),
                'end_time': local_time(data['observations'][rows[b-1]['position']]['timestamp']),
                'labels_ready_through_index': max(r['label_ready_index'] for r in rows[a:b]),
                'off_stride_final_endpoint': b == len(rows) and (b-plan['rolling_window']) % plan['rolling_stride'] != 0}
               for a, b in intervals]
    predictions, metrics, curves = {}, {}, {}
    for arm, path in paths.items():
        predictions[arm] = sources.read(path/'predictions-private.json.gz')
        saved = sources.read(path/'metrics-private.json')
        metrics[arm], curves[arm] = {}, {}
        for name, outputs in predictions[arm].items():
            if set(map(int, outputs)) != positions:
                raise ValueError('Missing forecast')
            metrics[arm][name] = {}
            per_row = values(rows, outputs)
            curves[arm][name] = {}
            for group in GROUPS:
                m = scoring.measure(parts[group], outputs)
                if m != saved[name][group]:
                    raise ValueError('Aggregate scoring mismatch: '+name)
                metrics[arm][name][group] = m
                curve = window_curve(per_row, masks[group], intervals)
                curves[arm][name][group] = curve
                # Independently exercise the complete scoring entry point for
                # beginning, middle and final windows of every model and arm.
                for i in sorted({0, len(intervals)//2, len(intervals)-1}):
                    a, b = intervals[i]
                    subset = [r for j, r in enumerate(rows[a:b], a) if masks[group][j]]
                    check = scoring.measure(subset, outputs)
                    expected = [check['rank_score']['mean'], check['exact']['top1'], check['exact']['top4']]
                    for key, value in zip(('score', 'top1', 'top4'), expected):
                        got = curve[i][key]
                        if (got is None) != (value is None) or (value is not None and not math.isclose(got, 100*value, abs_tol=1e-9)):
                            raise ValueError('Rolling score mismatch')
        print(json.dumps({'phase': 'independent_scoring_complete', 'arm': arm, 'models': len(metrics[arm])}), flush=True)
    if set(metrics['online']) != set(metrics['frozen']) or len(metrics['online']) != 182:
        raise ValueError('Different model registries')
    for group in GROUPS:
        scoring.require_comparable(*[m[group] for arm in metrics.values() for m in arm.values()])
    # Both runs train identically before the warmup boundary. Compare forecasts
    # at every evaluation point before any post-warmup label of ANY head matures.
    later_ready = [r['ready_index'] for r in data['streams']['hard_kind'] if r['ready_index'] > plan['warmup_ready_index']]
    first_new_ready = min(later_ready)
    shared_prefix = [r for r in candidates if r['input_index'] <= first_new_ready]
    if not shared_prefix:
        raise ValueError('Need a shared post-warmup forecast for fork parity')
    for name in metrics['online']:
        for row in shared_prefix:
            p = str(row['position'])
            if predictions['online'][name][p] != predictions['frozen'][name][p]:
                raise ValueError('Warmup fork forecast mismatch: '+name)
    audits = {}
    for arm, path in paths.items():
        audit = sources.read(path.with_name(path.name+'-base')/'chronology-audit-private.json')
        if any(r['latest_training_ready_index'] >= r['input_index'] for r in audit):
            raise ValueError('Future feedback')
        if arm == 'frozen' and (len({r['mature_head0_relations'] for r in audit}) != 1 or
                              any(r['latest_training_ready_index'] > plan['warmup_ready_index'] for r in audit)):
            raise ValueError('Frozen model learned after warmup')
        audits[arm] = {'first': audit[0], 'last': audit[-1]}
    order = sorted(metrics['online'], key=lambda n: (-metrics['online'][n]['evidence_total']['rank_score']['mean'],
                                                   -metrics['online'][n]['all']['rank_score']['mean'], n))
    compact = {'plan': plan, 'windows': windows, 'order': order, 'models': {}}
    for name in order:
        compact['models'][name] = {arm: {g: [[round(v[k], 4) if v[k] is not None else None
                                             for k in ('score', 'top1', 'top4')] for v in curves[arm][name][g]]
                                       for g in GROUPS} for arm in curves}
    write(out/'rolling-metrics-private.json.gz', {'windows': windows, 'curves': curves})
    write(out/'aggregate-metrics-private.json', metrics)
    template = Path(__file__).with_name('prequential_dashboard_template.html')
    html = template.read_text().replace('/*__CURVE_DATA__*/ null', json.dumps(compact, ensure_ascii=False, separators=(',', ':')).replace('</', '<\\/'))
    if '/*__CURVE_DATA__*/' in html:
        raise ValueError('Dashboard data injection failed')
    (out/'learning-curves-private.html').write_text(html)
    summary = {'complete': True, 'models_per_arm': 182, 'arms': list(paths), 'sizes': plan['sizes'],
        'rolling_windows': len(windows), 'insufficient_evidence_windows': sum(w['evidence_n'] < 10 for w in windows),
        'fork_parity_forecasts_per_model': len(shared_prefix), 'chronology_audits': audits,
        'all_aggregate_scores_recomputed': True, 'rolling_scorer_parity': True,
        'scoring_contract': scoring.contract(), 'source_runs': {k: str(v) for k, v in paths.items()}}
    write(out/'verification.json', summary)
    report(out, metrics, curves, plan, windows, order)
    if args.plot:
        plot(out, curves, windows)
    sources.verify_unchanged()
    write(out/'sources.json', sources.accessed)
    for p in (Path(__file__), template):
        (out/('source-'+p.name)).write_bytes(p.read_bytes())
    seal(out)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def report(out, metrics, curves, plan, windows, order):
    def cells(name):
        vals = []
        for g in GROUPS:
            on = metrics['online'][name][g]['rank_score']['mean']*100
            off = metrics['frozen'][name][g]['rank_score']['mean']*100
            vals += [f'{on:.2f}', f'{off:.2f}', f'{on-off:+.2f}']
        return ' | '.join(vals)
    def header():
        return ['| 模型 | 全集在线 | 全集冻结 | 差值 | 有证据在线 | 有证据冻结 | 差值 |',
                '|---|---:|---:|---:|---:|---:|---:|']
    lines = ['# 按历史顺序学习：在线更新与预热后冻结对照', '',
        f"前{plan['warmup_actual_outcomes']}次独立成熟行为预热（{plan['warmup_relations']}条去重起点—结果关系）。之后1026条可判定候选，统一审查后完整评测 **1006条**；其中完整有证据集 **278条，覆盖率27.6%**。保留所有合法样本，包括空预测。", '',
        '182组历史配置各运行在线／冻结两条轨迹，共364组。预测时仅学习截点前已经成熟的标签；冻结轨迹不接收预热后的任何监督头、词表、转移统计或参数更新。两者仍读取相同的当时观察上下文，因而不是把输入界面冻结。', '',
        f"评测时间（美国太平洋时间）：{windows[0]['start_time']} 至 {windows[-1]['end_time']}。", '',
        '综合分0–100：精确收益1，正确服务首页回退0.6，错误0；取Top1至Top4前缀最佳收益均值，再对样本平均。它不是正确率；图中另有精确Top1、Top4正确率。', '',
        f'[交互学习曲线：全部182组](<{out}/learning-curves-private.html>)', '',
        '## 代表方案的全程成绩', '', *header()]
    for label, name in REPRESENTATIVES:
        lines.append(f'| {label} | {cells(name)} |')
    lines += ['', '## 精确目标正确率（在线；命中数／样本数）', '',
        '| 方案 | 全集Top1 | 全集Top4 | 有证据Top1 | 有证据Top4 |', '|---|---:|---:|---:|---:|']
    for label, name in REPRESENTATIVES:
        columns = [f"{metrics['online'][name][g]['exact']['top'+str(k)+'_hits']}/{metrics['online'][name][g]['n']} ({100*metrics['online'][name][g]['exact']['top'+str(k)]:.1f}%)"
                   for g in GROUPS for k in (1, 4)]
        lines.append('| '+label+' | '+' | '.join(columns)+' |')
    lines += ['', '## 滚动变化', '',
        f"最近100条合法预测为一个窗口，每10条移动一次，共{len(windows)}个窗口（末尾另补一个完整100条窗口，终点1006）。有证据曲线始终使用同一窗口中的子集，不另取最近100条有证据样本。", '',
        f"各窗口有证据样本数{min(w['evidence_n'] for w in windows)}–{max(w['evidence_n'] for w in windows)}；其中{sum(w['evidence_n']<10 for w in windows)}个窗口少于10条，明确标记样本不足，仍保留实际分数。", '',
        '| 方案 | 全集首窗→末窗（在线／冻结） | 有证据首窗→末窗（在线／冻结） |', '|---|---:|---:|']
    for label, name in REPRESENTATIVES:
        fields = []
        for g in GROUPS:
            on, off = curves['online'][name][g], curves['frozen'][name][g]
            fields.append(f"{on[0]['score']:.2f}→{on[-1]['score']:.2f} ／ {off[0]['score']:.2f}→{off[-1]['score']:.2f}")
        lines.append('| '+label+' | '+' | '.join(fields)+' |')
    lines += ['', '## 全部历史配置（描述性排序，不用本轮选参）', '', *header()]
    lines += [f'| {name} | {cells(name)} |' for name in order]
    lines += ['', '## 解释与核验边界', '',
        '- 这是历史顺序重放，包含以前用于研究和选参的数据，不是新的独立留出测试。过去仅训练用的截点现在会先预测、后学习，不能和原270／94固定总榜直接相减当提升。旧总榜未修改。',
        '- 有证据成员按统一的预测前已成熟历史定义，但分组使用真实后续目标；属于事后诊断，不是线上已知的置信保证。冻结模型也按同一分组评分，不因它没有学到后来的支持而换分母。',
        '- 所有182组沿用历史配置和种子，没有按滚动曲线调参数、挑预热长度或自动部署。KMeans中心仅用预热数据拟合，后续在线更新其计数，不声称中心也在线重训。',
        '- 横轴是预测次序，不是固定天数；窗口时长和行为难度会变化。每个点使用已观察到后续结果的样本回顾计分，不能在标签尚未成熟时提前获得这个分数。重叠窗口不是独立重复实验。',
        '- 每个预测锚点计一次，锚点可能对应同一次实际后续到达；预热独立行为数和预测样本数不是同一个分母，不把1006条称为1006次独立行为。',
        '- 在线相对冻结的差值同时包含学会新目标与改善旧目标排序的收益，不能全部解释为旧目标上的参数优化。当前可观察时间背景在两臂继续变化，冻结的是学习状态，不是全部历史输入。',
        '- 两臂全程成绩由唯一评分入口独立重算，所有配置首／中／末窗口与完整评分入口交叉核对；预热后尚无新标签成熟的公共前缀预测必须一致。',
        '- 原始采集、旧run、运行模型与安装包均未修改。个人URL与路径仅留在私有训练目录。', '']
    (out/'report-private.md').write_text('\n'.join(lines))


def plot(out, curves, windows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'Arial Unicode MS', 'font.size': 10, 'axes.unicode_minus': False})
    x = [w['end'] for w in windows]
    names = [('平滑Top2', REFERENCE, '#0072b2'), ('固定损失决策', 'risk/fixed', '#d55e00')]
    for metric, title in [('score', '综合分'), ('top1', '精确Top1正确率（%）'), ('top4', '精确Top4正确率（%）')]:
        fig, axes = plt.subplots(2, 1, figsize=(11, 7.3), sharex=True, layout='constrained')
        for ax, group, label in zip(axes, GROUPS, ('完整合法预测', '同窗口有证据子集')):
            for name, key, color in names:
                for arm, style, suffix in [('online', '-', '持续学习'), ('frozen', '--', '预热后冻结')]:
                    ax.plot(x, [v[metric] for v in curves[arm][key][group]], linestyle=style, color=color,
                            linewidth=1.65, label=name+' · '+suffix)
            if group == 'evidence_total':
                for i, w in enumerate(windows):
                    if w['evidence_n'] < 10:
                        ax.axvspan(w['end']-5, w['end']+5, color='#999999', alpha=.2)
            ax.set(title=label, ylabel=title, ylim=(-2, 102), xlim=(x[0], x[-1]))
            ax.grid(alpha=.18)
        axes[0].legend(loc='upper left', ncol=2, fontsize=9)
        axes[-1].set_xlabel('预热后预测样本序号（最近100条；每10条更新）')
        fig.suptitle('历史顺序回放 · '+title+'\n预热100次独立成熟行为；灰带表示有证据样本不足10条', fontsize=13)
        fig.savefig(out/f'rolling-{metric}.png', dpi=160)
        plt.close(fig)


if __name__ == '__main__':
    main()
