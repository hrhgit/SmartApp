#!/usr/bin/env python3
"""Human-readable consolidation of completed private action/cleaning runs."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path

import compare_online_models as comparison
import evaluate_context_knn as io
from summarize_tuned_history import verify_run


def fmt(metric):
    return f"{metric['top4_hits']}/{metric['n']}（{metric['top4']:.1%}）" if metric['n'] else '0/0'


def architecture(config):
    family, p = config['family'], config['parameters']
    if family == 'same_origin_evidence_gate':
        return architecture(p['base']) + '；仅保留该清洗下同当前界面已有成熟转移证据的目标'
    if family == 'rank_fusion_ensemble':
        return (f"倒数排名融合：{1-p['right_weight']:.0%} × [{architecture(p['left'])}] + "
                f"{p['right_weight']:.0%} × [{architecture(p['right'])}]，排名偏移 {p['rank_offset']}")
    if family == 'bayesian_smoothing':
        return f"带先验平滑的当前界面转移频次（先验 {p['prior']}，强度 {p['alpha']}）"
    if family == 'recency':
        scope = {'global': '全局', 'app': '当前应用内', 'conditional': '当前界面内'}[p['scope']]
        return f"{scope}近期到达排序（使用已经成熟的真实到达）"
    if family == 'adaptive_hoeffding_tree':
        return (f"在线自适应 Hoeffding 决策树（分裂检查间隔 {p['grace_period']} 条、"
                f"delta={p['delta']}、叶节点按类别频次预测）")
    if family in {'softmax', 'logistic_ovr'}:
        name = '在线 Softmax' if family == 'softmax' else '在线一对多逻辑回归'
        return f"{name}（学习率 {p['learning_rate']}，L2 {p['l2']}）"
    return f"{family}（{json.dumps(p, ensure_ascii=False)}）"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--main-run', required=True, type=Path)
    parser.add_argument('--hierarchy-run', required=True, type=Path)
    parser.add_argument('--validation-json', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    if not out.is_relative_to(io.PRIVATE_ROOT.resolve()) or out == io.PRIVATE_ROOT.resolve():
        parser.error('Use a new private training run')
    runs = [args.main_run.resolve(), args.hierarchy_run.resolve()]
    hashes = {}
    for path in runs:
        provenance, checked = verify_run(path)
        if provenance.get('complete') is False:
            raise ValueError('Incomplete source run')
        hashes[str(path)] = checked
    primary = json.loads((runs[0] / 'selection.json').read_text())
    hierarchy = json.loads((runs[1] / 'selection.json').read_text())
    candidates = [(runs[0], primary['best_current_only']), (runs[1], hierarchy['best_current_only'])]
    source, best = min(candidates, key=lambda pair: comparison.selection_key(pair[1]))
    ablations = json.loads((source / 'cleaning-ablation.json').read_text())[best['config']['id']]
    cleaning = json.loads((runs[0] / 'cleaning-summary.json').read_text())
    names = {p['id']: p['name_zh'] for p in cleaning['policies']}
    errors = json.loads((source / 'selected-errors-private.json').read_text())
    if isinstance(errors, dict):
        errors = errors[best['training_cleaning'] + ':' + best['config']['id']]
    scores = best['scores']['5']
    validation = json.loads(args.validation_json.read_text()) if args.validation_json else None
    os.umask(0o077)
    out.mkdir(parents=True, mode=0o700, exist_ok=False)
    result = {'source_run': str(source), 'selected': best, 'fixed_architecture_cleaning': ablations,
              'source_hashes': hashes, 'hierarchy_experiment': hierarchy,
              'cleaning_policy_names': names, 'remaining_errors': errors,
              'independent_validation': validation,
              'test_previously_seen': True, 'online_model_installed': False}
    io.write_json(out / 'summary.json', result)
    io.write_json(out / 'errors-private.json', errors)
    if validation is not None:
        io.write_json(out / 'independent-validation.json', validation)
    old = primary['old_recommendations_new_metric']['5']['supported']
    lines = ['# 快速模型：可启动目标与清洗方案的实验结果', '',
        f"最佳前四命中：**{fmt(scores['supported'])}**；第一名命中：**{scores['supported']['top1_hits']}/{scores['supported']['n']}**。",
        f"原更后期38条：{fmt(scores['late_supported'])}。旧模型同一动作评分口径：{fmt(old)}。", '',
        '评分保留固定的62条有证据样本。普通窗口按所属应用的激活/启动动作评分；网页、目录和文件仍需命中具体目的地。应用命中不宣称恢复了原窗口或页面。此数据反复用于选择模型和参数，属于开发回放，不能解释为新数据上的稳定准确率。', '',
        '## 数据清洗', '',
        f"本轮实际探索19种训练清洗，最终选择：**{names[best['training_cleaning']]}**。清洗仅改变历史学习标签，评分样本与答案保持不变。", '',
        '范围包括：原始到达、仅合并同次到达补全、1/2/3/5/8/10秒、只合并确证同标签页导航、只处理快速返回、确证导航与跨目标切换、按目标类型区分，以及依据过去停留分布自适应估计阈值。', '',
        '原始事件不删除；缺失、失败和中断不当作短暂停留。自适应阈值只依据预测截点之前已经完整结束的停留。所有方案均等原30秒观察窗成熟后才学习，并按当前界面与实际后续到达去重。', '',
        '| 训练清洗 | 全历史合格训练对数 | 固定最终架构 Top4 | Top1 |', '|---|---:|---:|---:|']
    for row in ablations:
        name = row['training_cleaning']
        m = row['scores']['5']['supported']
        lines.append(f"| {names[name]} | {cleaning['summaries'][name]['unique_origin_outcomes']} | {fmt(m)} | {m['top1_hits']}/{m['n']} |")
    if best['training_cleaning'] == 'kind_app5_resource2':
        lines += ['', '入选规则：应用和普通窗口跳过严格小于5秒的已完成停留，网页、目录和文件跳过严格小于2秒的已完成停留；达到阈值即保留。同次到达补全后按最终可确认类型判断，短暂返回当前目标时结束，不继续寻找更后面的目标。正常导航中间步骤单独标记，不直接认定为误选。', '',
            '相对统一5秒，该规则保留更多2至不足5秒的具体资源访问。短于阈值只表示在本轮派生标签中跳过，并不证明该次访问是错误。']
    lines += ['', '训练对总数描述整份历史；每次预测仍只使用当时已成熟的那一部分。以上固定架构、固定参数的横向表才用于比较清洗，不能把各清洗分别选出的不同模型分数全归因于清洗。', '',
        '## 模型架构', '', architecture(best['config']), '',
        '输入仍是当前界面（当前目标或窗口类别、所属应用、目标类型），没有加入前序界面序列或后台相似度。清洗后的下一步行为一旦成熟，就逐条更新模型；即使没有打开推荐菜单，自然行为也可进入训练。', '',
        '候选按截点前的准入和定位符构建：已知应用可激活或启动，已知具体URL/路径可重新打开。后续才出现的目标不会倒填为之前的候选；没有实时窗口记录也不会直接当作确认关闭。', '',
        '```json', json.dumps(best['config'], ensure_ascii=False, indent=2), '```', '',
        f"主比较包含27个基础统计/在线模型家族、{primary['base_config_count']}组参数、{primary['base_replay_count']}次基础回放、{primary['fusion_count']}组融合和{primary['evidence_gate_count']}组同起点证据限制。另有双层监督补充实验，区分主目标与应用层评分。没有声称穷尽所有可能的算法。家族榜单包含额外时钟、库存近期性对照，最终主结论限制为 current_only 配置。", '',
        f"最终完整架构本机离线重放：单次增量学习中位数 {best['timing']['learn']['median_ms']:.3f} ms，单次预测中位数 {best['timing']['predict']['median_ms']:.3f} ms，预测 P95 {best['timing']['predict']['p95_ms']:.3f} ms。这是当前数据量下的 Python 回放计时，并非已集成菜单的端到端延迟。", '',
        '完整模型家族排名和较早样本选参对照见[主实验报告](../' + runs[0].name + '/report-private.md)。', '',
        '## 证据与目标分组', '', '| 分组 | Top4 | Top1 |', '|---|---:|---:|']
    for name, m in scores['grades'].items():
        lines.append(f"| { {'strong':'反复且集中','repeated':'重复支持','single':'单次支持'}[name]} | {fmt(m)} | {m['top1_hits']}/{m['n']} |")
    for name, m in scores['by_original_kind'].items():
        if m['n']:
            lines.append(f"| 原记录类型 {name} | {fmt(m)} | {m['top1_hits']}/{m['n']} |")
    lines += ['', f"去掉与当前前台相同应用、操作意义含糊的案例后：{fmt(scores['without_same_foreground_app_ambiguous'])}；这些案例仍保留在主62条中。严格来源完整子集：{fmt(scores['strict_supported'])}。", '',
        f"不筛证据、在全部162条上：{fmt(scores['all'])}。该模型刻意只保留同起点有历史证据的推荐，不能把有证据子集上的成绩当作任意情景的准确率；无证据情景的回退模型尚未纳入本轮目标。", '',
        '## 还没有预测中的情景', '', '| 当前界面 | 实际下一步 | 该清洗下支持/起点总数 | 排名 | 原因 |', '|---|---|---:|---:|---|']
    for e in errors:
        reason = {'cleaning_removed_same_origin_support': '此清洗下没有保留下同起点的该目标证据',
                  'rank_below_four': '存在历史支持，但排名低于前四',
                  'candidate_unavailable': '截点时没有可用目标依据',
                  'hierarchy_suppression': '展示时被层级去重抑制'}.get(e['reason'], e['reason'])
        rank = e['raw_rank'] if e['raw_rank'] is not None else '未进入证据候选'
        lines.append(f"| {e['origin']} | {e['actual']} | {e['training_support']}/{e['training_origin_total']} | {rank} | {reason} |")
    lines += ['', '支持数来自各自训练清洗；证据分组和评分成员仍沿用冻结定义。样本因有真实下一步的历史支持而被事后挑出，不等于线上能预先知道本次就属于这个集合。', '',
        '## 双层监督的独立对照', '',
        '将同一次到达的所属应用也用于辅助模型训练，没有把它计成另一次到达。最终评分仍区分应用动作与具体资源，应用分数过高可能挤占页面或目录的推荐名额。', '',
        '| 辅助应用层权重 | 该权重下最佳Top4 | 训练清洗 |', '|---|---:|---|']
    for weight, row in hierarchy['best_per_app_weight'].items():
        lines.append(f"| {weight} | {fmt(row['scores']['5']['supported'])} | {names[row['training_cleaning']]} |")
    if validation:
        lines += ['', '## 独立复核', '']
        lines += ['- ' + note for note in validation.get('notes', [])]
    lines += ['', '全部是离线实验，尚未替换线上菜单，也未实测每项应用启动或具体资源重新打开的操作系统执行结果。原始数据、既有完成run和实时采集均保持原样。', '',
        f"主实验：{runs[0]}", f"双层实验：{runs[1]}", f"最终预测来源：{source}", '']
    (out / 'report-private.md').write_text('\n'.join(lines))
    (out / ('source-' + Path(__file__).name)).write_bytes(Path(__file__).read_bytes())
    io.write_json(out / 'run.json', {'version': 'action-cleaning-consolidation-v1',
        'sources': hashes, 'selected_source': str(source), 'complete': True,
        'source_runs_modified': False, 'test_previously_seen': True})
    io.write_json(out / 'artifact-hashes.json', {p.name: io.sha(p) for p in out.iterdir() if p.is_file()})
    print(json.dumps({'complete': True, 'source': str(source), 'output': str(out),
        'cleaning': names[best['training_cleaning']], 'main': scores['supported'],
        'errors': dict(Counter(e['reason'] for e in errors))}, ensure_ascii=False))


if __name__ == '__main__':
    main()
