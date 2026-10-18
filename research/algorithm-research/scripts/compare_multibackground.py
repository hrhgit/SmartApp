"""Chronological training of current-only, learned multi-scale backgrounds."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
import evaluation_contract as scoring

import numpy as np

import evaluate_context_knn as io
from compare_semantic_trajectory import groups, training_rows
from compare_temporal_context import selection_key
from multibackground_model import JointNavigationModel, SCALES, VERSION, prepare_inputs
from navigation_hierarchy import NavigationHierarchy, SERVICE, learning_record, service
from semantic_online import blend, normalize, rank
from summarize_tuned_history import verify_run


SOURCES=('evaluation_contract.py','known_inactivity.py','compare_multibackground.py','multibackground_model.py','prepare_multibackground_dataset.py',
         'temporal_context.py','navigation_hierarchy.py','semantic_online.py','semantic_annotations.py',
         'diagnose_action_granularity.py','trajectory_supervision.py','compare_semantic_trajectory.py',
         'compare_temporal_context.py','evaluate_context_knn.py','summarize_tuned_history.py')


def configurations():
    result=[]
    for family in ('interaction_softmax','mlp'):
        for lr in (.03,.1):
            for background in ('none','recent','long','combined'):
                result.append({'id':f'{family}_{background}_rich_lr{lr}',
                    'family':family,'backgrounds':background,'channels':'rich','lr':lr,
                    'dimensions':2048,'rehearsal':7,'seed':20260923})
            result.append({'id':f'{family}_combined_count_lr{lr}',
                'family':family,'backgrounds':'combined','channels':'count','lr':lr,
                'dimensions':2048,'rehearsal':7,'seed':20260923})
    return result


def advance(model,stream,cursor,inputs,cutoff):
    while cursor<len(stream) and stream[cursor]['ready_index']<cutoff:
        row=stream[cursor];q=inputs[row['position']]
        if (q['index']>=cutoff or q['index']>row['ready_index']
                or q['background_through_index']!=q['index']
                or row['record']['target_evidence_index']>row['ready_index']):
            raise ValueError('Training input or outcome was not available at its declared cutoff')
        record=learning_record(row,q)
        if isinstance(model,JointNavigationModel):model.learn(q,record,row['ready_index'])
        else:model.learn(record)
        cursor+=1
    return cursor


def bundle(scores,home):
    raw=rank(scores,10**9)
    resolved=[home if home and label==SERVICE else label for label in raw]
    resolved=list(dict.fromkeys(resolved))
    return {'predictions':resolved[:4],'raw':raw[:4],'rankable':resolved}


def replay_baseline(inputs,stream,evaluation,homes):
    model=NavigationHierarchy();cursor=0;scores={};outputs={}
    for row in sorted(evaluation,key=lambda r:r['position']):
        pos=row['position'];q=inputs[pos]
        cursor=advance(model,stream,cursor,inputs,q['index'])
        scores[pos]=normalize(model.scores(q['target']))
        outputs[pos]=bundle(scores[pos],homes[pos])
    return scores,outputs


def replay_joint(inputs,stream,evaluation,homes,baseline,config,asof_index):
    model=JointNavigationModel(config);cursor=0;outputs={1.:{},.5:{}};latencies=[]
    for row in sorted(evaluation,key=lambda r:r['position']):
        pos=row['position'];q=inputs[pos]
        cursor=advance(model,stream,cursor,inputs,q['index'])
        started=time.perf_counter();scores=model.scores(q)
        latencies.append((time.perf_counter()-started)*1000)
        for joint_weight in outputs:
            mixed=scores if joint_weight==1. else blend([scores,baseline[pos]],[joint_weight,1-joint_weight])
            outputs[joint_weight][pos]=bundle(mixed,homes[pos])
    advance(model,stream,cursor,inputs,asof_index+1)
    return model,outputs,latencies


metrics = scoring.grouped_metrics


def choose(results,background=None):
    values=[r for r in results if background is None or r['config'].get('backgrounds')==background]
    return max(values,key=lambda r:selection_key(r['validation']))


def cell(m,key):
    return '—' if not m['n'] else f"{m[key+'_hits']}/{m['n']} ({m[key]:.1%})"


def report(summary):
    baseline=summary['baseline'];selected=summary['selected_combined']
    lines=['# 当前目标与多时间尺度背景：重新训练', '',
           '前台有序序列固定为长度 1，只输入当前目标。过去的目标只进入无序统计背景，没有前一步、最近路径或顺序后缀特征。',
           '后台状态继续采集，本轮模型不读取后台应用、窗口、标签页、窗格或推荐候选库存。', '',
           f"数据新增 {summary['data']['new_observations']} 个合格训练截点，新增 {summary['data']['new_evaluation_n']} 个可评分样本。旧 162 个评测样本、清洗和标签保持不变。",
           '新数据中返回当前目标或缺少足够停留证据的截点按原规则不评分；它们的已确认行为仍可进入背景，成熟且符合原训练规则的标签仍用于训练。', '',
           '## 学习内容', '',
           '- 近期背景：过去 5 分钟和 60 分钟；较长背景：过去 24 小时和截至截点的全部记录。',
           '- 各尺度分别保留独立到达频次、最近访问的衰减值、已观察的非空闲停留及覆盖标记；频次单通道作为消融。',
           '- 比较直接学习交互系数的 softmax 和带线性分支的 32 单元 tanh 神经网络。模型输出类别只从已成熟标签增加，不由未来标签或当前库存预先限定。',
           '- 每个新成熟样本触发一次 AdaGrad 更新，并随机复习最多 7 个此前成熟样本；复习次数不当作新增独立行为。',
           '- 每套权重同时评分纯学习输出与 50% 旧统计模型融合。融合权重预先固定，参数与背景组合仅按旧较早验证段选择。',
           f"- 长期背景当前最多覆盖约 {summary['recorded_days']:.2f} 天记录，还不足以代表更长期的稳定习惯。", '',
           f"按较早验证段选出的多背景配置：`{selected['config']['id']}`。", '',
           summary['score_scope'], '| 评测范围 | 模型 | 接受 Top1 | 接受 Top4 |',
           '|---|---|---:|---:|']
    for split,group,label in [('validation','original','旧较早验证／固定有证据'),
                              ('test','original','旧较晚／固定有证据'),
                              ('old_full','original','旧全部／固定有证据'),
                              ('new_test','new_exact_history','新增／同起点有证据'),
                              ('new_test','expanded','新增／含语义有证据'),
                              ('new_test','all','新增／全部'),('old_full','all','旧全部')]:
        for name,result in [('旧统计模型',baseline),('当前＋近期＋长期',selected)]:
            m=result[split][group]
            lines.append(f"| {label} | {name} | {cell(m,'top1')} | {cell(m,'top4')} |")
    lines+=['','## 背景消融','','各组分别仅用同一早期验证段选参，故此表包含模型参数选择的差异；同架构、同学习率的所有对照见 results.json。', '',
            '| 背景组 | 选中配置 | 验证有证据 Top4 | 旧较晚有证据 Top4 | 新增全部 Top4 |',
            '|---|---|---:|---:|---:|']
    for name,r in summary['selected_by_background'].items():
        lines.append(f"| {name} | `{r['config']['id']}` | {cell(r['validation']['original'],'top4')} | {cell(r['test']['original'],'top4')} | {cell(r['new_test']['all'],'top4')} |")
    lines+=['','## 解释边界与验证','','- 旧评测集已在多轮开发中查看，不能声称是全新独立测试。新增数据在本轮选参时保留，不用其分数挑配置。',
            '- 新增评测按时间在线进行：每次预测只学习该截点之前已成熟的标签；前面已成熟的新数据可以帮助后面的预测。最终保存的权重才包含全部新成熟样本。',
            '- “有证据”是依据真实后续目标诊断的分组，并非线上置信门槛。新增全部样本仍是有可观测下一步且通过标签规则的截点，不能代表所有使用时刻。',
            '- 统一审查样本合法性，服务接受与精确目标分数分别保存；当前应用的粗粒度标签退出分母。分数受模型、数据量和固定输出语义共同影响。',
            '- 统计背景分别归一化；多尺度重复计入特征不构成多个独立到达或额外训练样本。暂停、中断、初始/恢复、补全和空闲沿用显式观测边界。',
            f"- 旧运行时基线预测不一致：{summary['baseline_parity_mismatches']}；原始快照确为旧快照的完整前缀扩展；新旧来源及脚本均校验哈希。",
            '- 保存了真实权重、优化器状态、已成熟训练记忆及随机数状态，重载预测与继续增量更新均检查一致。产物仅用于离线研究，没有部署。', '',
            '## 模型产物','']
    for name,m in summary['saved_models'].items():
        lines.append(f"- {name}: `{m['state_file']}`；{m['records']} 个去重训练关系，其中 {m['navigation_updates']} 个导航关系参与参数学习，{m['new_records']} 个来自新增截点。")
    return '\n'.join(lines)+'\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--baseline-run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.expanduser().resolve()
    if not out.is_relative_to(io.PRIVATE_ROOT.resolve()) or out==io.PRIVATE_ROOT.resolve() or out.exists():
        parser.error('Use a new private SmartApp training run')
    provenance,dataset_hashes=verify_run(args.dataset)
    _,baseline_hashes=verify_run(args.baseline_run)
    snapshot=Path(provenance['snapshot']);expected=provenance['snapshot_hashes']['events.jsonl']
    if io.sha(snapshot/'events.jsonl')!=expected:raise ValueError('Raw snapshot changed')
    with gzip.open(args.dataset/'dataset-private.json.gz','rt') as f:data=json.load(f)
    data_summary=json.loads((args.dataset/'summary.json').read_text())
    events=io.load_lines(snapshot/'events.jsonl')
    sources={n:Path(__file__).with_name(n).read_bytes() for n in SOURCES}
    inputs,homes,audit=prepare_inputs(events,data['observations'],io.KNOWN_TEST_SESSIONS)
    print(json.dumps({'phase':'features_complete','inputs':len(inputs),'audit':audit}),flush=True)
    stream=[r for r in training_rows(data,'hard_kind') if r['head']==0]
    evaluation=scoring.prepare_evaluation(data);old=[r for r in evaluation if r['split']!='new_test']
    new=[r for r in evaluation if r['split']=='new_test']
    test=[r for r in evaluation if r['split']=='test']
    test_index=min(inputs[r['position']]['index'] for r in test)
    validation=[r for r in evaluation if r['split']=='validation'
                and data['observations'][r['position']]['ready_index']<test_index]
    if not validation or not any(r['evidence_grade']=='original' for r in validation):
        raise ValueError('Missing mature old validation cohort')
    configs=configurations()
    os.umask(0o077);out.mkdir(mode=0o700,parents=True,exist_ok=False)
    scoring.prepare_evaluation(data, out)
    plan={'version':VERSION,'configs':configs,'joint_weights':[1.,.5],
          'current_sequence_length':1,'background_scales_seconds':SCALES,
          'ordered_past_features':False,'background_inventory_features':False,
          'label_cleaning':'hard_kind first retained label, unchanged',
          'score_labels':'old exact labels frozen; new uses same stable5s scoring policy',
          'selection':'old validation original Top4, expanded Top4, original Top1, expanded Top1, all Top4, all Top1; first config on tie',
          'validation_n':len(validation),'old_n':len(old),'new_n':len(new),
          'validation_maturity_before_index':test_index,'new_data_used_to_choose_config':False,
          'online_learning_from_mature_new_labels':True,'rehearsal_previous_examples':7,
          'source_hashes':{n:hashlib.sha256(b).hexdigest() for n,b in sources.items()},
          'dataset_hashes':dataset_hashes,'baseline_hashes':baseline_hashes}
    io.write_json(out/'plan.json',plan)
    with gzip.open(out/'inputs-private.json.gz','wt') as f:json.dump(inputs,f,ensure_ascii=False)
    baseline_scores,baseline_outputs=replay_baseline(inputs,stream,evaluation,homes)
    expected_predictions=json.loads((args.baseline_run/'predictions-private.json').read_text())['v2_exact']
    mismatches=sum(baseline_outputs[r['position']]['predictions']!=expected_predictions[str(r['position'])] for r in old)
    if mismatches:raise ValueError(f'Old baseline parity failed: {mismatches}')
    baseline={'config':{'id':'current_statistics'},'validation':metrics(validation,baseline_outputs)}
    results=[];all_outputs={'current_statistics':baseline_outputs};retained={}
    for config in configs:
        fitted,outputs,latencies=replay_joint(inputs,stream,evaluation,homes,baseline_scores,config,len(events)-1)
        for weight,values in outputs.items():
            name=config['id']+('_pure' if weight==1. else '_mix50')
            setting={**config,'id':name,'joint_weight':weight}
            result={'config':setting,'validation':metrics(validation,values),
                    'prediction_ms_median':statistics.median(latencies),'prediction_ms_max':max(latencies)}
            results.append(result);all_outputs[name]=values
        # Retain only validation winners' final weights. No late/new score is read.
        winners={mode:choose(results,mode) for mode in {r['config']['backgrounds'] for r in results}}
        for mode,chosen in winners.items():
            chosen_base=chosen['config']['id'].rsplit('_',1)[0]
            if chosen_base==config['id']:retained[mode]=fitted
        print(json.dumps({'finished':config['id'],
                          'pure_validation_top4':results[-2]['validation']['original']['top4_hits'],
                          'mixed_validation_top4':results[-1]['validation']['original']['top4_hits']}),flush=True)
    selected_by_background={mode:choose(results,mode) for mode in ('none','recent','long','combined')}
    selected=selected_by_background['combined']
    io.write_json(out/'selection.json',{'selected_combined':deepcopy(selected),
        'selected_by_background':deepcopy(selected_by_background),
        'selection_uses_old_test_or_new_labels':False})
    for result in [baseline]+results:
        outputs=all_outputs[result['config']['id']]
        for name,rows in [('test',test),('old_full',old),('new_test',new)]:
            result[name]=metrics(rows,outputs)
            result['exact_'+name]={g: m['exact'] for g, m in metrics(rows,outputs).items()}
    saved={}
    for mode,result in selected_by_background.items():
        fitted=retained[mode];fitted.config=deepcopy(result['config'])
        state=fitted.save(out,'model-'+mode,len(events)-1,
                          {'dataset':str(args.dataset.resolve()),'events_sha256':expected,
                           'selection':'old early validation only','asof_timestamp_ms':events[-1]['timestampMs']})
        restored=JointNavigationModel.load(state)
        for probe in inputs[-3:]:
            a=fitted.serving_scores(probe);b=restored.serving_scores(probe)
            if a.keys()!=b.keys() or not np.allclose(list(a.values()),list(b.values()),rtol=0,atol=1e-12):
                raise ValueError('Saved trained model prediction differs')
        # Check future incremental updates preserve optimizer/rehearsal/RNG state.
        # These probe copies are never scored as causal historical predictions.
        if fitted.classifier.memory:
            q=inputs[-1];label=fitted.classifier.labels[0]
            restored.classifier.learn(q,label)
            c=deepcopy(fitted.classifier);c.learn(q,label)
            if any(not np.array_equal(a,restored.classifier.state_arrays()[name]) for name,a in c.state_arrays().items()):
                raise ValueError('Saved optimizer cannot resume identically')
        saved[mode]={'state_file':state.name,'weights_file':'model-'+mode+'-weights-private.npz',
            'config':result['config'],'records':len(fitted.training_rows),
            'navigation_updates':fitted.classifier.update_count,'classes':len(fitted.classifier.labels),
            'new_records':sum(r['input']['index']>=provenance['new_data_start_index'] for r in fitted.training_rows),
            'reload_score_parity':True,'resume_optimizer_parity':True}
    coverage={}
    for scale in SCALES:
        values=[inputs[r['position']]['backgrounds'][scale] for r in evaluation]
        coverage[scale]={'n':len(values),'with_confirmed_arrivals':sum(v['confirmed_arrivals']>0 for v in values),
            'median_arrivals':statistics.median(v['confirmed_arrivals'] for v in values),
            'left_truncated_n':sum(v['left_truncated'] for v in values)}
    summary={'score_scope':scoring.report_scope(evaluation),'scoring_contract':scoring.contract(),'version':VERSION,'fitted_config_count':len(configs),'scored_config_count':len(results)+1,
        'data':data_summary,'baseline':baseline,'selected_combined':selected,
        'selected_by_background':selected_by_background,'saved_models':saved,
        'baseline_parity_mismatches':mismatches,'feature_audit':audit,'coverage':coverage,
        'recorded_days':max(q['recorded_history_seconds'] for q in inputs)/86400,
        'current_sequence_length':1,'new_data_used_to_choose_config':False,
        'collection_or_runtime_changed':False}
    io.write_json(out/'summary.json',summary);io.write_json(out/'results.json',[baseline]+results)
    io.write_json(out/'predictions-private.json',all_outputs)
    (out/'report-private.md').write_text(report(summary),encoding='utf-8')
    if (verify_run(args.dataset)[1]!=dataset_hashes or verify_run(args.baseline_run)[1]!=baseline_hashes
            or io.sha(snapshot/'events.jsonl')!=expected):raise ValueError('Input changed during training')
    for name,content in sources.items():
        if Path(__file__).with_name(name).read_bytes()!=content:raise ValueError('Source changed during training')
        (out/('source-'+name)).write_bytes(content)
    io.write_json(out/'run.json',{'version':VERSION,'complete':True,'dataset':str(args.dataset.resolve()),
        'snapshot':str(snapshot),'dataset_hashes':dataset_hashes,'events_sha256':expected,
        'baseline_run':str(args.baseline_run.resolve()),'baseline_hashes':baseline_hashes,
        'source_hashes':plan['source_hashes'],'created_at':datetime.now(timezone.utc).isoformat(),
        'input_and_source_hashes_verified_before_after':True,'saved_models':saved,
        'collection_or_runtime_changed':False,'raw_modified':False})
    io.write_json(out/'artifact-hashes.json',{p.name:io.sha(p) for p in out.iterdir() if p.is_file()})
    if any(p.stat().st_mode&0o777!=0o600 for p in out.iterdir() if p.is_file()):
        raise ValueError('Private artifacts must have mode 0600')
    print(json.dumps({'output':str(out),'selected':selected['config'],
                     'baseline_new':baseline['new_test']['all'],'selected_new':selected['new_test']['all'],
                     'saved_models':saved},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
