#!/usr/bin/env python3
"""Describe late-test performance by available historical supervision, without tuning."""
import argparse
import json
import os
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

import evaluate_context_knn as core


def verify_run(path):
    hashes=json.loads((path/'artifact-hashes.json').read_text())
    for name,digest in hashes.items():
        if Path(name).name!=name or core.sha(path/name)!=digest:raise ValueError('Run artifact changed')
    return json.loads((path/'run.json').read_text()),hashes


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tuning-run',required=True,type=Path)
    parser.add_argument('--diagnosis-run',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();out=args.output.expanduser().resolve()
    if not out.is_relative_to(core.PRIVATE_ROOT.resolve()) or out==core.PRIVATE_ROOT.resolve():
        parser.error('Use a new private training directory')
    provenance,tune_hashes=verify_run(args.tuning_run)
    diagnosis,diagnosis_hashes=verify_run(args.diagnosis_run)
    assert provenance['source_run']==diagnosis['source_run'],'Different labels / original runs'
    base=Path(provenance['source_run']);verify_run(base)
    metrics=json.loads((args.tuning_run/'metrics.json').read_text())
    preds=json.loads((args.tuning_run/'predictions-private.json').read_text())
    previous=json.loads((base/'predictions-private.json').read_text())
    cases=json.loads((args.diagnosis_run/'cases-private.json').read_text())
    ledger=json.loads((base/'labels-private.json').read_text())
    events=core.load_lines(Path(provenance['snapshot'])/'events.jsonl')
    byid={e['id']:e for e in events}
    plan=json.loads((args.tuning_run/'search-plan.json').read_text())
    def local_time(r):
        return datetime.fromtimestamp(byid[r['anchor_id']]['timestampMs']/1000,ZoneInfo('America/Los_Angeles')).isoformat()
    exemplar=next(iter(ledger.values()));train_end=plan['train_end_original_position'];test_start=plan['test_start_original_position']
    timing={'timezone':'America/Los_Angeles','train_first':local_time(exemplar[0]),
        'validation_first':local_time(exemplar[train_end]),'test_first':local_time(exemplar[test_start]),
        'test_last':local_time(exemplar[-1]),'original_sample_count':len(exemplar),
        'train_n':train_end,'validation_n':test_start-train_end,'test_n':len(exemplar)-test_start}
    output={'timing':timing,'results':{}}
    for t,r in metrics['results'].items():
        cs={x['position']:x for x in cases[t]}
        models={'selected':preds[t]['selected'],'recency':preds[t]['recency'],
            'previous_parameters':previous[t]['knn_previous_validation_selected']}
        for rows in models.values():
            assert set(cs)=={x['position'] for x in rows}
        groups={'all_late_test':set(cs),
            'has_mature_target_history':{i for i,c in cs.items() if c['mature_clean_support']>=1},
            'no_mature_target_history':{i for i,c in cs.items() if c['mature_clean_support']==0},
            'at_least_3_mature_target_examples':{i for i,c in cs.items() if c['mature_clean_support']>=3},
            'target_in_candidates':{x['position'] for x in models['selected'] if x['covered']}}
        output['results'][t]={'best_config':r['best_config'],'groups':{}}
        for name,positions in groups.items():
            output['results'][t]['groups'][name]={m:core.metric([x for x in rows if x['position'] in positions]) for m,rows in models.items()}
    os.umask(0o077);out.mkdir(parents=True,mode=0o700,exist_ok=False)
    core.write_json(out/'metrics.json',output)
    lines=['# 后期测试：有无历史学习样本的分层结果','',
        f"采用原始时间顺序 60% / 20% / 20% 分割：{timing['train_n']} 个积累学习、{timing['validation_n']} 个验证、{timing['test_n']} 个后期测试截点（清洗后数量减少）。",
        f"验证开始：{timing['validation_first']}；测试：{timing['test_first']} 至 {timing['test_last']}，时区 America/Los_Angeles。",'',
        '参数仍然只按整个合格验证段选择；下面的分组仅用于理解结果，没有为有历史子集重新选参。已有历史指预测截点前，至少有一个已经满 30 秒、标签为该具体目标的清洗后样本。不是仅仅出现在后台，也不使用之后才观察到的样本。',
        '“有历史”按本次实际正确目标划分，是事后诊断子集，不能视为在线系统可以预先识别的全部请求。','',
        '| 阈值 | 后期测试分组 | 数量 | 新参数 Top1 | 新参数 Top4 | 旧参数 Top4 | 近期访问 Top4 |',
        '|---|---|---:|---:|---:|---:|---:|']
    names={'all_late_test':'全部后期测试','has_mature_target_history':'已有至少 1 个成熟历史样本',
        'no_mature_target_history':'没有成熟历史样本','at_least_3_mature_target_examples':'已有至少 3 个成熟历史样本',
        'target_in_candidates':'正确目标在候选中'}
    for t,r in output['results'].items():
        for group,m in r['groups'].items():
            s=m['selected'];lines.append(f"| {t} 秒 | {names[group]} | {s['n']} | {core.percent(s['top1'])} | {core.percent(s['top4'])} | {core.percent(m['previous_parameters']['top4'])} | {core.percent(m['recency']['top4'])} |")
    lines += ['', '主结果仍是全部后期测试；后面的子集可能重叠，不相加。不同清洗阈值改变标签和样本，不能直接据此断言哪种行为定义正确。']
    (out/'report-private.md').write_text('\n'.join(lines)+'\n')
    (out/'analysis-source.py').write_text(Path(__file__).read_text())
    core.write_json(out/'run.json',{'tuning_run':str(args.tuning_run.resolve()),'tuning_artifact_hashes':tune_hashes,
        'diagnosis_run':str(args.diagnosis_run.resolve()),'diagnosis_artifact_hashes':diagnosis_hashes,
        'selection_changed':False,'source_runs_modified':False})
    core.write_json(out/'artifact-hashes.json',{p.name:core.sha(p) for p in out.iterdir() if p.is_file()})
    print(json.dumps({'output':str(out),**output},ensure_ascii=False))


if __name__=='__main__':main()
