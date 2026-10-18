"""Audit labels and re-score immutable predictions through the common evaluator."""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path

import evaluation_contract as scoring


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(path):
    hashes = json.loads((path/'artifact-hashes.json').read_text())
    for name, value in hashes.items():
        if Path(name).name != name or sha(path/name) != value:
            raise ValueError('Changed source artifact: '+name)
    return hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--prediction-run', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    root = (Path.home()/'Library/Application Support/SmartApp/training').resolve()
    if out.exists() or out == root or not out.is_relative_to(root):
        parser.error('Use a new private training run')
    sources = list(dict.fromkeys([args.dataset.resolve(), *[p.resolve() for p in args.prediction_run]]))
    hashes = {str(p): verify(p) for p in sources}
    with gzip.open(args.dataset/'dataset-private.json.gz', 'rt') as f:
        data = json.load(f)
    os.umask(0o077); out.mkdir(mode=0o700)
    rows = scoring.prepare_evaluation(data, out); parts = scoring.cohorts(rows)
    scores = {}; predictions = {}
    for run in args.prediction_run:
        path = run/'predictions-private.json.gz'
        if path.exists():
            with gzip.open(path, 'rt') as f: outputs = json.load(f)
        else:
            outputs = json.loads((run/'predictions-private.json').read_text())
        predictions[run.name] = outputs
        scores[run.name] = {name: {group: scoring.measure(members, pred)
                                 for group, members in parts.items()} for name, pred in outputs.items()}
    for group in parts:
        scoring.require_comparable(*[m[group] for run in scores.values() for m in run.values()])
    summary = {'scoring_contract': scoring.contract(), 'scope': scoring.report_scope(rows),
               'cohort_sizes': scoring.cohort_sizes(parts), 'scores': scores,
               'audit': {name: scoring.audit_summary(part) for name, part in parts.items()},
               'prediction_changed': False, 'trained': False, 'selected_new_model': False}
    def write(name, value):
        (out/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    write('summary.json', summary)
    # Save hit ledgers once for every model, so totals and miss diagnosis share membership.
    ledger = {run: {name: {str(r['position']): scoring.hits(r, scoring.prediction_at(pred, r['position']))
                           for r in rows if scoring.eligible(r)} for name, pred in outputs.items()}
              for run, outputs in predictions.items()}
    with gzip.open(out/'hits-private.json.gz', 'wt') as f: json.dump(ledger, f)
    lines = ['# 统一评分与样本合法性复核', '', summary['scope'], '',
             '复用历史预测，没有训练、重新选模型或修改原 run。以下所有模型使用同一合法样本和同一评分代码。', '',
             f'实际平均损失按第一项输出计算：精确 0，正确首页回退 {scoring.FALLBACK_LOSS:g}，错误具体页面／错误服务／空输出 1。它不是模型预测的后验期望损失。', '',
             f'综合排名分为 Top1 至 Top{scoring.RANK_CUTOFF} 最佳匹配收益的平均，再乘 100。匹配收益为精确 1、正确首页回退 {scoring.FALLBACK_GAIN:g}、错误 0；每个前缀只取最佳收益。', '',
             '| 来源／模型 | 接受 Top1 | 接受 Top2 | 接受 Top4 | 精确 Top1 | 精确 Top2 | 精确 Top4 | 实际平均损失↓ | 综合排名分↑ |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for run, models in scores.items():
        for name, groups in models.items():
            m = groups['evidence_total']
            cells = [f"{level[f'top{k}_hits']}/{level['n']}" for level in (m, m['exact']) for k in (1, 2, 4)]
            cells.append(f"{m['decision_loss']['total']:g}/{m['n']} = {m['decision_loss']['mean']:.4f}" if m['n'] else '—')
            cells.append(f"{m['rank_score']['total']:g}/{m['n']} × 100 = {100*m['rank_score']['mean']:.2f}" if m['n'] else '—')
            lines.append('| '+run+' / '+name+' | '+' | '.join(cells)+' |')
    (out/'report-private.md').write_text('\n'.join(lines)+'\n')
    for source in sources:
        if verify(source) != hashes[str(source)]: raise ValueError('Source changed during scoring')
    (out/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    write('run.json', {'complete': True, 'created_at': datetime.now(timezone.utc).isoformat(),
                      'dataset': str(args.dataset.resolve()), 'sources': hashes,
                      'scoring_contract': scoring.contract(), 'raw_modified': False, 'runtime_changed': False})
    write('artifact-hashes.json', {p.name: sha(p) for p in out.iterdir() if p.is_file()})
    assert all(p.stat().st_mode & 0o777 == 0o600 for p in out.iterdir() if p.is_file())
    print(json.dumps({'complete': True, 'output': str(out), 'scope': summary['scope']}, ensure_ascii=False))


if __name__ == '__main__': main()
