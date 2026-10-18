"""Union the entire fixed benchmark and its new tail, never training anchors."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil

import evaluation_contract as scoring
import evaluate_context_knn as core
from review_historical_models import ROOT, Sources, sha, write


def combine(old, fresh):
    observations = fresh['observations']
    lookup = {q['sample_id']:q for q in observations}
    references = {r['position']:r for r in fresh['reference_stream']}
    evaluation=[]; seen=set()
    for source, segment in ((old,'fixed'),(fresh,'tail')):
        for value in source['evaluation']:
            if value['sample_id'] in seen: raise ValueError('Duplicate validation member')
            seen.add(value['sample_id']); q=lookup[value['sample_id']]
            oldq=source['observations'][value['position']]
            if any(q[k]!=oldq[k] for k in ('sample_id','anchor_id','index','origin','timestamp','target','annotation')):
                raise ValueError('Original validation input changed')
            if references[q['position']]['record']['label']!=value['truth_action']:
                raise ValueError('Original scoring label changed')
            row=deepcopy(value);row.pop('_scoring',None)
            row.update(position=q['position'],input_index=q['index'],label_ready_index=q['ready_index'],
                       complete_validation_segment=segment)
            evaluation.append(row)
    result={**fresh,'evaluation':sorted(evaluation,key=lambda r:r['position'])}
    if len(evaluation)!=len(old['evaluation'])+len(fresh['evaluation']):raise ValueError('Member count mismatch')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixed',type=Path,required=True)
    parser.add_argument('--fresh',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve();sources=Sources()
    if out.exists() or out==ROOT.resolve() or not out.is_relative_to(ROOT.resolve()):parser.error('Use a new private run')
    old=sources.read(args.fixed/'dataset-private.json.gz');fresh=sources.read(args.fresh/'dataset-private.json.gz')
    meta=sources.read(args.fresh/'run.json');data=combine(old,fresh)
    os.umask(0o077);out.mkdir(mode=0o700)
    rows=scoring.prepare_evaluation(data,out)
    parts=scoring.cohorts(rows);sizes=scoring.cohort_sizes({k:parts[k] for k in ('all','evidence_total')})
    boundary=min(data['observations'][r['position']]['index'] for r in rows)
    plan={'evaluation_scope':'complete_two_cohorts','members':'all fixed validation members plus complete appended tail',
          'fixed_candidate_n':len(old['evaluation']),'tail_candidate_n':len(fresh['evaluation']),
          'candidate_n':len(rows),'sizes':sizes,'boundary_event_index':boundary,
          'report_cohorts':['all','evidence_total'],'primary_metric':'rank_score.mean times 100',
          'old_members_and_labels_preserved':True,'training_anchors_not_added_to_evaluation':True,
          'historical_development_set_not_new_holdout':True,'scoring_contract':scoring.contract()}
    write(out/'plan.json',plan);write(out/'dataset-private.json.gz',data)
    for name in ('observation-objects-private.json.gz','label-ledger-private.json.gz'):
        shutil.copyfile(sources.check(args.fresh/name),out/name)
    summary={**plan,'observations':len(data['observations']),
        'scope':scoring.report_scope(rows),'validation_range_utc':[
            core.utc(data['observations'][r['position']]['timestamp']) for r in (rows[0],rows[-1])]}
    write(out/'summary.json',summary)
    sources.verify_unchanged()
    write(out/'run.json',{**meta,'created_at':datetime.now(timezone.utc).isoformat(),
        'evaluation_scope':'complete_two_cohorts','boundary_event_index':boundary,
        'fixed_dataset':str(args.fixed.resolve()),'tail_dataset':str(args.fresh.resolve()),
        'source_files':sources.accessed,'historical_development_set_not_new_holdout':True})
    for p in Path(__file__).parent.glob('*.py'):(out/('source-'+p.name)).write_bytes(p.read_bytes())
    for p in out.iterdir():
        if p.is_file():p.chmod(0o600)
    write(out/'artifact-hashes.json',{p.name:sha(p) for p in out.iterdir() if p.is_file()})
    print(plan,flush=True)


if __name__=='__main__':main()
