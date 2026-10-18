"""Append newly observed mature behavior while preserving the frozen benchmark."""
import argparse
from collections import Counter
from copy import deepcopy
import gzip
import json
import os
from pathlib import Path

import describe_dwell as dwell
import evaluate_context_knn as core
from evaluate_dwell_cleaning import derive_label
from prepare_semantic_trajectory import inputs_at_anchors
from semantic_annotations import annotate_target,compare_annotations
from summarize_tuned_history import verify_run
from trajectory_supervision import TrajectoryEvents,derive_trajectory,action_key


VERSION='multibackground-appended-dataset-v1'
SOURCES=('known_inactivity.py','prepare_multibackground_dataset.py','describe_dwell.py','evaluate_context_knn.py',
         'evaluate_dwell_cleaning.py','prepare_semantic_trajectory.py','trajectory_supervision.py',
         'semantic_annotations.py','reopenable_actions.py','evaluate_ordered_current.py')


def checked_snapshot(path):
    required={'events.jsonl','target-history.jsonl','navigation-samples.jsonl','manifest.json'}
    hashes={}
    for line in (path/'SHA256SUMS').read_text().splitlines():
        digest,name=line.split(maxsplit=1);name=name.lstrip('*')
        if name not in required or core.sha(path/name)!=digest:raise ValueError('Snapshot checksum failed')
        hashes[name]=digest
    if set(hashes)!=required:raise ValueError('Incomplete snapshot checksum manifest')
    manifest=json.loads((path/'manifest.json').read_text())
    if manifest['sampleSchemaVersion']!=3:raise ValueError('Requires v3 samples')
    return hashes,manifest


def require_prefix(old,new):
    if len(new)<len(old) or new[:len(old)]!=old:
        raise ValueError('New raw snapshot is not an exact extension of the frozen snapshot')


def evidence_for_new(rows,observations,reference):
    history=[];cursor=0
    reference=sorted(reference,key=lambda r:(r['ready_index'],r['position']))
    for row in rows:
        q=observations[row['position']]
        while cursor<len(reference) and reference[cursor]['ready_index']<q['index']:
            history.append(reference[cursor]);cursor+=1
        witnesses={}
        priority={'weak':1,'strong':2,'exact':3}
        for previous in history:
            if previous['record']['label']!=row['truth_action']:continue
            p=observations[previous['position']]
            relation=({'level':'exact','score':1.} if p['origin']==q['origin']
                      else compare_annotations(p['annotation'],q['annotation']))
            if relation['level'] in {'exact','strong','weak'}:
                key=previous['record']['outcome_id']
                if key not in witnesses or priority[relation['level']]>priority[witnesses[key]['level']]:
                    witnesses[key]=relation
        levels={r['level'] for r in witnesses.values()}
        row['evidence_grade']=('added_exact' if 'exact' in levels else 'semantic_strong' if 'strong' in levels
                               else 'semantic_weak' if 'weak' in levels else 'unsupported')
        row['evidence_count']=len(witnesses)
        row['new_exact_history']=bool('exact' in levels)
        row['evidence_basis']='diagnostic_after_truth; only pre-cutoff mature reference labels'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-dataset',type=Path,required=True)
    parser.add_argument('--snapshot',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.expanduser().resolve()
    if not out.is_relative_to(core.PRIVATE_ROOT.resolve()) or out==core.PRIVATE_ROOT.resolve() or out.exists():
        parser.error('Use a new private training directory')
    base_provenance,base_hashes=verify_run(args.base_dataset)
    hashes,manifest=checked_snapshot(args.snapshot)
    old_events=core.load_lines(Path(base_provenance['snapshot'])/'events.jsonl')
    if core.sha(Path(base_provenance['snapshot'])/'events.jsonl')!=base_provenance['migration']['current']['events.jsonl']:
        raise ValueError('Frozen original raw events changed')
    events=core.load_lines(args.snapshot/'events.jsonl');require_prefix(old_events,events)
    samples=core.load_lines(args.snapshot/'navigation-samples.jsonl')
    if len(events)!=manifest['events'] or len(samples)!=manifest['samples']:raise ValueError('Export counts differ')
    # The sole legacy menu-display record is rejected by current native export.
    # Do not accept unexplained admission rejection counts.
    rejected=[i for i,e in enumerate(events) if e['kind']=='recommendation.show']
    if manifest['eventsExcludedFromSamples']!=len(rejected):
        raise ValueError('Unexplained native admission exclusions require an explicit audit')
    source_bytes={n:Path(__file__).with_name(n).read_bytes() for n in SOURCES}
    with gzip.open(args.base_dataset/'dataset-private.json.gz','rt') as f:old=json.load(f)
    items,extraction,*_=core.make_observations(events,samples,core.KNOWN_TEST_SESSIONS)
    observed=inputs_at_anchors(core,items,events,annotate_target)
    old_n=len(old['observations']);new_items=[(i,q) for i,q in enumerate(items) if q.index>=len(old_events)]
    by_anchor={q['anchor_id']:q for q in observed}
    for q in old['observations']:
        check=by_anchor.get(q['anchor_id'])
        if not check or any(check[k]!=q[k] for k in ('position','index','ready_index','origin','timestamp','target','annotation')):
            raise ValueError('Frozen original observation timing or identity changed')
    observations=deepcopy(old['observations'])
    for i,q in new_items:
        if observed[i]['position']!=len(observations):raise ValueError('Unexpected new observation insertion')
        observations.append(observed[i])
    event_index={e['id']:i for i,e in enumerate(events)}
    lookup=TrajectoryEvents({e['id']:e for e in events},event_index)
    visits,_,visit_audit=dwell.extract_visits(events,samples,core.KNOWN_TEST_SESSIONS)
    by_visit={v['arrival_event_id']:v for v in visits}
    streams={k:deepcopy(old['streams'][k]) for k in ('hard_kind','resolved')}
    reference=deepcopy(old['reference_stream'])
    new_eval=[];label_audit=Counter();new_ledgers=[]
    for position,q in new_items:
        for policy in streams:
            labels,audit=derive_trajectory(q,by_visit,lookup,event_index,policy=policy)
            new_ledgers.append({'position':position,'policy':policy,**audit})
            for label in labels:
                streams[policy].append({'position':position,'input_index':q.index,
                    'ready_index':q.ready_index,'record':label})
        scored,why=derive_label(q,by_visit,lookup,5)
        label_audit[why['status']]+=1
        if scored is None:continue
        truth=action_key(scored.truth_meta)
        if truth is None:
            label_audit['unusable_scoring_target']+=1;continue
        row={'position':position,'sample_id':q.sample_id,'split':'new_test','cohort':'new_data',
             'truth_action':truth,'truth_metadata':scored.truth_meta,
             'score_policy':'frozen_style_5s_stable_within30s',
             'input_index':q.index,'label_ready_index':q.ready_index,
             'evidence_grade':'unsupported','strict':q.strict}
        new_eval.append(row)
        outcome=lookup[scored.next_event_id]
        reference.append({'position':position,'input_index':q.index,'ready_index':q.ready_index,
            'record':{'origin':observations[position]['origin'],'label':truth,
                      'outcome_id':scored.next_event_id,'outcome_timestamp':outcome['timestampMs']}})
    evidence_for_new(new_eval,observations,reference)
    evaluation=deepcopy(old['evaluation'])+new_eval
    assert evaluation[:len(old['evaluation'])]==old['evaluation']
    for policy in streams:
        streams[policy].sort(key=lambda r:(r['ready_index'],r['position'],r['record']['observed_step']))
    os.umask(0o077);out.mkdir(mode=0o700,parents=True,exist_ok=False)
    payload={'observations':observations,'evaluation':evaluation,'streams':streams,
             'reference_stream':reference,'original_cleaning_stream':old['original_cleaning_stream']}
    with gzip.open(out/'dataset-private.json.gz','wt') as f:json.dump(payload,f,ensure_ascii=False)
    core.write_json(out/'new-label-audit-private.json',new_ledgers)
    summary={'old_observations':old_n,'new_observations':len(new_items),'total_observations':len(observations),
             'old_evaluation_n':len(old['evaluation']),'new_evaluation_n':len(new_eval),
             'new_label_audit':dict(label_audit),'new_evidence':dict(Counter(r['evidence_grade'] for r in new_eval)),
             'raw_events_added':len(events)-len(old_events),'extraction':extraction,
             'legacy_rejected_display_boundaries':len(rejected),'raw_prefix_equal':True,
             'new_observations_strict':sum(q.strict for _,q in new_items),
             'streams':{k:len(v) for k,v in streams.items()},'visit_audit':visit_audit}
    core.write_json(out/'summary.json',summary)
    if checked_snapshot(args.snapshot)[0]!=hashes or verify_run(args.base_dataset)[1]!=base_hashes:
        raise ValueError('Source data changed during append')
    for n,b in source_bytes.items():
        if Path(__file__).with_name(n).read_bytes()!=b:raise ValueError('Source changed during append')
        (out/('source-'+n)).write_bytes(b)
    core.write_json(out/'run.json',{'version':VERSION,'complete':True,'base_dataset':str(args.base_dataset.resolve()),
        'base_hashes':base_hashes,'snapshot':str(args.snapshot.resolve()),'snapshot_hashes':hashes,
        'new_data_start_index':len(old_events),'old_evaluation_n':len(old['evaluation']),
        'old_membership_labels_inputs_unchanged':True,'new_data_is_selection_holdout':True,
        'collection_or_runtime_changed':False,'raw_modified':False})
    core.write_json(out/'artifact-hashes.json',{p.name:core.sha(p) for p in out.iterdir() if p.is_file()})
    print(json.dumps({'output':str(out),**summary},ensure_ascii=False))


if __name__=='__main__':main()
