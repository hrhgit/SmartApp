"""Complete the SAME fresh validation with historical fusion and two-head families.

The base run contains newly refitted models, never historical forecasts. All
combined predictions are re-scored together, with checked sample/code hashes.
"""
import argparse
from copy import deepcopy
import gzip
import itertools
import os
from pathlib import Path
import pickle

from compare_fresh_models import Experiment, ROOT, log, save_object
from review_historical_models import write, sha, base_query, build_model, navigation_label
from compare_semantic_trajectory import training_rows, MIXES
import evaluation_contract as scoring
import compare_multibackground as comparison
from compare_temporal_fusion import LEARNING_WEIGHTS, fuse_scores
from hierarchical_action_model import HierarchicalActionModel
import multibackground_model as multi
from semantic_online import blend
import semantic_online_classifier as classifiers


def complete(exp, base):
    meta = exp.sources.read(base/'run.json')
    if not meta['complete'] or Path(meta['dataset']) != exp.dataset or meta['scoring_contract'] != scoring.contract():
        raise ValueError('Completion requires same new dataset and scorer')
    for name in exp.sources.read(base/'artifact-hashes.json'):
        exp.sources.check(base/name)
    exp.outputs = {name:{int(p):v for p,v in values.items()} for name,values in
                   exp.sources.read(base/'predictions-private.json.gz').items()}
    if any(set(values) != exp.positions for values in exp.outputs.values()):
        raise ValueError('Incomplete new validation forecasts')
    exp.models = exp.sources.read(base/'models.json')
    for model in exp.models.values():
        if model.get('state'): model['state'] = str(base/model['state'])
        model['same_fresh_validation_source'] = str(base)
    exp.preference_weights = {mode:{int(p):w for p,w in values.items()} for mode,values in
                              exp.sources.read(base/'common-frequency-weights-private.json').items()}
    source = exp.sources.read(ROOT/'hierarchical-action-v1-20260923-supplement-01/selection.json')
    primary = source['best_current_only']['config']['parameters']['primary']
    while primary['family'] == 'same_origin_evidence_gate': primary = primary['parameters']['base']
    hierarchy_jobs = [(family,weight) for family in ('frequency','softmax') for weight in (0.,.1,.25,.5)]
    classifier_jobs = ([{'family':'softmax','lr':lr,'l2':0.,'semantic':semantic}
                        for lr,semantic in itertools.product((.1,.3),(False,True))]
        + [{'family':'multinomial_nb','alpha':alpha,'semantic':semantic}
           for alpha,semantic in itertools.product((.1,1.),(False,True))])
    config = next(c for c in exp.multi_configs if c['id']=='interaction_softmax_combined_rich_lr0.1')
    exp.plan.update(same_new_validation_base=str(base), all_combined_scores_recomputed=True,
        supplement={'fusion_weights':LEARNING_WEIGHTS,'fusion_learning_config':config,
            'hierarchical_primary':primary,'hierarchy_jobs':hierarchy_jobs,
            'classifier_jobs':classifier_jobs,'multistep_mix':[1.,.5,.25]},
        historical_results_used_for_new_model_selection=False)
    write(exp.out/'plan.json',exp.plan)

    # Time-statistics and learning fusion: new causal branch scores, not final
    # weights used retrospectively. Both endpoints must equal this new base run.
    with gzip.open(exp.sources.check(base/'causal-examples-private.pkl.gz'),'rb') as handle:
        examples=pickle.load(handle)['examples']
    statistics={pos:dict(zip(examples[pos]['labels'], map(float,examples[pos]['prior']))) for pos in exp.positions}
    del examples
    learned={}; cursor=0; model=multi.JointNavigationModel(config)
    for row in exp.rows:
        pos=row['position']; q=exp.inputs[pos]
        cursor=comparison.advance(model,exp.stream,cursor,exp.inputs,q['index'])
        learned[pos]=model.scores(q)
    for weight, name in ((0.,'statistics/time'),(1.,'background/'+config['id']+'_pure')):
        for pos in exp.positions:
            got=comparison.bundle(fuse_scores(statistics[pos],learned[pos],weight),exp.homes[pos])
            if got['predictions']!=exp.outputs[name][pos]['predictions']:
                raise ValueError('Fresh fusion endpoint parity failed')
    state=save_object(exp.out,'fusion-learning',model,lambda m:m.scores(exp.inputs[-1]))
    for weight in LEARNING_WEIGHTS:
        outputs={pos:comparison.bundle(fuse_scores(statistics[pos],learned[pos],weight),exp.homes[pos]) for pos in exp.positions}
        exp.add(f'fusion/learning_{weight:.1f}',outputs,'time_statistical_learning_fusion',
                {'learning_weight':weight,'statistics_weight':1-weight},state)
    log(phase='fresh_fusion_complete',models=len(LEARNING_WEIGHTS))

    stream=[r for r in training_rows(exp.data,'hard_kind')
            if navigation_label(r['label'], exp.data['observations'][r['position']])]
    for number,config in enumerate(classifier_jobs):
        model=classifiers.build(config); cursor=0; outputs={}; future={}
        for row in exp.rows:
            pos=row['position']; q=exp.data['observations'][pos]
            while cursor<len(stream) and stream[cursor]['ready_index']<q['index']:
                item=stream[cursor]
                model.learn(exp.data['observations'][item['position']],item['label'],weight=item['weight'],step=item['head'])
                cursor+=1
            def bundle(values):
                return comparison.bundle({k:v for k,v in values.items() if navigation_label(k,q)},exp.homes[pos])
            outputs[pos]=bundle(model.predict(q,0))
            future[pos]=bundle(blend([model.predict(q,h) for h in (1,2,3)],[1.,.5,.25]))
        state=save_object(exp.out,'semantic-classifier-'+str(number),model,lambda m:m.predict(q,0))
        exp.add(f'semantic_classifier/{number:02d}_single',outputs,'semantic_classifier',config,state)
        exp.add(f'semantic_classifier/{number:02d}_multistep',future,'semantic_multistep_classifier',
                {**config,'future_weights':[1.,.5,.25]},state)
    log(phase='fresh_semantic_classifiers_complete',models=2*len(classifier_jobs))

    for family, weight in hierarchy_jobs:
        head=({'id':'stat_frequency_conditional','family':'frequency','parameters':{'scope':'conditional'}}
              if family=='frequency' else {'id':'online-softmax-0.0-1.0','family':'softmax',
                                          'parameters':{'learning_rate':1.,'l2':0.}})
        model=HierarchicalActionModel(build_model(primary),build_model(head),weight)
        cursor=0; vocabulary=set(); recent={}; outputs={}
        for row in exp.rows:
            pos=row['position']; q=exp.data['observations'][pos]
            while cursor<len(exp.stream) and exp.stream[cursor]['ready_index']<q['index']:
                item=exp.stream[cursor]; target=item['record']['target']
                record=base_query(exp.data['observations'][item['position']])
                owner=target.get('_bundle') or target.get('bundleID')
                if target.get('_bundle') and target.get('bundleID') and target['_bundle']!=target['bundleID']: owner=None
                record.update(label=item['label'],outcome_id=item['outcome_id'],outcome_timestamp=item['outcome_timestamp'],
                              destination_app=owner)
                model.learn(record)
                vocabulary.add(item['label']); recent[item['label']]=item['outcome_timestamp']
                if owner:
                    vocabulary.add('app:'+owner); recent['app:'+owner]=item['outcome_timestamp']
                cursor+=1
            query=base_query(q); query.update(labels=sorted(vocabulary),candidate_last_seen=dict(recent))
            scores=model.scores(query)
            ordered=[k for k in sorted(scores,key=lambda k:(-scores[k],k)) if navigation_label(k,q)]
            outputs[pos]={'predictions':ordered[:4],'rankable':ordered}
        state=save_object(exp.out,f'hierarchical-{family}-{weight}',model,lambda m:m.scores(query))
        exp.add(f'hierarchical/{family}_{weight}',outputs,'hierarchical_action_heads',
                {'primary':primary,'application':head,'app_weight':weight,'same_origin_gate_removed':True},state)
    log(phase='fresh_hierarchical_complete',models=len(hierarchy_jobs))
    if exp.meta.get('evaluation_scope')=='complete_two_cohorts':
        # Re-run the adapted legacy methods with the absent ordered-history
        # channel explicitly zero, not a perfect empty-dictionary Jaccard match.
        replaced=[name for name in exp.outputs if name.startswith('early/')]
        for name in replaced:
            del exp.outputs[name];del exp.models[name]
        exp.plan['early_current_only_similarity_audit']='removed history channel contributes zero, not an empty-set match'
        exp.plan['replayed_early_configurations']=replaced
        write(exp.out/'plan.json',exp.plan)
        exp.early_context()
    write(exp.out/'predictions-private.json.gz',exp.outputs)
    write(exp.out/'models.json',exp.models)
    write(exp.out/'common-frequency-weights-private.json',exp.preference_weights)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();base=args.base.resolve();out=args.output.resolve()
    if out.exists() or out==ROOT.resolve() or not out.is_relative_to(ROOT.resolve()): parser.error('Use a new private run')
    import json
    dataset=Path(json.loads((base/'run.json').read_text())['dataset'])
    os.umask(0o077);out.mkdir(mode=0o700)
    exp=Experiment(dataset,out);complete(exp,base);exp.evaluate();exp.finish()


if __name__=='__main__':main()
