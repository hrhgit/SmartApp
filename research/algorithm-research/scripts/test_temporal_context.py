"""Synthetic leakage, continuity, counting and model-parity regression checks."""
from copy import deepcopy
import unittest

from compare_temporal_context import choose, configurations, model_inputs, replay
from navigation_hierarchy import NavigationHierarchy
from semantic_annotations import annotate_target
from temporal_context import TemporalHistory, TemporalNavigationModel, build_contexts, ordered_similarity


def target(key, kind='app', owner=None):
    owner = owner or (key.removeprefix('app:') if kind == 'app' else 'browser')
    t = {'id': key, 'kind': kind, 'bundleID': owner,
         'admission': {'basis':'fixture', 'policyVersion':'target-eligibility-v4'}}
    if kind in {'page','directory','file'}:
        t['locator'] = ('https://example.test/' + key if kind == 'page' else '/example/' + key)
    return t


def event(n, seconds, kind, data=None, source='workspace', session='S'):
    return {'id':str(n), 'sessionID':session, 'timestampMs':seconds*1000,
            'monotonicMs':seconds*1000, 'kind':kind, 'source':source, 'data':data or {}}


def context(n, seconds, key, **data):
    return event(n,seconds,'context.change',{'targetID':key,'reason':'foreground_change',**data})


def fixture_events():
    return [event(0,0,'target.update',{'target':target('app:A')}),
            event(1,0,'target.update',{'target':target('app:B')}),
            context(2,0,'app:A'), context(3,10,'app:B')]


def history(events):
    h = TemporalHistory()
    for i,e in enumerate(events):
        h.consume(e,i)
    return h


def query(n, key='app:A'):
    t = target(key)
    return {'position':n,'index':n,'timestamp':n*1000,'origin':key,
            'target':t,'annotation':annotate_target(t),
            'temporal': {'through_index':n,'sequence':['app:B',key],
                'windows':{str(w):{'count':{'target:app:B':1},'decayed_count':{'target:app:B':.5},
                                  'recent':{'target:app:B':.5},'dwell':{'target:app:B':30}}
                           for w in (5,15,60)}}}


def record(n, q, label):
    return {'origin':q['origin'],'annotation':q['annotation'],'label':label,
            'outcomeID':str(n),'outcomeTimestamp':q['timestamp']+1000,'weight':1.,
            'destination':annotate_target(target(label))['identity'],'source':'synthetic','observedStep':1}


class HistoryTests(unittest.TestCase):
    def test_initial_and_heartbeat_are_not_visits(self):
        es=fixture_events()[:3];h=history(es)
        self.assertEqual(h.snapshot(es[-1])['windows']['5']['confirmed_arrivals'],0)
        for i in range(3,10):
            h.consume(event(i,i,'heartbeat'),i)
        e=context(10,10,'app:B');h.consume(e,10)
        s=h.snapshot(e)
        self.assertEqual(s['windows']['5']['confirmed_arrivals'],1)
        self.assertEqual(s['windows']['5']['dwell']['target:app:A'],10)

    def test_future_prefix_invariance_and_refinement_not_backfilled(self):
        es=fixture_events()
        es += [event(4,11,'target.update',{'target':target('page:P',kind='page',owner='B')}),
               context(5,12,'page:P',reason='resolution_update',resolvesContextEventID='3')]
        h=history(es[:4]);old=deepcopy(h.snapshot(es[3]))
        for i in (4,5):h.consume(es[i],i)
        now=h.snapshot(es[5])
        self.assertEqual(old,history(es[:4]).snapshot(es[3]))
        self.assertIn('target:app:B',old['windows']['5']['count'])
        self.assertNotIn('target:page:P',old['windows']['5']['count'])
        self.assertEqual(now['windows']['5']['confirmed_arrivals'],1)
        self.assertEqual(now['windows']['5']['dwell']['target:app:B'],2)
        self.assertNotIn('target:page:P',now['windows']['5']['dwell'])
        self.assertEqual(now['sequence'],['app:A','page:P'])

    def test_gap_and_initial_recovery_do_not_add_arrival_or_unknown_dwell(self):
        es=fixture_events()+[event(4,20,'capture.gap'),context(5,100,'app:B'),context(6,110,'app:A')]
        h=history(es);s=h.snapshot(es[-1])
        self.assertEqual(s['windows']['5']['confirmed_arrivals'],2)
        self.assertEqual(s['windows']['5']['dwell']['target:app:B'],10)
        self.assertEqual(s['sequence'],['app:B','app:A'])

    def test_idle_duration_is_excluded(self):
        es=fixture_events()+[event(4,20,'idle.change',{'idle':True}),
            event(5,90,'idle.change',{'idle':False}),context(6,100,'app:A')]
        s=history(es).snapshot(es[-1])
        self.assertEqual(s['windows']['5']['dwell']['target:app:B'],20)

    def test_window_expiry_and_partial_overlap(self):
        es=fixture_events()+[context(4,610,'app:A')]
        s=history(es).snapshot(es[-1])
        self.assertEqual(s['windows']['5']['confirmed_arrivals'],1)
        self.assertEqual(s['windows']['15']['confirmed_arrivals'],2)
        self.assertEqual(s['windows']['5']['dwell']['target:app:B'],300)

    def test_source_failure_censors_but_background_failure_does_not(self):
        es=fixture_events()+[event(4,20,'source.status',{'observationStatus':'ready'}),
                            event(5,200,'source.status',{'observationStatus':'unavailable'}),
                            context(6,210,'app:A')]
        s=history(es).snapshot(es[-1])
        self.assertEqual(s['windows']['5']['dwell']['target:app:B'],10)
        self.assertEqual(s['windows']['5']['confirmed_arrivals'],1)
        es[5]['source']='unrelated-background'
        s=history(es).snapshot(es[-1])
        self.assertEqual(s['windows']['5']['dwell']['target:app:B'],200)

    def test_inventory_and_background_metadata_do_not_change_inputs(self):
        es=fixture_events()
        baseline=history(es).snapshot(es[-1])
        h=history(es)
        h.consume(event(4,10,'target.update',{'target':target('app:HIDDEN')}),4)
        h.consume(event(5,10,'inventory.change',{'entrances':[{'targetID':'app:HIDDEN'}]}),5)
        s=h.snapshot(es[-1]);s['through_index']=baseline['through_index']
        self.assertEqual(s,baseline)

    def test_cross_session_and_excluded_sessions(self):
        es=fixture_events()+[event(4,100,'target.update',{'target':target('app:C')},session='T'),
                            event(5,100,'context.change',{'targetID':'app:C'},session='T')]
        h=history(es);s=h.snapshot(es[-1])
        self.assertEqual(s['sequence'],['app:C'])
        self.assertEqual(s['windows']['5']['confirmed_arrivals'],1)
        h=TemporalHistory(['S'])
        for i,e in enumerate(es):h.consume(e,i)
        self.assertEqual(h.snapshot(es[-1])['windows']['5']['confirmed_arrivals'],0)

    def test_window_instance_ids_do_not_enter_vectors(self):
        es=fixture_events()+[event(4,20,'target.update',{'target':target('ephemeral:123','window','B')}),
                            context(5,20,'ephemeral:123')]
        s=history(es).snapshot(es[-1])
        self.assertIn('window_class:B',s['sequence'])
        self.assertNotIn('ephemeral:123',str(s))

    def test_sequence_preserves_order_and_length(self):
        self.assertEqual(ordered_similarity(['A','B','C'],['B','A','C']),0)
        self.assertIsNone(ordered_similarity(['C'],['A','B','C']))
        self.assertEqual(ordered_similarity(['A','B','C'],['A','B','C']),1)

    def test_full_stream_cannot_change_earlier_frozen_context(self):
        es=fixture_events()
        q={'position':0,'index':3,'anchor_id':'3','session':'S'}
        old=build_contexts(es,[q])[0]
        es += [event(4,15,'target.update',{'target':target('app:NEW')}),context(5,20,'app:NEW')]
        self.assertEqual(build_contexts(es,[q])[0],old)

    def test_explicit_refinement_chain_keeps_single_episode(self):
        es=fixture_events()+[
            event(4,11,'target.update',{'target':target('window:one','window','B')}),
            context(5,12,'window:one',reason='resolution_update',resolvesContextEventID='3'),
            event(6,13,'target.update',{'target':target('page:P','page','B')}),
            context(7,14,'page:P',reason='resolution_update',resolvesContextEventID='5')]
        h=history(es);s=h.snapshot(es[-1])
        self.assertEqual(s['windows']['5']['confirmed_arrivals'],1)
        self.assertEqual(s['sequence'],['app:A','page:P'])
        self.assertEqual(h.audit['refinements_without_extra_count'],2)


class ModelTests(unittest.TestCase):
    def test_baseline_native_hierarchy_parity(self):
        model=TemporalNavigationModel(configurations()[0]);old=NavigationHierarchy()
        for i,(start,label) in enumerate([('app:A','app:B'),('app:A','app:C'),('app:A','app:B'),('app:D','app:E')]):
            q=query(i,start);r=record(i,q,label);model.learn(q,r);old.learn(r)
        q=query(20)
        self.assertEqual(model.ranked(q),old.ranked(q['target']))
        for k,v in model.scores(q).items():self.assertAlmostEqual(v,old.scores(q['target'])[k])

    def test_context_reweights_but_does_not_restrict_learned_outputs(self):
        config={'mode':'rich','windows':[5],'sequence_weight':.25,'floor':.1}
        model=TemporalNavigationModel(config)
        a,b=query(0),query(1)
        for profile in b['temporal']['windows'].values():
            for k in profile:profile[k]={'target:unrelated':1.}
        b['temporal']['sequence']=['unrelated','app:A']
        model.learn(a,record(0,a,'app:X'));model.learn(b,record(1,b,'app:Y'))
        q=query(20);scores=model.scores(q)
        self.assertEqual(set(scores),{'app:X','app:Y'})
        self.assertGreater(scores['app:X'],scores['app:Y'])
        dirty=deepcopy(q);dirty.update(background_state={'leak':1},truth_action='app:Z',
                                     candidateTargetIDs=['app:Z'],evidence_grade='original')
        self.assertEqual(scores,model.scores(dirty))

    def test_input_whitelist(self):
        q=query(0);q['background_state']={'bad':'secret'};q['future']=['leak']
        x=model_inputs({'observations':[q]},{0:q['temporal']})[0]
        self.assertNotIn('background_state',x);self.assertNotIn('future',x)

    def test_late_scores_cannot_select_config(self):
        def result(name,hits,test_hits):
            score={g:{'top4_hits':hits,'top1_hits':0} for g in ['original','expanded','all']}
            return {'config':{'id':name},'validation':score,'test':{'hits':test_hits}}
        self.assertEqual(choose([result('early',3,0),result('late',2,100)])['config']['id'],'early')

    def test_same_event_maturity_not_available_to_prediction(self):
        obs=[query(0),query(1),query(2)]
        row={'position':0,'ready_index':1,'label':'app:B','weight':1.,'outcome_id':'out',
             'outcome_timestamp':1000,'record':{'target':target('app:B'),'observed_step':1,'target_evidence_index':1}}
        pred,_,_,_=replay(obs,[row],[{'position':1},{'position':2}],configurations()[0],{})
        self.assertEqual(pred[1],[]);self.assertEqual(pred[2],['app:B'])

    def test_restore_rejects_future_maturity(self):
        from export_temporal_models import restore
        from temporal_context import VERSION
        q=query(1)
        payload={'version':VERSION,'background_features':False,'config':configurations()[0],
                 'asof_event_index':2,'records':[{'input':q,'ready_index':3,'record':record(1,q,'app:B')}]}
        with self.assertRaisesRegex(ValueError,'unavailable'):
            restore(payload)


if __name__ == '__main__':
    unittest.main()
