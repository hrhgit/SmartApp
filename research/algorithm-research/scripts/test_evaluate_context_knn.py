"""Synthetic regression tests for temporal leakage, missingness and ranking."""
import copy
import tempfile
import unittest
from pathlib import Path

import evaluate_context_knn as m


def target(name, kind='app'):
    t = {'id': name, 'name': name, 'kind': kind, 'bundleID': name,
         'admission': {'basis': 'standalone_application', 'policyVersion': 'target-eligibility-v4'}}
    if kind == 'page':
        t['locator'] = 'https://example.invalid/' + name
    return t


def observation(index=10, truth='B', bg=None, quality=None):
    return m.Observation(index, index+2, index*1000, 'session', str(index), 's'+str(index), 0,
        'A', 'A', {'app:A':1}, bg or {}, {'recent:A':1}, ('A',),
        {k:{**target(k), '_bundle':k} for k in ('B','C','D','E','F')},
        {'C':5, 'B':4}, {}, truth, {**target(truth),'_bundle':truth}, 1000, 'complete', True,
        0,index,quality or {},str(index+1))


class ModelTests(unittest.TestCase):
    def test_delayed_labels_not_visible_at_their_own_prediction(self):
        a,b,c=observation(1),observation(3),observation(4)
        self.assertEqual(m.eligible_neighbors([a,b,c],1),[])
        self.assertEqual(m.eligible_neighbors([a,b,c],2),[0])

    def test_background_weight_changes_prediction_with_same_history(self):
        b=observation(1,'B',{'browser':{'course':1}}, {'browser':1})
        c=observation(2,'C',{'browser':{'code':1}}, {'browser':1})
        q=observation(10,'B',{'browser':{'course':1}}, {'browser':1})
        items=[b,c,q]
        pairs=[[],[],[(j,m.similarity(q,items[j])) for j in (0,1)]]
        history,_=m.predict_knn(items,2,pairs,weight=0,k=1)
        state,_=m.predict_knn(items,2,pairs,weight=.55,k=1)
        self.assertEqual(history[0],'C')
        self.assertEqual(state[0],'B')

    def test_missing_background_is_not_confirmed_empty(self):
        missing=observation()
        empty=observation(bg={'browser':{}},quality={'browser':1})
        self.assertEqual(m.similarity(missing,empty)[-1],0)
        self.assertEqual(m.similarity(empty,empty)[-1],1)
        self.assertEqual(m.similarity(empty,empty)[-2],1)

    def test_legacy_background_is_downweighted(self):
        known=observation(bg={'browser':{'B':1}},quality={'browser':1})
        old=observation(bg={'browser':{'B':1}},quality={'browser':.5})
        self.assertEqual(m.similarity(old,known)[-1],.5)

    def test_future_reference_fails(self):
        a={'id':'a','sessionID':'s','sequence':1}
        f={'id':'f','sessionID':'s','sequence':2}
        with self.assertRaises(ValueError):m.assert_asof(f,a,{'a':0,'f':1})

    def test_transient_windows_not_cross_session_identities(self):
        t=target('w','window')
        self.assertNotEqual(m.key_for(t,'s1'),m.key_for(t,'s2'))
        self.assertNotIn('target:w',m.target_tokens(t))

    def test_parent_child_do_not_consume_two_slots(self):
        parent={**target('P'), '_bundle':'bundle'}
        child={**target('C','page'), '_bundle':'bundle'}
        self.assertEqual(m.select_four(['C','P'],{'C':child,'P':parent}),['C'])

    def test_impossible_candidates_still_count_as_misses(self):
        q=observation(truth='unseen')
        row=m.score_row(q,['B'])
        self.assertFalse(row['covered'])
        self.assertFalse(row['hit4'])
        self.assertEqual(m.metric([row])['top4'],0)
        self.assertIsNone(m.metric([row])['top4_given_covered'])

    def test_recency_fallback_only_known_visited_candidates(self):
        q=observation()
        raw,shown=m.rank_votes({},q)
        self.assertEqual(raw,['C','B'])
        self.assertEqual(shown,['C','B'])


class FeatureTests(unittest.TestCase):
    def fixture(self):
        events=[]
        def add(kind,t,data,app=None):
            i=len(events)
            e={'id':str(i),'sequence':i+1,'sessionID':'s','timestampMs':t,
               'monotonicMs':t,'kind':kind,'source':'running_apps','data':data}
            if app:e['app']={'bundleID':app}
            events.append(e)
            return e
        add('session.start',0,{})
        add('target.update',1,{'target':target('A')})
        inv=add('inventory.change',2,{'entrances':[{'id':'entry','targetID':'A','bundleID':'A'}]})
        anchor=add('context.change',3,{'targetID':'A','reason':'app_activation'},'A')
        add('target.update',4,{'target':target('NEW','page')})
        outcome=add('context.change',5,{'targetID':'NEW','reason':'navigation'},'browser')
        add('heartbeat',30003,{})
        sample={'schemaVersion':3,'id':'sample','sessionID':'s','anchorEventID':anchor['id'],
                'inputThroughSequence':anchor['sequence'],'status':'observed_horizon',
                'origin':'natural_observed','outcomeCoverage':'initial_coverage_unverified',
                'recentContextEventIDs':[anchor['id']],'inventoryEventIDs':[inv['id']],
                'inputCoverage':{'status':'unknown','sources':[
                    {'source':'running_apps','status':'legacy_observed'}],'missingSources':[]},
                'outcomes':[{'eventID':outcome['id'],'targetID':'NEW','offsetMs':2,
                    'hierarchy':{'appTargetID':'app:browser'}}]}
        return events,sample

    def test_future_target_label_is_not_current_candidate_or_feature(self):
        events,s=self.fixture()
        rows,audit,*_=m.make_observations(events,[s],set())
        q=rows[0]
        self.assertEqual(q.truth,'NEW')
        self.assertNotIn('NEW',q.candidates)
        self.assertNotIn('target:NEW',q.foreground)
        self.assertFalse(q.truth_meta['_known_at_cutoff'])
        self.assertEqual(q.ready_index,6)
        self.assertEqual(audit['max_input_reference_violations'],0)

    def test_future_refinement_not_an_independent_arrival(self):
        events,s=self.fixture()
        events[5]['data']['reason']='resolution_update'
        with self.assertRaises(ValueError):m.make_observations(events,[s],set())

    def test_explicit_source_failure_excludes_sample(self):
        events,s=self.fixture()
        s['outcomeCoverage']='source_unavailable'
        rows,audit,*_=m.make_observations(events,[s],set())
        self.assertEqual(rows,[])
        self.assertEqual(audit['exclusions']['explicit_partial_or_unreliable_outcome_coverage'],1)

    def test_validation_session_excluded_from_training(self):
        events,s=self.fixture()
        rows,audit,arrivals,*_=m.make_observations(events,[s],{'s'})
        self.assertEqual(rows,[])
        self.assertEqual(sum(sum(c.values()) for c in arrivals.values()),0)

    def test_empty_legacy_inventory_not_observed_empty(self):
        events,s=self.fixture()
        events[2]['data']['entrances']=[]
        s['inputCoverage']['sources'][0]['status']='unknown'
        rows,*_=m.make_observations(events,[s],set())
        self.assertEqual(rows[0].background,{})


if __name__=='__main__':
    unittest.main()
