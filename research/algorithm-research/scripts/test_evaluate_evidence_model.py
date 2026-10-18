import copy
import unittest

import evaluate_context_knn as core
import evaluate_evidence_model as model
from test_evaluate_context_knn import observation,target


class EvidenceTests(unittest.TestCase):
    def fixture(self,truths=('B','B')):
        items=[observation(i+1,truth=x,bg={'browser':{'course':1}},quality={'browser':1}) for i,x in enumerate(truths)]
        q=observation(100,truth='B',bg={'browser':{'course':1}},quality={'browser':1})
        items.append(q)
        ledger=[{'target_event_id':f'outcome{i}'} for i in range(len(truths))]
        times={f'outcome{i}':('session',i*60000) for i in range(len(truths))}
        config={'foreground':'exact','background_min':.75,'history_min':.3,'half_life_hours':6.,'representation':'exact'}
        pairs=model.prepare_pairs(core,items,set(range(len(truths))),[len(truths)])
        return items,ledger,times,pairs,config

    def test_missing_background_never_becomes_identical_scene(self):
        a=observation(1);q=observation(100)
        self.assertEqual(model.prepare_pairs(core,[a,q],{0},[1])[1],[])

    def test_acceptance_is_independent_of_query_future_truth(self):
        items,ledger,times,pairs,cfg=self.fixture()
        before=model.evidence(core,items,ledger,times,2,pairs,cfg)
        self.assertTrue(model.accept(before,2,.85))
        items[2].truth='never_seen';items[2].truth_meta=target('never_seen')
        after=model.evidence(core,items,ledger,times,2,pairs,cfg)
        self.assertEqual(before,after)

    def test_overlapping_labels_are_not_two_repeated_behaviors(self):
        items,ledger,times,pairs,cfg=self.fixture()
        ledger[1]['target_event_id']=ledger[0]['target_event_id']
        e=model.evidence(core,items,ledger,times,2,pairs,cfg)
        self.assertEqual(e['evidence_count'],1)
        self.assertFalse(model.accept(e,2,.85))

    def test_adjacent_distinct_arrivals_are_only_one_support_group(self):
        items,ledger,times,pairs,cfg=self.fixture()
        times['outcome1']=('session',1000)
        e=model.evidence(core,items,ledger,times,2,pairs,cfg)
        self.assertEqual(e['evidence_groups'],1)
        self.assertFalse(model.accept(e,2,.85))

    def test_unavailable_history_still_counts_as_disagreement(self):
        items,ledger,times,pairs,cfg=self.fixture(('B','B','unavailable','unavailable'))
        e=model.evidence(core,items,ledger,times,4,pairs,cfg)
        self.assertGreater(e['agreement'],.4)
        self.assertLess(e['agreement'],.6)
        self.assertEqual(len(e['votes']),2)
        self.assertFalse(model.accept(e,2,.85))

    def test_window_class_requires_unique_current_destination(self):
        q=observation()
        q.candidates={'one':{'id':'one','kind':'window','_bundle':'app'}}
        self.assertEqual(model.resolve_label('window_class:app',q),'one')
        q.candidates['two']={'id':'two','kind':'window','_bundle':'app'}
        self.assertIsNone(model.resolve_label('window_class:app',q))

    def test_unmatured_sample_cannot_supply_evidence(self):
        a=observation(1,bg={'browser':{'course':1}},quality={'browser':1})
        q=observation(2,bg={'browser':{'course':1}},quality={'browser':1})
        self.assertEqual(model.prepare_pairs(core,[a,q],{0},[1])[1],[])

    def test_prediction_spacing_does_not_use_outcomes(self):
        items=[observation(i) for i in [0,1,30,31,60]]
        self.assertEqual(model.independent_positions(items,list(range(5))),[0,2,4])


if __name__=='__main__':unittest.main()
