import unittest
from dataclasses import fields

import evaluate_context_knn as core
import evaluate_dwell_cleaning as m


class CleaningTests(unittest.TestCase):
    def setUp(self):
        values={f.name:None for f in fields(core.Observation)}
        values.update(anchor_id='anchor',next_event_id='B',session='s',current='A',truth='old',
            ready_index=100,index=0)
        self.q=core.Observation(**values)
        self.events={'anchor':{'monotonicMs':0}}

    def visit(self,name,start,end,next_id='C',complete=True,refinements=None):
        target={'id':name,'kind':'page','locator':f'https://example.invalid/{name}','bundleID':'browser'}
        return {'session':'s','left_censored':False,'arrival_start_monotonic_ms':start,
            'initial_target':target,'refinements':refinements or [],'end_monotonic_ms':end,
            'complete':complete,'end_event_id':next_id,'observed_seconds':(end-start)/1000,
            'navigation_class':'same_tab_navigation'}

    def test_skip_short_and_learn_stable_successor(self):
        visits={'B':self.visit('B',1000,2000),'C':self.visit('C',2000,12000)}
        q,d=m.derive_label(self.q,visits,self.events,3)
        self.assertEqual(q.truth,'C')
        self.assertEqual(d['skipped'][0]['category'],'navigation_intermediate')
        self.assertEqual(q.ready_index,100)

    def test_return_is_not_relabelled_as_later_destination(self):
        visits={'B':self.visit('B',1000,2000,next_id='A'),'A':self.visit('A',2000,4000)}
        q,d=m.derive_label(self.q,visits,self.events,3)
        self.assertIsNone(q)
        self.assertEqual(d['status'],'returned_to_current_target')

    def test_censor_before_threshold_not_skipped(self):
        q,d=m.derive_label(self.q,{'B':self.visit('B',1000,2000,complete=False)},self.events,3)
        self.assertIsNone(q)
        self.assertEqual(d['status'],'observation_boundary_before_threshold')

    def test_long_visit_can_mature_before_later_censor(self):
        q,d=m.derive_label(self.q,{'B':self.visit('B',1000,100000,complete=False)},self.events,3)
        self.assertIsNotNone(q)
        self.assertEqual(d['evidence_available_monotonic_ms'],4000)

    def test_stability_after_horizon_cannot_be_used(self):
        q,d=m.derive_label(self.q,{'B':self.visit('B',29000,50000)},self.events,3)
        self.assertIsNone(q)
        self.assertEqual(d['status'],'insufficient_dwell_before_30s')

    def test_future_resolution_cannot_change_label(self):
        refinement={'monotonic_ms':35000,'target':{'id':'C','kind':'page','locator':'https://example.invalid/C'}}
        q,d=m.derive_label(self.q,{'B':self.visit('B',1000,50000,refinements=[refinement])},self.events,3)
        self.assertEqual(q.truth,'B')

    def test_exact_threshold_is_retained(self):
        q,d=m.derive_label(self.q,{'B':self.visit('B',1000,4000)},self.events,3)
        self.assertIsNotNone(q)
        self.assertEqual(d['skipped'],[])


if __name__=='__main__':unittest.main()
