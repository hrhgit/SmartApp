import unittest
from collections import defaultdict

import evaluate_context_knn as core
import evaluate_context_rules as model
from test_evaluate_context_knn import observation


class ContextRuleTests(unittest.TestCase):
    def fixture(self):
        items=[observation(i,truth='B',bg={'browser':{'app:background':1}},quality={'browser':1}) for i in [1,2,100]]
        for q in items:q.history={'position:1:app:previous':1}
        keys=[model.conditions(q) for q in items];index=defaultdict(list)
        for i in range(2):
            for key in keys[i]:index[key].append(i)
        ledger=[{'target_event_id':'a'},{'target_event_id':'b'}]
        times={'a':('session',0),'b':('session',60000)}
        config={'foreground':'exact','history':'previous_app','lookback_hours':24.,'representation':'exact'}
        return items,ledger,times,keys,index,config

    def test_rule_requires_shared_background_and_previous_app(self):
        items,ledger,times,keys,index,config=self.fixture()
        rules=model.query_rules(items,ledger,times,2,keys,index,config)
        result=model.predict(core,items[2],rules,2,.85)
        self.assertEqual(result['predictions'][0],'B')
        items[2].history={'position:1:app:different':1};keys[2]=model.conditions(items[2])
        self.assertEqual(model.query_rules(items,ledger,times,2,keys,index,config),[])

    def test_query_label_and_future_rule_members_do_not_affect_prediction(self):
        items,ledger,times,keys,index,config=self.fixture()
        before=model.query_rules(items,ledger,times,2,keys,index,config)
        items[2].truth='C'
        for key in keys[2]:index[key].append(2)
        after=model.query_rules(items,ledger,times,2,keys,index,config)
        self.assertEqual(before,after)

    def test_unmatured_history_cannot_complete_support(self):
        items,ledger,times,keys,index,config=self.fixture()
        items[1].ready_index=items[2].index
        rules=model.query_rules(items,ledger,times,2,keys,index,config)
        self.assertEqual(model.predict(core,items[2],rules,2,.85)['predictions'],[])

    def test_repeated_anchor_for_same_arrival_does_not_complete_support(self):
        items,ledger,times,keys,index,config=self.fixture()
        ledger[1]['target_event_id']='a'
        rules=model.query_rules(items,ledger,times,2,keys,index,config)
        self.assertEqual(model.predict(core,items[2],rules,2,.85)['predictions'],[])

    def test_unavailable_majority_is_not_discarded_to_raise_confidence(self):
        items,ledger,times,keys,index,config=self.fixture()
        items[0].truth=items[1].truth='unavailable'
        rules=model.query_rules(items,ledger,times,2,keys,index,config)
        result=model.predict(core,items[2],rules,2,.85)
        self.assertEqual(result['predictions'],[])
        self.assertEqual(result['abstain_reason'],'winner_unavailable_or_ambiguous')


if __name__=='__main__':unittest.main()
