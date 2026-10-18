from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import evaluation_contract as scoring
from compare_fresh_models import Experiment
from prepare_complete_validation import combine


def observation(position):
    return {'position':position,'sample_id':f's{position}','anchor_id':f'a{position}',
        'index':10+position*10,'ready_index':19+position*10,'origin':'app:current',
        'timestamp':position,'target':{'id':'app:current','kind':'app','bundleID':'current'},'annotation':{}}


def row(position):
    return {'position':position,'sample_id':f's{position}','truth_action':'app:next',
        'truth_metadata':{'id':'app:next','kind':'app','bundleID':'next'},
        'evidence_grade':'original','evidence_count':1,'split':'test'}


class CompleteValidationTests(unittest.TestCase):
    def datasets(self):
        observations=[observation(i) for i in range(3)]
        old={'observations':observations[:2],'evaluation':[row(1)]}
        fresh={'observations':observations,'evaluation':[row(2)],'reference_stream':[
            {'position':i,'record':{'label':'app:next'}} for i in range(3)]}
        return old,fresh

    def test_union_preserves_all_fixed_members_without_adding_training_anchors(self):
        old,fresh=self.datasets();result=combine(old,fresh)
        self.assertEqual([r['sample_id'] for r in result['evaluation']],['s1','s2'])
        self.assertEqual(result['evaluation'][0]['truth_action'],old['evaluation'][0]['truth_action'])
        self.assertEqual(result['evaluation'][0]['evidence_grade'],'original')
        self.assertEqual(len(result['observations']),3)
        self.assertNotIn('complete_validation_segment',old['evaluation'][0])

    def test_duplicate_members_rejected(self):
        old,fresh=self.datasets();fresh['evaluation']=[row(1)]
        with self.assertRaisesRegex(ValueError,'Duplicate'):combine(old,fresh)

    def test_changed_old_scoring_label_rejected(self):
        old,fresh=self.datasets();fresh['reference_stream'][1]['record']['label']='wrong'
        with self.assertRaisesRegex(ValueError,'scoring label changed'):combine(old,fresh)

    def test_changed_original_input_rejected(self):
        old,fresh=self.datasets();old=deepcopy(old);old['observations'][1]['index']=11
        with self.assertRaisesRegex(ValueError,'input changed'):combine(old,fresh)

    def test_exactly_two_cohorts_and_composite_sort_not_top4_sort(self):
        with tempfile.TemporaryDirectory() as directory:
            exp=Experiment.__new__(Experiment);exp.out=Path(directory)
            exp.meta={'evaluation_scope':'complete_two_cohorts'}
            exp.rows=scoring.prepare_evaluation({'observations':[observation(i) for i in range(2)],
                                                'evaluation':[row(i) for i in range(2)]})
            reference='joint/linear_top2_lr0.1_seed20260923'
            exp.outputs={reference:{0:{'predictions':['app:next']},1:{'predictions':[]}},
                'a_higher_top4':{i:{'predictions':['x','y','z','app:next']} for i in range(2)}}
            exp.preference_weights={mode:{0:1.,1:1.} for mode in ('fixed','global_frequency','context_frequency')}
            exp.scope={'sizes':{'all':2,'evidence_total':2},'fixed_candidate_n':1,'tail_candidate_n':1}
            metrics=exp.evaluate()
            self.assertEqual(set(metrics[reference]),{'all','evidence_total'})
            self.assertLess(metrics[reference]['all']['top4_hits'],metrics['a_higher_top4']['all']['top4_hits'])
            self.assertGreater(metrics[reference]['all']['rank_score']['mean'],metrics['a_higher_top4']['all']['rank_score']['mean'])
            report=(exp.out/'report-private.md').read_text()
            self.assertLess(report.index('| '+reference),report.index('| a_higher_top4'))
            self.assertIn('不称全新独立测试',report)


if __name__=='__main__':unittest.main()
