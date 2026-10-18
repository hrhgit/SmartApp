"""Synthetic checks of current-only input, causal learning and saved weights."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from compare_multibackground import advance, choose, configurations
from multibackground_model import (BACKGROUND_SETS, JointClassifier, JointNavigationModel,
    SCALES, background_snapshot, features, prepare_inputs)
from prepare_multibackground_dataset import evidence_for_new, require_prefix
from semantic_annotations import annotate_target
from test_temporal_context import context, event, fixture_events, history, record, target


def query(n=0,current='app:A',background='app:C'):
    t=target(current)
    profile={'count':{'target:'+background:2.},'recent':{'target:'+background:.5},
             'dwell':{'target:'+background:10.},'confirmed_arrivals':2,
             'observed_nonidle_seconds':10.,'available_span_seconds':100.,
             'left_truncated':False,'requested_span_seconds':300}
    return {'position':n,'index':n,'timestamp':n*1000,'origin':current,'app':current[4:],
            'target':t,'annotation':annotate_target(t),'current_sequence':[current],
            'background_through_index':n,'backgrounds':{k:deepcopy(profile) for k in SCALES}}


def config(family='interaction_softmax',backgrounds='combined'):
    return {'family':family,'backgrounds':backgrounds,'channels':'rich','lr':.1,
            'dimensions':512,'seed':12,'rehearsal':7}


class InputTests(unittest.TestCase):
    def test_sequence_length_one_is_enforced(self):
        q=query();q['current_sequence']=['app:C','app:A']
        with self.assertRaises(ValueError):features(q,config())

    def test_future_background_is_rejected(self):
        q=query();q['background_through_index']+=1
        with self.assertRaises(ValueError):features(q,config())

    def test_poisoned_unselected_data_never_enters_features(self):
        q=query();before=features(q,config())
        q.update(sequence=['app:FUTURE'],previous_target='app:PREVIOUS',truth_action='app:ANSWER',
                 x={'background/INVENTORY':900},background_state={'candidates':['app:INVENTORY']})
        self.assertEqual(features(q,config()),before)

    def test_background_order_is_ignored(self):
        q=query();q['backgrounds']['recent5m']['count']={'target:a':1,'target:b':2}
        r=deepcopy(q);r['backgrounds']['recent5m']['count']={'target:b':2,'target:a':1}
        self.assertEqual(features(q,config()),features(r,config()))

    def test_scales_and_channels_have_separate_normalization(self):
        x=features(query(),config())
        for scale in SCALES:
            for channel in ('count','recent','dwell'):
                total=sum(v for k,v in x.items() if k.startswith(f'background/{scale}/{channel}/'))
                self.assertAlmostEqual(total,1/3)

    def test_background_ablation_excludes_unselected_scales(self):
        for mode,scales in BACKGROUND_SETS.items():
            x=features(query(),config(backgrounds=mode))
            for scale in SCALES:
                self.assertEqual(any(k.startswith('background/'+scale+'/') for k in x),scale in scales)

    def test_recent_and_long_history_are_different(self):
        es=fixture_events()+[event(4,20,'capture.gap'),context(5,90000,'app:A')]
        values=background_snapshot(history(es),es[-1],0)
        self.assertEqual(values['recent5m']['confirmed_arrivals'],0)
        self.assertEqual(values['long24h']['confirmed_arrivals'],0)
        self.assertEqual(values['recorded_history']['confirmed_arrivals'],1)
        self.assertEqual(values['recorded_history']['observed_nonidle_seconds'],10)
        self.assertFalse(values['recent5m']['left_truncated'])

    def test_refinement_does_not_backfill_frozen_inputs(self):
        es=fixture_events();q=query(0,'app:B');q.update(index=3,anchor_id='3',session='S')
        before=prepare_inputs(es,[q])[0]
        es+=[event(4,11,'target.update',{'target':target('page:P','page','B')}),
             context(5,12,'page:P',reason='resolution_update',resolvesContextEventID='3')]
        self.assertEqual(prepare_inputs(es,[q])[0],before)
        self.assertEqual(before[0]['current_sequence'],['app:B'])
        self.assertNotIn('sequence',before[0]['backgrounds'])

    def test_legacy_display_inserts_boundary_but_no_feature(self):
        es=fixture_events()+[event(4,100,'recommendation.show',{'candidates':['app:X']}),
                             context(5,200,'app:A')]
        q=query(0);q.update(index=5,anchor_id='5',session='S')
        inputs,_,audit=prepare_inputs(es,[q])
        self.assertEqual(inputs[0]['backgrounds']['recorded_history']['confirmed_arrivals'],1)
        self.assertEqual(inputs[0]['backgrounds']['recorded_history']['observed_nonidle_seconds'],10)
        self.assertNotIn('app:X',json.dumps(inputs))
        self.assertEqual(audit['boundary:capture.gap'],1)


class ClassifierTests(unittest.TestCase):
    def test_labels_appear_only_after_learning(self):
        model=JointClassifier(config());q=query()
        self.assertEqual(model.predict(q),{})
        model.learn(q,'app:NEW')
        self.assertEqual(set(model.predict(q)),{'app:NEW'})

    def test_joint_relation_can_change_with_current_and_background(self):
        # XOR: marginal current or background alone cannot predict all four.
        cases=[(query(i,'app:A' if a==0 else 'app:B','app:C' if b==0 else 'app:D'),
                'app:E' if a==b else 'app:F') for i,(a,b) in enumerate([(0,0),(0,1),(1,0),(1,1)])]
        for family in ('interaction_softmax','mlp'):
            model=JointClassifier(config(family))
            for _ in range(150):
                for q,label in cases:model.learn(q,label)
            for q,label in cases:
                self.assertEqual(max(model.predict(q),key=model.predict(q).get),label,family)

    def test_duplicate_arrival_does_not_update_twice(self):
        model=JointNavigationModel(config());q=query();r=record(1,q,'app:B')
        model.learn(q,r,1);model.learn(q,r,1)
        self.assertEqual(model.classifier.update_count,1)
        self.assertEqual(len(model.training_rows),1)

    def test_seed_reproducibility(self):
        a=JointClassifier(config('mlp'));b=JointClassifier(config('mlp'))
        for i in range(20):
            q=query(i);label='app:B' if i%2 else 'app:D'
            a.learn(q,label);b.learn(q,label)
        self.assertEqual(a.predict(query()),b.predict(query()))

    def test_saved_weights_and_optimizer_resume_exactly(self):
        for family in ('interaction_softmax','mlp'):
            model=JointNavigationModel({**config(family),'joint_weight':.5})
            for i in range(12):
                q=query(i,background='app:C' if i%2 else 'app:D')
                model.learn(q,record(i,q,'app:B' if i%2 else 'app:E'),i+1)
            with tempfile.TemporaryDirectory() as folder:
                path=model.save(folder,'fixture',20,{})
                restored=JointNavigationModel.load(path)
                self.assertEqual(model.serving_scores(query(20)),restored.serving_scores(query(20)))
                q=query(21);r=record(21,q,'app:NEW')
                model.learn(q,r,22);restored.learn(q,r,22)
                for key,values in model.classifier.state_arrays().items():
                    np.testing.assert_array_equal(values,restored.classifier.state_arrays()[key])

    def test_weights_corruption_is_rejected(self):
        model=JointNavigationModel(config());q=query();model.learn(q,record(1,q,'app:B'),1)
        with tempfile.TemporaryDirectory() as folder:
            path=model.save(folder,'fixture',1,{})
            weights=Path(folder)/'fixture-weights-private.npz'
            with weights.open('ab') as f:f.write(b'corrupt')
            with self.assertRaisesRegex(ValueError,'checksum'):JointNavigationModel.load(path)

    def test_unmatured_save_is_rejected(self):
        model=JointNavigationModel(config());q=query();model.learn(q,record(1,q,'app:B'),5)
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):model.save(folder,'fixture',4,{})


class ReplayTests(unittest.TestCase):
    def test_label_becomes_available_strictly_after_ready_index(self):
        q=query();stream=[{'position':0,'ready_index':5,'label':'app:B','outcome_id':'1',
            'outcome_timestamp':1000,'weight':1.,'record':{'target':target('app:B'),
                'target_evidence_index':4,'observed_step':1}}]
        model=JointNavigationModel(config())
        self.assertEqual(advance(model,stream,0,[q],5),0)
        self.assertEqual(model.classifier.labels,[])
        self.assertEqual(advance(model,stream,0,[q],6),1)
        self.assertEqual(model.classifier.labels,['app:B'])

    def test_future_outcome_metadata_is_rejected(self):
        stream=[{'position':0,'ready_index':5,'record':{'target_evidence_index':6}}]
        with self.assertRaises(ValueError):advance(JointNavigationModel(config()),stream,0,[query()],7)

    def test_new_export_must_preserve_exact_old_prefix(self):
        require_prefix([{'id':1}],[{'id':1},{'id':2}])
        with self.assertRaises(ValueError):require_prefix([{'id':1}],[{'id':2}])

    def test_selection_cannot_read_late_or_new_metrics(self):
        validation={k:{'top1_hits':1,'top4_hits':1} for k in ('original','expanded','all')}
        a={'config':{'id':'a','backgrounds':'combined'},'validation':validation,'new_test':None}
        b=deepcopy(a);b['config']['id']='b';b['new_test']={'score':999999}
        self.assertIs(choose([a,b]),a)

    def test_same_outcome_keeps_strongest_available_evidence(self):
        observations=[query(0,'app:A'),query(1,'app:C'),query(10,'app:A')]
        rows=[{'position':2,'truth_action':'app:B'}]
        reference=[{'position':i,'ready_index':i+5,
                    'record':{'label':'app:B','outcome_id':'same'}} for i in (0,1)]
        with patch('prepare_multibackground_dataset.compare_annotations',return_value={'level':'weak','score':.1}):
            evidence_for_new(rows,observations,reference)
        self.assertEqual(rows[0]['evidence_grade'],'added_exact')
        self.assertEqual(rows[0]['evidence_count'],1)

    def test_new_evidence_excludes_unmatured_reference(self):
        rows=[{'position':1,'truth_action':'app:B'}]
        reference=[{'position':0,'ready_index':10,'record':{'label':'app:B','outcome_id':'future'}}]
        evidence_for_new(rows,[query(0),query(10)],reference)
        self.assertEqual(rows[0]['evidence_grade'],'unsupported')

    def test_grid_cannot_enable_ordered_recent_features(self):
        self.assertEqual(len(configurations()),20)
        for c in configurations():
            self.assertNotIn('sequence',c)
            self.assertNotIn('previous',c)


if __name__=='__main__':unittest.main()
