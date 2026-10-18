import unittest

import describe_dwell as m


class DwellTests(unittest.TestCase):
    def fixture(self,contexts):
        targets={
            'A':{'id':'A','name':'A','kind':'page','bundleID':'browser','locator':'https://example.invalid/a'},
            'B':{'id':'B','name':'B','kind':'page','bundleID':'browser','locator':'https://example.invalid/b'},
            'app':{'id':'app','name':'Browser','kind':'app','bundleID':'browser'},
            'other':{'id':'other','name':'Other','kind':'app','bundleID':'other'}}
        events=[]
        def add(kind,t,data,source='browser-extension:browser',app=None):
            e={'id':str(len(events)),'sessionID':'session','sequence':len(events)+1,
               'kind':kind,'source':source,'data':data,'timestampMs':t+10000,'monotonicMs':t}
            if app:e['app']={'bundleID':app,'name':app}
            events.append(e);return e
        add('session.start',-10,{})
        for target in targets.values():
            target['admission']={'basis':'supported_browser_page','policyVersion':'target-eligibility-v4'}
            add('target.update',-1,{'target':target})
        anchor=None
        for kind,t,value in contexts:
            if kind=='context.change':
                data={'reason':'foreground_change',**(value if isinstance(value,dict) else {'targetID':value})}
                anchor=add(kind,t,data,app=targets[data['targetID']]['bundleID'])
            elif kind=='resolution':
                add('context.change',t,{'targetID':value,'reason':'resolution_update',
                    'resolvesContextEventID':anchor['id'] if anchor else None},app=targets[value]['bundleID'])
            else:add(kind,t,value)
        samples=[{'anchorEventID':e['id'],'inputCoverage':{'status':'complete'}} for e in events if e['kind']=='context.change']
        return m.extract_visits(events,samples,set())

    def test_short_visit_requires_observed_departure(self):
        visits,_,_=self.fixture([('context.change',0,'A'),('context.change',500,'B')])
        self.assertTrue(visits[0]['complete'])
        self.assertEqual(visits[0]['observed_seconds'],.5)
        self.assertEqual(visits[0]['transient_candidate_thresholds_seconds'],[1,2,3,5])

    def test_resolution_is_same_arrival_not_short_app_visit(self):
        visits,_,_=self.fixture([('context.change',0,'app'),('resolution',300,'A'),('context.change',900,'B')])
        self.assertEqual(len(visits),2)
        self.assertEqual(visits[0]['target']['id'],'A')
        self.assertEqual(visits[0]['observed_seconds'],.6)
        self.assertEqual(visits[0]['arrival_episode_seconds'],.9)

    def test_repeat_notification_does_not_restart_clock(self):
        visits,_,_=self.fixture([('context.change',0,'A'),('context.change',400,'A'),('context.change',1000,'B')])
        self.assertEqual(visits[0]['observed_seconds'],1)
        self.assertEqual(visits[0]['repeat_observations'],1)

    def test_gap_does_not_prove_short_visit(self):
        visits,_,_=self.fixture([('context.change',0,'A'),('capture.gap',200,{}),('context.change',20000,'B')])
        self.assertFalse(visits[0]['complete'])
        self.assertEqual(visits[0]['transient_candidate_thresholds_seconds'],[])
        self.assertEqual(visits[0]['observed_seconds'],.2)

    def test_coarse_same_app_observation_does_not_prove_page_exit(self):
        visits,_,_=self.fixture([('context.change',0,'A'),('context.change',200,'app')])
        self.assertEqual(visits[0]['end_reason'],'same_app_granularity_loss')
        self.assertFalse(visits[0]['complete'])

    def test_idle_is_flagged_not_deleted(self):
        visits,_,_=self.fixture([('context.change',0,'A'),('idle.change',60000,{'idle':True}),('context.change',100000,'B')])
        self.assertEqual(visits[0]['observed_seconds'],100)
        self.assertEqual(visits[0]['idle_seconds'],40)

    def test_app_duration_merges_internal_page_navigation(self):
        _,apps,_=self.fixture([('context.change',0,'A'),('context.change',500,'B'),('context.change',2000,'other')])
        self.assertEqual(apps[0]['observed_seconds'],2)
        self.assertTrue(apps[0]['complete'])

    def test_fast_return_is_annotation_not_confirmed_error(self):
        visits,_,_=self.fixture([('context.change',0,'A'),('context.change',1000,'B'),('context.change',1300,'A')])
        self.assertTrue(visits[1]['returns_to_previous'])
        self.assertEqual(visits[1]['short_status'],'duration_observed')

    def test_source_failure_censors(self):
        visits,_,_=self.fixture([('context.change',0,'A'),('source.status',200,{'observationStatus':'unavailable'}),('context.change',1000,'B')])
        self.assertEqual(visits[0]['end_reason'],'foreground_source_unavailable')
        self.assertFalse(visits[0]['complete'])

    def test_survival_keeps_right_censoring(self):
        result=m.kaplan_meier([{'observed_seconds':1,'complete':True},
            {'observed_seconds':2,'complete':False},{'observed_seconds':3,'complete':True}])
        self.assertEqual(result['median_seconds'],3)

    def test_browser_transition_requires_window_and_tab_evidence(self):
        a={'kind':'page','bundleID':'browser','locator':'https://example.invalid/a'}
        b={**a,'locator':'https://elsewhere.invalid/b'}
        tab={'id':'tab:epoch:1','tabID':'1','containerID':'window:epoch:1'}
        self.assertEqual(m.navigation_class(a,tab,b,tab),'same_tab_navigation')
        self.assertEqual(m.navigation_class(a,tab,b,{**tab,'id':'tab:epoch:2','tabID':'2'}),'same_window_tab_switch')
        self.assertEqual(m.navigation_class(a,tab,b,{**tab,'containerID':'window:epoch:2'}),'browser_window_switch')
        self.assertEqual(m.navigation_class(a,None,b,tab),'unknown_browser_transition')

    def test_entrance_uses_past_inventory_not_future(self):
        # Existing contexts with no inventory must remain unknown even if later inventory is present.
        contexts=[('context.change',0,{'targetID':'A','entranceID':'new'}),('context.change',500,'B'),
            ('inventory.change',600,{'entrances':[{'id':'new','targetID':'A','tabID':'1','containerID':'w'}]})]
        visits,_,_=self.fixture(contexts)
        self.assertIsNone(visits[0]['entrance'])
        self.assertEqual(visits[0]['navigation_class'],'unknown_browser_transition')

    def test_same_url_in_different_tabs_keeps_two_arrivals(self):
        entries=[{'id':f'tab{x}','targetID':'A','tabID':str(x),'containerID':'w'} for x in [1,2]]
        visits,_,_=self.fixture([('inventory.change',-0.5,{'entrances':entries}),
            ('context.change',0,{'targetID':'A','entranceID':'tab1'}),
            ('context.change',500,{'targetID':'A','entranceID':'tab2'})])
        self.assertEqual(len(visits),2)
        self.assertEqual(visits[0]['navigation_class'],'same_window_tab_switch')
        self.assertTrue(visits[0]['complete'])


if __name__=='__main__':unittest.main()
