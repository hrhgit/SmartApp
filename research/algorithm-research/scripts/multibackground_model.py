"""Learn next destinations from current-only foreground and unordered backgrounds.

Past/future sequences, inventories and outcomes are not model inputs. Dynamic
output classes are introduced only by mature labels. NumPy implements small
online classifiers with explicitly bounded rehearsal of already mature records.
"""
from collections import Counter
from copy import deepcopy
import hashlib
import gzip
import json
import math
from pathlib import Path

import numpy as np

import evaluate_context_knn as core
from navigation_hierarchy import NavigationHierarchy, SERVICE, ENTRY, disposition, service
from semantic_online import blend, normalize, rank
from temporal_context import TemporalHistory, tokens


VERSION = 'current-one-multibackground-v1'
SCALES = {'recent5m': 5*60, 'recent60m': 60*60, 'long24h': 24*3600, 'recorded_history': None}
BACKGROUND_SETS = {'none': (), 'recent': ('recent5m','recent60m'),
                   'long': ('long24h','recorded_history'), 'combined': tuple(SCALES)}


def background_snapshot(history, event, recorded_start):
    """Unordered aggregates using only information observed through this event."""
    now = event['timestampMs']
    current_span = history._span(event)
    spans = history.spans + ([current_span] if current_span else [])
    result = {}
    for name, seconds in SCALES.items():
        start = recorded_start if seconds is None else now-seconds*1000
        available_seconds = max(0., (now-max(start,recorded_start))/1000)
        count, recent, dwell = Counter(), Counter(), Counter()
        arrivals, observed_seconds = 0, 0.
        half_life = (seconds/2 if seconds else 7*24*3600)*1000
        for episode in history.episodes:
            if not episode['confirmed'] or not start <= episode['arrival_ms'] <= now:
                continue
            arrivals += 1
            known_at = max(episode['arrival_ms'],episode['identity_known_ms'])
            age_weight = 2.**(-max(0.,now-known_at)/half_life)
            for key,value in tokens(episode['target']).items():
                count[key] += value
                recent[key] = max(recent[key],value*age_weight)
        for span in spans:
            width = span['end_ms']-span['start_ms']
            overlap = max(0.,min(now,span['end_ms'])-max(start,span['start_ms']))
            observed = span['seconds']*overlap/width if width>0 else 0.
            observed_seconds += observed
            for key,value in tokens(span['target']).items():
                dwell[key] += value*observed
        result[name] = {'count':dict(count),'recent':dict(recent),'dwell':dict(dwell),
                        'confirmed_arrivals':arrivals,'observed_nonidle_seconds':observed_seconds,
                        'available_span_seconds':available_seconds,'requested_span_seconds':seconds,
                        'history_start_ms':max(start,recorded_start),
                        'left_truncated':seconds is not None and recorded_start>start,
                        'observation_basis':'confirmed_arrivals_and_nonidle_foreground_continuity'}
    return result


def prepare_inputs(events, observations, excluded_sessions=()):
    history = TemporalHistory(excluded_sessions)
    by_index = {q['index']:q for q in observations}
    if len(by_index)!=len(observations):
        raise ValueError('Duplicate prediction cutoffs')
    output,homes,catalog = {},{},{}
    start = None
    for index,event in enumerate(events):
        # Current native replay rejects archived menu-display records and inserts
        # an identity-free observation boundary. Honor that boundary in features.
        observation_event=event
        if event['kind']=='recommendation.show':
            observation_event={**event,'kind':'capture.gap','data':{'reason':'target_eligibility_boundary'}}
        history.consume(observation_event,index)
        if event['sessionID'] not in history.excluded and start is None:
            start = event['timestampMs']
        if event['kind']=='target.update':
            t = event['data']['target']
            if core.admitted(t) and not core.synthetic(t):catalog[t['id']]=t
            else:catalog.pop(t['id'],None)
        if index not in by_index:
            continue
        q = by_index[index]
        if q['anchor_id']!=event['id'] or q['session']!=event['sessionID']:
            raise ValueError('Frozen prediction input changed')
        x = {k:deepcopy(q[k]) for k in ('position','index','timestamp','origin','app','annotation','target')}
        x.update(current_sequence=[q['origin']],
                 backgrounds=background_snapshot(history,event,start),
                 background_through_index=index,
                 recorded_history_seconds=(event['timestampMs']-start)/1000)
        output[q['position']]=x
        homes[q['position']]=next((k for k,t in sorted(catalog.items())
                                  if t['kind']=='page' and t.get('locator')==ENTRY),None)
    if len(output)!=len(observations):raise ValueError('Missing model inputs')
    return [output[i] for i in range(len(output))],homes,dict(history.audit)


def _distribution(values):
    positive = {k:v for k,v in values.items() if v>0 and math.isfinite(v)}
    total = sum(positive.values())
    return {k:v/total for k,v in positive.items()} if total else {}


def features(q, config):
    """Current target is length 1; no arbitrary x/sequence/background_state read."""
    if q.get('current_sequence') != [q['origin']]:
        raise ValueError('Foreground sequence must contain exactly the current target')
    if q['background_through_index'] != q['index']:
        raise ValueError('Background must be frozen at the input cutoff')
    current = q['origin']
    x = {'current='+current:1.,'kind='+q['target']['kind']:.1}
    if q.get('app'):x['owner='+q['app']]=.25
    # Observed semantic tags share a fixed normalization budget; no body content.
    tags = {t['key']:t['strength']*t['confidence'] for t in q['annotation'].get('tags',())}
    divisor = max(1.,sum(tags.values()))
    x.update({'semantic='+k:.25*v/divisor for k,v in tags.items() if v>0})
    selected = BACKGROUND_SETS[config['backgrounds']]
    channels = ('count',) if config.get('channels')=='count' else ('count','recent','dwell')
    background = {}
    for scale in selected:
        profile = q['backgrounds'][scale]
        # Separate namespaces retain the scale and channel, with comparable mass.
        for channel in channels:
            distribution = _distribution(profile[channel])
            for key,value in distribution.items():
                background[f'{scale}/{channel}/{key}']=value/len(channels)
            background[f'{scale}/observed_{channel}']=float(bool(distribution))*.1
        available = profile['available_span_seconds']
        background[f'{scale}/log_arrivals']=min(1.,math.log1p(profile['confirmed_arrivals'])/8)*.1
        background[f'{scale}/observed_nonidle_fraction']=(min(1.,profile['observed_nonidle_seconds']/available)
                                                        if available>0 else 0.)*.1
        background[f'{scale}/left_truncated']=float(profile['left_truncated'])*.1
    # Deviation from long-run use is a background statistic, not an ordered path.
    if config['backgrounds']=='combined':
        short = _distribution(q['backgrounds']['recent5m']['count'])
        long = _distribution(q['backgrounds']['recorded_history']['count'])
        difference = {k:short.get(k,0.)-long.get(k,0.) for k in short.keys()|long.keys()}
        norm = max(1.,sum(abs(v) for v in difference.values()))
        background.update({'recent_vs_history/'+k:v/norm*.5 for k,v in difference.items() if v})
    x.update({'background/'+k:v for k,v in background.items() if v})
    if config['family']=='interaction_softmax':
        # Learned coefficients can use different backgrounds for each current target.
        x.update({'interaction/'+current+'/'+k:.5*v for k,v in background.items() if v})
    return x


class HashedEncoder:
    """A fixed, signed hash space; no future-derived feature vocabulary."""
    def __init__(self, dimensions):
        self.dimensions=dimensions
        self.cache={}

    def encode(self, values):
        result=Counter()
        for key,value in sorted(values.items()):
            if key not in self.cache:
                digest=hashlib.blake2b(key.encode(),digest_size=8,person=b'SmartApp').digest()
                number=int.from_bytes(digest,'little')
                self.cache[key]=(number%self.dimensions,1. if number>>63 else -1.)
            index,sign=self.cache[key]
            result[index]+=sign*value
        keys=sorted(result)
        return np.asarray(keys,dtype=np.int64),np.asarray([result[k] for k in keys],dtype=np.float64)


class JointClassifier:
    """Online AdaGrad softmax or 32-unit tanh MLP with a learned linear skip.

    Each unique observation is stored once. One optimizer batch contains the
    new mature example and up to seven uniformly sampled prior mature examples;
    this is explicit optimizer rehearsal, not extra independent behavior counts.
    """
    def __init__(self, config):
        self.config=deepcopy(config)
        self.dimensions=config.get('dimensions',2048)
        self.encoder=HashedEncoder(self.dimensions)
        self.rng=np.random.default_rng(config.get('seed',20260923))
        self.labels=[];self.label_index={};self.memory=[];self.update_count=0
        self.W=np.zeros((0,self.dimensions));self.AW=np.zeros_like(self.W)
        self.bias=np.zeros(0);self.Ab=np.zeros(0)
        self.hidden=32 if config['family']=='mlp' else 0
        if self.hidden:
            # Inputs have unit-scale sparse groups, not 2048 dense unit entries.
            # This scale avoids an almost linear tanh layer at initialization.
            self.E=self.rng.normal(0.,.25,(self.dimensions,self.hidden));self.AE=np.zeros_like(self.E)
            self.hb=np.zeros(self.hidden);self.Ahb=np.zeros_like(self.hb)
            self.V=np.zeros((0,self.hidden));self.AV=np.zeros_like(self.V)

    def _class(self,label):
        if label not in self.label_index:
            self.label_index[label]=len(self.labels);self.labels.append(label)
            self.W=np.vstack([self.W,np.zeros((1,self.dimensions))])
            self.AW=np.vstack([self.AW,np.zeros((1,self.dimensions))])
            self.bias=np.append(self.bias,0.);self.Ab=np.append(self.Ab,0.)
            if self.hidden:
                self.V=np.vstack([self.V,np.zeros((1,self.hidden))])
                self.AV=np.vstack([self.AV,np.zeros((1,self.hidden))])
        return self.label_index[label]

    @staticmethod
    def probabilities(logits):
        if logits.shape[-1]==0:return logits.copy()
        e=np.exp(logits-logits.max(axis=-1,keepdims=True))
        return e/e.sum(axis=-1,keepdims=True)

    def _forward(self,X):
        logits=X@self.W.T+self.bias
        h=np.tanh(X@self.E+self.hb) if self.hidden else None
        if self.hidden:logits+=h@self.V.T
        return self.probabilities(logits),h

    def _step(self,name,gradient):
        weights=getattr(self,name);acc=getattr(self,'Ab' if name=='bias' else 'A'+name)
        if not np.isfinite(gradient).all():raise ValueError('Nonfinite classifier gradient')
        acc+=gradient*gradient
        weights-=self.config['lr']*gradient/(np.sqrt(acc)+.1)

    def learn(self,q,label,weight=1.):
        if not math.isfinite(weight) or weight<=0:raise ValueError('Invalid training weight')
        label_index=self._class(label)
        ids,values=self.encoder.encode(features(q,self.config))
        sample=(ids,values,label_index,weight)
        size=min(len(self.memory),self.config.get('rehearsal',7))
        selected=(self.rng.choice(len(self.memory),size,replace=False).tolist() if size else [])
        batch=[sample]+[self.memory[i] for i in selected]
        self.memory.append(sample)
        X=np.zeros((len(batch),self.dimensions));y=[];weights=[]
        for i,(indices,values,label_i,w) in enumerate(batch):
            X[i,indices]=values;y.append(label_i);weights.append(w)
        p,h=self._forward(X);p[np.arange(len(batch)),y]-=1
        error=p*np.asarray(weights)[:,None]/sum(weights)
        gradients={'W':error.T@X,'bias':error.sum(axis=0)}
        # Compute all gradients with pre-update parameters.
        if self.hidden:
            hidden_error=(error@self.V)*(1-h*h)
            gradients.update(V=error.T@h,E=X.T@hidden_error,hb=hidden_error.sum(axis=0))
        for name,gradient in gradients.items():self._step(name,gradient)
        self.update_count+=1

    def predict(self,q):
        if not self.labels:return {}
        indices,values=self.encoder.encode(features(q,self.config))
        logits=self.W[:,indices]@values+self.bias
        if self.hidden:
            h=np.tanh(values@self.E[indices]+self.hb)
            logits+=self.V@h
        p=self.probabilities(logits)
        return dict(zip(self.labels,map(float,p)))

    def state_arrays(self):
        names=['W','AW','bias','Ab']+(['E','AE','hb','Ahb','V','AV'] if self.hidden else [])
        return {name:getattr(self,name) for name in names}


class JointNavigationModel:
    def __init__(self,config):
        self.config=deepcopy(config)
        self.hierarchy=NavigationHierarchy()
        self.classifier=JointClassifier(config)
        self.training_rows=[]

    def learn(self,q,record,ready_index=None):
        if ready_index is not None and ready_index < q['index']:
            raise ValueError('A label cannot mature before its input')
        old=len(self.hierarchy.records)
        self.hierarchy.learn(record)
        if len(self.hierarchy.records)==old:return
        r=self.hierarchy.records[-1]
        self.training_rows.append({'input':q,'record':deepcopy(record),'ready_index':ready_index})
        if r['navigationDisposition']=='navigation':
            self.classifier.learn(q,r['navigationLabel'],r['weight'])

    def _resolve_scores(self,scores,q):
        result=Counter()
        for label,score in scores.items():
            selected=(self.hierarchy.specific(label,q['origin']) or label) if label==SERVICE else label
            if disposition(selected,q['annotation']['identity'])!='navigation':continue
            if selected==SERVICE and q['target'].get('locator')==ENTRY:continue
            result[selected]+=score
        return normalize(result)

    def scores(self,q):
        return self._resolve_scores(self.classifier.predict(q),q)

    def baseline_scores(self,q):
        return normalize(self.hierarchy.scores(q['target']))

    def ranked(self,q,n=4):
        return rank(self.serving_scores(q),n)

    def serving_scores(self,q):
        weight=self.config.get('joint_weight',1.)
        if weight==1.:return self.scores(q)
        return blend([self.scores(q),self.baseline_scores(q)],[weight,1-weight])

    def save(self,directory,name,asof_index,provenance):
        """Save trained weights, optimizer, rehearsal memory and causal metadata."""
        directory=Path(directory)
        if Path(name).name!=name:raise ValueError('Model name must be a basename')
        if any(r['ready_index'] is None or r['ready_index']>asof_index for r in self.training_rows):
            raise ValueError('Model contains labels without verified maturity')
        weights=directory/(name+'-weights-private.npz')
        state=directory/(name+'-state-private.json.gz')
        if weights.exists() or state.exists():raise ValueError('Model artifacts are immutable')
        np.savez_compressed(weights,**self.classifier.state_arrays())
        payload={'version':VERSION,'config':self.config,'foreground_sequence_length':1,
                 'background_inventory_features':False,'deployment':False,
                 'asof_event_index':asof_index,'provenance':provenance,
                 'weights_file':weights.name,'weights_sha256':core.sha(weights),
                 'labels':self.classifier.labels,'updates':self.classifier.update_count,
                 'rng_state':self.classifier.rng.bit_generator.state,'training_rows':self.training_rows}
        with gzip.open(state,'wt') as f:json.dump(payload,f,ensure_ascii=False)
        return state

    @classmethod
    def load(cls,state_path):
        state_path=Path(state_path)
        with gzip.open(state_path,'rt') as f:payload=json.load(f)
        if (payload['version']!=VERSION or payload['foreground_sequence_length']!=1
                or payload['background_inventory_features'] is not False):
            raise ValueError('Unsupported model input contract')
        filename=payload['weights_file']
        if Path(filename).name!=filename:raise ValueError('Weights must be in the same artifact directory')
        weights=state_path.with_name(filename)
        if core.sha(weights)!=payload['weights_sha256']:raise ValueError('Trained weights checksum mismatch')
        model=cls(payload['config']);c=model.classifier
        for label in payload['labels']:c._class(label)
        with np.load(weights,allow_pickle=False) as arrays:
            if set(arrays.files)!=set(c.state_arrays()):raise ValueError('Unexpected weight arrays')
            for name,expected in c.state_arrays().items():
                values=arrays[name]
                if values.shape!=expected.shape or not np.isfinite(values).all():
                    raise ValueError('Invalid saved weight shape or value')
                setattr(c,name,values.copy())
        model.training_rows=payload['training_rows']
        for row in model.training_rows:
            q=row['input'];ready=row['ready_index']
            if ready is None or not q['index']<=ready<=payload['asof_event_index']:
                raise ValueError('Saved input or label is beyond the declared model cutoff')
            before=len(model.hierarchy.records)
            model.hierarchy.learn(row['record'])
            if len(model.hierarchy.records)!=before+1:raise ValueError('Duplicate or invalid saved behavior')
            record=model.hierarchy.records[-1]
            if record['navigationDisposition']=='navigation':
                ids,values=c.encoder.encode(features(q,c.config))
                c.memory.append((ids,values,c.label_index[record['navigationLabel']],record['weight']))
        if len(c.memory)!=payload['updates']:raise ValueError('Optimizer sample count mismatch')
        c.update_count=payload['updates'];c.rng.bit_generator.state=payload['rng_state']
        return model
