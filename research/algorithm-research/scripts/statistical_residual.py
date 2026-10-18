"""Online contextual ranking corrections to a causal statistical prior.

score = log(statistical probability) + learned correction. There is no mixture
coefficient. Training examples retain their original, pre-outcome score vectors
and output vocabularies; a newly discovered outcome cannot vote for itself.
"""
from collections import Counter
from copy import deepcopy
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix


DIMENSIONS = 4096
SCALES = ('recent5m', 'recent60m', 'long24h', 'recorded_history')


def encode_features(q, scores, descriptions):
    if q['current_sequence'] != [q['origin']] or q['background_through_index'] != q['index']:
        raise ValueError('Expected causal current-only input')
    labels = sorted(scores)
    p = np.array([scores[k] for k in labels], dtype=float)
    if not np.isfinite(p).all() or np.any(p <= 0):
        raise ValueError('Prior probabilities must be finite and positive')
    if len(p): p /= p.sum()
    ranks = {k: i+1 for i, k in enumerate(sorted(labels, key=lambda k: (-scores[k], k)))}
    profiles = {}
    for scale in SCALES:
        profile = q['backgrounds'][scale]
        for channel in ('count', 'recent', 'dwell'):
            values = profile[channel]
            total = sum(values.values())
            profiles[scale, channel] = {k: v/total for k, v in values.items()} if total else {}
    indices, values, indptr = [], [], [0]
    for label, probability in zip(labels, p):
        description = descriptions.get(label, {})
        owner, kind = description.get('owner', ''), description.get('kind', 'unknown')
        base = {'log_probability': max(-4., math.log(probability)/5),
                'probability': probability, 'reciprocal_rank': 1/ranks[label],
                'target/'+label: .5, 'current/'+q['origin']+'/next/'+label: .5,
                'kind/'+kind: .25, 'current_kind/'+q['target']['kind']+'/next/'+kind: .25}
        context = {}
        aliases = description.get('aliases', [label])
        target_keys = ['target:'+alias for alias in aliases]
        masses = {}
        for scale in SCALES:
            profile = q['backgrounds'][scale]
            available = profile['available_span_seconds']
            coverage = min(1., profile['observed_nonidle_seconds']/available) if available else 0.
            for channel in ('count', 'recent', 'dwell'):
                dist = profiles[scale, channel]
                mass = sum(dist.get(k, 0.) for k in target_keys)
                masses[scale, channel] = mass
                prefix = scale+'/'+channel
                context[prefix+'/target'] = math.sqrt(mass)
                context[prefix+'/owner'] = math.sqrt(dist.get('owner:'+owner, 0.)) if owner else 0.
                context[prefix+'/present'] = .2*float(mass > 0)
                context[prefix+'/observed_target'] = math.sqrt(mass)*coverage
        context['recent_vs_history'] = (math.sqrt(masses['recent5m', 'count'])-
                                         math.sqrt(masses['recorded_history', 'count']))
        context.update({'current/'+q['origin']+'/'+k: .5*v for k, v in list(context.items()) if v})
        encoded = Counter()
        for offset, features in ((0, base), (DIMENSIONS//2, context)):
            for key, value in features.items():
                if not value: continue
                digest = hashlib.blake2b(key.encode(), digest_size=8, person=b'StatResidual').digest()
                number = int.from_bytes(digest, 'little')
                encoded[offset+number % (DIMENSIONS//2)] += value*(1 if number >> 63 else -1)
        for index in sorted(encoded):
            indices.append(index); values.append(encoded[index])
        indptr.append(len(indices))
    X = csr_matrix((values, indices, indptr), shape=(len(labels), DIMENSIONS), dtype=float)
    return {'labels': labels, 'prior': p, 'X': X}


def loss_gradient(logits, truth, objective):
    """CE or unit-margin hinge against the second highest incorrect target."""
    if not len(logits) or not 0 <= truth < len(logits):
        raise ValueError('Truth must be in the frozen prediction vocabulary')
    gradient = np.zeros(len(logits))
    if objective == 'cross_entropy':
        exp = np.exp(logits-logits.max()); p = exp/exp.sum()
        gradient = p.copy(); gradient[truth] -= 1
        return float(-math.log(max(p[truth], 1e-300))), gradient
    if objective != 'top2_hinge': raise ValueError('Unknown objective')
    other = sorted((i for i in range(len(logits)) if i != truth), key=lambda i: (-logits[i], i))
    if len(other) < 2: return 0., gradient
    competitor = other[1]
    loss = max(0., 1.+logits[competitor]-logits[truth])
    if loss:
        gradient[competitor] = 1.; gradient[truth] = -1.
    return loss, gradient


class ResidualRanker:
    def __init__(self, config):
        self.config = deepcopy(config)
        self.weights = np.zeros(DIMENSIONS)
        self.accumulator = np.zeros(DIMENSIONS)
        self.memory, self.seen = [], set()
        self.rng = np.random.default_rng(config.get('seed', 20260923))
        self.audit = Counter()

    def logits(self, example):
        return np.log(example['prior']) + np.asarray(example['X'] @ self.weights).ravel()

    def scores(self, example):
        if not example['labels']: return {}
        z = self.logits(example); p = np.exp(z-z.max()); p /= p.sum()
        return dict(zip(example['labels'], map(float, p)))

    def learn(self, item, examples, cutoff):
        if item['ready_index'] >= cutoff or item['input_index'] > item['ready_index']:
            raise ValueError('Only strictly matured labels may train a prediction')
        key = tuple(item['key'])
        if key in self.seen: return
        self.seen.add(key)
        self.audit['mature_relations'] += 1
        example = examples[item['position']]
        if item['label'] not in example['labels']:
            self.audit['truth_unavailable_at_original_cutoff'] += 1
            return
        if not math.isfinite(item['weight']) or item['weight'] <= 0:
            raise ValueError('Invalid sample weight')
        size = min(len(self.memory), self.config.get('rehearsal', 7))
        old = self.rng.choice(len(self.memory), size, replace=False).tolist() if size else []
        batch = [item]+[self.memory[i] for i in old]
        gradient = np.zeros(DIMENSIONS); total = sum(r['weight'] for r in batch)
        for row in batch:
            x = examples[row['position']]
            _, g = loss_gradient(self.logits(x), x['labels'].index(row['label']), self.config['objective'])
            gradient += np.asarray(x['X'].T @ g).ravel()*row['weight']/total
        if not self.config['context']: gradient[DIMENSIONS//2:] = 0
        if not np.isfinite(gradient).all(): raise ValueError('Nonfinite update')
        self.accumulator += gradient*gradient
        self.weights -= self.config['lr']*gradient/(np.sqrt(self.accumulator)+.1)
        self.memory.append(deepcopy(item)); self.audit['updates'] += 1

    def save(self, directory, name):
        directory = Path(directory)
        weights = directory/(name+'-weights.npz')
        state = directory/(name+'-state.json.gz')
        if weights.exists() or state.exists(): raise ValueError('Model artifacts are immutable')
        np.savez_compressed(weights, weights=self.weights, accumulator=self.accumulator)
        payload = {'config': self.config, 'memory': self.memory, 'seen': sorted(self.seen),
                   'audit': dict(self.audit), 'rng': self.rng.bit_generator.state,
                   'weights': weights.name, 'sha256': hashlib.sha256(weights.read_bytes()).hexdigest()}
        with gzip.open(state, 'wt') as f: json.dump(payload, f)
        return state

    @classmethod
    def load(cls, state):
        state = Path(state)
        with gzip.open(state, 'rt') as f: payload = json.load(f)
        if Path(payload['weights']).name != payload['weights']: raise ValueError('Invalid weights path')
        path = state.with_name(payload['weights'])
        if hashlib.sha256(path.read_bytes()).hexdigest() != payload['sha256']: raise ValueError('Changed weights')
        model = cls(payload['config'])
        with np.load(path, allow_pickle=False) as arrays:
            for name in ('weights', 'accumulator'):
                value = arrays[name]
                if value.shape != (DIMENSIONS,) or not np.isfinite(value).all(): raise ValueError('Invalid state')
                setattr(model, name, value.copy())
        model.memory = payload['memory']; model.seen = {tuple(k) for k in payload['seen']}
        model.audit = Counter(payload['audit']); model.rng.bit_generator.state = payload['rng']
        return model


def select_top2(results):
    # A fixed early-validation cohort; later and aggregate scores cannot select.
    return max(results, key=lambda r: tuple(r['validation'][group][metric] for group, metric in (
        ('original', 'top2_hits'), ('original', 'top1_hits'),
        ('expanded', 'top2_hits'), ('expanded', 'top1_hits'),
        ('original', 'top4_hits'), ('all', 'top2_hits'), ('all', 'top1_hits'))))
