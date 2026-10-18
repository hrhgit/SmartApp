"""Alternative trainable corrections over an immutable causal statistical prior."""
from collections import Counter
from copy import deepcopy
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from statistical_residual import DIMENSIONS, ResidualRanker, loss_gradient


REFERENCE_SEED = 20260923
SEEDS = (REFERENCE_SEED, REFERENCE_SEED+1, REFERENCE_SEED+2)
FAMILIES = ('linear_top2', 'linear_ce', 'linear_smooth_top2', 'context_gate_ce', 'nonlinear_top2')


def configurations():
    objectives = {'linear_top2': 'top2_hinge', 'linear_ce': 'cross_entropy',
                  'linear_smooth_top2': 'top2_logistic', 'context_gate_ce': 'cross_entropy',
                  'nonlinear_top2': 'top2_hinge'}
    return [{'id': f'{family}_lr{lr}_seed{seed}', 'group': f'{family}_lr{lr}', 'family': family,
             'objective': objectives[family], 'lr': lr, 'seed': seed, 'rehearsal': 7, 'context': True}
            for family in FAMILIES for lr in (.03, .1) for seed in SEEDS]


def top2_logistic(logits, truth):
    if not 0 <= truth < len(logits): raise ValueError('Truth not in vocabulary')
    other = sorted((i for i in range(len(logits)) if i != truth), key=lambda i: (-logits[i], i))
    gradient = np.zeros(len(logits))
    if len(other) < 2: return 0., gradient
    competitor = other[1]; delta = logits[competitor]-logits[truth]
    slope = float(np.exp(-np.logaddexp(0., -delta)))
    gradient[competitor] = slope; gradient[truth] = -slope
    return float(np.logaddexp(0., delta)), gradient


def gate_features(q, prior):
    if q['current_sequence'] != [q['origin']] or q['background_through_index'] != q['index']:
        raise ValueError('Invalid current-only input')
    n = len(prior); ranked = sorted(prior, reverse=True)
    entropy = float(-np.sum(prior*np.log(prior))/math.log(n)) if n > 1 else 0.
    gap = (math.log(ranked[0]/ranked[1])/10) if n > 1 else 0.
    values = [1., entropy, ranked[0] if n else 0., min(1., gap), min(1., math.log1p(n)/10)]
    for name in ('recent5m', 'recent60m', 'long24h', 'recorded_history'):
        p = q['backgrounds'][name]; available = p['available_span_seconds']
        requested = p.get('requested_span_seconds')
        values.extend([min(1., p['observed_nonidle_seconds']/available) if available else 0.,
                       min(1., available/requested) if requested else float(available > 0),
                       min(1., math.log1p(p['confirmed_arrivals'])/8)])
    return np.array(values)


class JointRanker(ResidualRanker):
    def __init__(self, config):
        super().__init__(config)
        self.E = self.V = self.AE = self.AV = self.gate = self.Agate = None
        if config['family'] == 'nonlinear_top2':
            self.E = np.random.default_rng(config['seed']+100).normal(0., .25, (DIMENSIONS, 16))
            self.V = np.zeros(16); self.AE = np.zeros_like(self.E); self.AV = np.zeros_like(self.V)
        if config['family'] == 'context_gate_ce':
            self.gate = np.zeros(17); self.Agate = np.zeros_like(self.gate)

    def forward(self, example):
        z = super().logits(example)
        h = np.tanh(example['X'] @ self.E) if self.E is not None else None
        if h is not None: z = z + h @ self.V
        if self.gate is None: return z, {'h': h}
        r = np.exp(z-z.max()); r /= r.sum()
        a = float(np.exp(-np.logaddexp(0., -float(example['gate'] @ self.gate))))
        mixture = (1-a)*example['prior']+a*r
        return np.log(mixture), {'r': r, 'a': a, 'mixture': mixture}

    def logits(self, example):
        return self.forward(example)[0]

    def gradients(self, example, label):
        logits, aux = self.forward(example)
        loss, g = (top2_logistic(logits, label) if self.config['objective'] == 'top2_logistic'
                   else loss_gradient(logits, label, self.config['objective']))
        gradients = {}
        if self.gate is not None:
            r, a, mixture = (aux[k] for k in ('r', 'a', 'mixture'))
            u = g*a/mixture
            error = r*(u-np.sum(r*u))
            gradients['gate'] = example['gate']*a*(1-a)*np.sum(g*(r-example['prior'])/mixture)
        else: error = g
        gradients['weights'] = np.asarray(example['X'].T @ error).ravel()
        if self.E is not None:
            h = aux['h']
            gradients['V'] = h.T @ error
            gradients['E'] = np.asarray(example['X'].T @ ((error[:, None]*self.V)*(1-h*h)))
        return loss, gradients

    def learn(self, item, examples, cutoff):
        if item['ready_index'] >= cutoff or item['input_index'] > item['ready_index']:
            raise ValueError('Only strictly matured labels may train a prediction')
        key = tuple(item['key'])
        if key in self.seen: return
        self.seen.add(key); self.audit['mature_relations'] += 1
        x = examples[item['position']]
        if item['label'] not in x['labels']:
            self.audit['truth_unavailable_at_original_cutoff'] += 1
            return
        if not math.isfinite(item['weight']) or item['weight'] <= 0: raise ValueError('Invalid weight')
        size = min(len(self.memory), self.config.get('rehearsal', 7))
        old = self.rng.choice(len(self.memory), size, replace=False).tolist() if size else []
        batch = [item]+[self.memory[i] for i in old]; total = sum(row['weight'] for row in batch)
        gradients = {}
        for row in batch:
            x = examples[row['position']]
            _, one = self.gradients(x, x['labels'].index(row['label']))
            for name, gradient in one.items():
                if name not in gradients: gradients[name] = np.zeros_like(gradient)
                gradients[name] += gradient*row['weight']/total
        for name, gradient in gradients.items():
            if not np.isfinite(gradient).all(): raise ValueError('Nonfinite gradient')
            accumulator = getattr(self, {'weights': 'accumulator', 'gate': 'Agate', 'E': 'AE', 'V': 'AV'}[name])
            accumulator += gradient*gradient
            getattr(self, name)[:] -= self.config['lr']*gradient/(np.sqrt(accumulator)+.1)
        self.memory.append(deepcopy(item)); self.audit['updates'] += 1

    def arrays(self):
        return {name: getattr(self, name) for name in ('weights', 'accumulator', 'E', 'V', 'AE', 'AV', 'gate', 'Agate')
                if getattr(self, name) is not None}

    def save(self, directory, name):
        directory = Path(directory); weights = directory/(name+'-weights.npz'); state = directory/(name+'-state.json.gz')
        if weights.exists() or state.exists(): raise ValueError('Immutable model artifacts')
        np.savez_compressed(weights, **self.arrays())
        with gzip.open(state, 'wt') as f:
            json.dump({'config': self.config, 'memory': self.memory, 'seen': sorted(self.seen),
                       'audit': dict(self.audit), 'rng': self.rng.bit_generator.state,
                       'weights': weights.name, 'sha256': hashlib.sha256(weights.read_bytes()).hexdigest()}, f)
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
            if set(arrays.files) != set(model.arrays()): raise ValueError('Unexpected arrays')
            for name, expected in model.arrays().items():
                if arrays[name].shape != expected.shape or not np.isfinite(arrays[name]).all(): raise ValueError('Invalid array')
                setattr(model, name, arrays[name].copy())
        model.memory = payload['memory']; model.seen = {tuple(k) for k in payload['seen']}
        model.audit = Counter(payload['audit']); model.rng.bit_generator.state = payload['rng']
        return model


def evidence(row):
    return row['evidence_grade'] == 'original' or bool(row.get('new_exact_history'))


def selection_key(rows, predictions):
    from evaluation_contract import measure
    groups = ([r for r in rows if evidence(r)], [r for r in rows if r['evidence_grade'] != 'unsupported'], rows)
    # Sum across predeclared seeds; never select the best seed.
    key = []
    for part in groups:
        scores = [measure(part, p) for p in predictions]
        for k in (2, 1, 4):
            key.append(sum(s[f'top{k}_hits'] for s in scores))
    return tuple(key)


def choose_group(rows, groups):
    return max(groups, key=lambda name: selection_key(rows, groups[name]))


def past_rows(rows, observations, cutoff):
    return [r for r in rows if observations[r['position']]['index'] < cutoff and
            observations[r['position']]['ready_index'] < cutoff]
