#!/usr/bin/env python3
"""Causal, local-only replay of weighted context/history kNN. Standard library only.

No model is installed in SmartApp. Inputs are immutable canonical v3 samples.
Outputs containing personal destinations are restricted to the private training root.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import heapq
import json
import math
import os
import statistics
import time
from known_inactivity import annotate_events, active_time, is_boundary
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

VERSION = 'context-knn-replay-v2'
PRIVATE_ROOT = Path.home() / 'Library/Application Support/SmartApp/training'
KNOWN_TEST_SESSIONS = {
    '9E7C1B9B-5A48-407F-9C3A-6794B4E95723',
    '3183C01D-D87C-406D-9432-5F99E0B3F6AE',
    'E17A21A0-82F5-4578-9DA6-732F884C114C',
    '55D022D2-816F-40AD-B5F6-D8636DACCC5C',
    'A81BC909-4A1F-45A9-B906-0615E7BBCB10',
}
TEST_MARKERS = ('smartapp_extension_test', 'smartapp_auto_test', 'smartapp_capture_test',
                'smartapp_state_test', 'smartapp-validation', 'smartapp-live-capture',
                'smartapp-state-v3', 'smartapp-state-validation', 'smartapp-document-test')
KINDS = {'app', 'window', 'page', 'directory', 'file'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def utc(ms):
    return datetime.fromtimestamp(ms / 1000, timezone.utc).isoformat()


def verified_snapshot(path):
    required = {'events.jsonl', 'navigation-samples.jsonl', 'target-history.jsonl', 'manifest.json'}
    hashes = {}
    for line in (path / 'SHA256SUMS').read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        name = name.lstrip('*')
        if name not in required:
            raise ValueError('Unexpected checksum path')
        if sha(path / name) != digest:
            raise ValueError(f'Checksum mismatch: {name}')
        hashes[name] = digest
    if hashes.keys() != required:
        raise ValueError('Missing required checksums')
    manifest = json.loads((path / 'manifest.json').read_text())
    if manifest.get('sampleSchemaVersion') != 3 or manifest.get('eventsExcludedFromSamples') != 0:
        raise ValueError('Requires v3 export with no unhandled admission-rejected events')
    return hashes, manifest


def admitted(t):
    return (t.get('kind') in KINDS and t.get('admission', {}).get('basis')
            and t.get('admission', {}).get('policyVersion') in
            {'target-eligibility-v1', 'target-eligibility-v2', 'target-eligibility-v3', 'target-eligibility-v4'})


def synthetic(t):
    return any(s in (str(t.get('locator', '')) + ' ' + str(t.get('name', ''))).lower()
               for s in TEST_MARKERS)


def key_for(t, session):
    # A window instance is never a cross-session semantic identity.
    return ('window@' + session + ':' + t['id']) if t['kind'] == 'window' else t['id']


def target_tokens(t, bundle=None):
    """Semantic hierarchy, never window/tab/pane/PID identities or body text."""
    result = {'kind:' + t['kind']: .15}
    bundle = bundle or t.get('bundleID')
    if bundle:
        result['app:' + bundle] = .5
    if t['kind'] != 'window':
        result['target:' + t['id']] = 1.
    else:
        result['window-class:' + str(bundle)] = 1.
    loc = t.get('locator')
    if loc:
        u = urlsplit(loc)
        base = u.scheme + '://' + u.netloc
        if u.netloc:
            result['origin:' + base] = .3
        # Retain allowed path hierarchy as features; full locator remains target identity.
        parts = [s for s in u.path.split('/') if s]
        for depth in range(1, min(4, len(parts)) + 1):
            result['path:' + base + '/' + '/'.join(parts[:depth])] = .15
    return result


def merge_max(a, b, prefix='', scale=1.):
    for k, v in b.items():
        a[prefix + k] = max(a.get(prefix + k, 0.), v * scale)


def weighted_jaccard(a, b):
    if not a and not b:
        return 1.
    overlap = sum(min(v, b.get(k, 0.)) for k, v in a.items())
    union = sum(a.values()) + sum(b.values()) - overlap
    return overlap / union if union else 1.


def family(source):
    if source.startswith(('browser:', 'browser-extension:')):
        return 'browser'
    if source.startswith('finder:'):
        return 'finder'
    if source.startswith('qspace:'):
        return 'qspace'
    if source.startswith('ax:'):
        return 'desktop_windows'
    return source if source == 'running_apps' else 'other'


@dataclass
class Observation:
    index: int
    ready_index: int
    timestamp: float
    session: str
    anchor_id: str
    sample_id: str
    segment: int
    current: str
    current_app: str | None
    foreground: dict
    background: dict
    history: dict
    history_keys: tuple
    candidates: dict
    recency: dict
    last_visits: dict
    truth: str
    truth_meta: dict
    offset_ms: float
    coverage: str
    strict: bool
    background_count: int
    reference_max_index: int
    source_quality: dict
    next_event_id: str


def assert_asof(reference, anchor, event_indices):
    if reference['sessionID'] != anchor['sessionID'] or reference['sequence'] > anchor['sequence']:
        raise ValueError('Future or cross-session input reference')
    if event_indices[reference['id']] > event_indices[anchor['id']]:
        raise ValueError('Future input reference')


def eligible_neighbors(items, i):
    return [j for j in range(i) if items[j].ready_index < items[i].index]


def make_observations(events, samples, excluded_sessions):
    events = annotate_events(events)
    byid = {e['id']: e for e in events}
    indices = {e['id']: i for i, e in enumerate(events)}
    if len(byid) != len(events):
        raise ValueError('Duplicate event IDs')
    anchors = {s['anchorEventID']: s for s in samples}
    if len(anchors) != len(samples):
        raise ValueError('Duplicate prediction anchors')
    session_indices, session_times = defaultdict(list), defaultdict(list)
    sequence_indices = {(e['sessionID'], e['sequence']): i for i, e in enumerate(events) if 'sequence' in e}
    for i, e in enumerate(events):
        session_indices[e['sessionID']].append(i)
        session_times[e['sessionID']].append(e['monotonicMs'])
    for times in session_times.values():
        if any(a > b for a, b in zip(times, times[1:])):
            raise ValueError('Nonmonotonic session event order')
    excluded = Counter()
    catalog, seen, last_visits, arrival_counts = {}, set(), {}, Counter()
    arrivals_by_kind, inventory_counts = defaultdict(Counter), defaultdict(Counter)
    display, items, last_context, event_features = {}, [], None, {}
    segment, current_session = 0, None
    for i, e in enumerate(events):
        session, data = e['sessionID'], e.get('data', {})
        if session != current_session:
            current_session = session
            catalog = {}  # The app reconstructs admitted targets in each session.
            last_context = None
            segment += 1
        if is_boundary(e):
            last_context = None
            segment += 1
        controlled = session in excluded_sessions
        if e['kind'] == 'target.update':
            t = data['target']
            if admitted(t) and not synthetic(t):
                catalog[t['id']] = t
                if not controlled:
                    key = key_for(t, session)
                    display[key] = dict(t)
                    seen.add(key)
            else:
                catalog.pop(t['id'], None)
        if controlled:
            if e['id'] in anchors:
                excluded['controlled_validation_session'] += 1
            continue
        if e['kind'] == 'inventory.change':
            for entry in data.get('entrances', []):
                t = catalog.get(entry.get('targetID'))
                if t:
                    inventory_counts[t['kind']][key_for(t, session)] += 1
        if e['kind'] != 'context.change':
            continue
        t = catalog.get(data.get('targetID'))
        if not t:
            if e['id'] in anchors:
                excluded['unavailable_or_synthetic_anchor_target'] += 1
            last_context = None
            continue
        current = key_for(t, session)
        app = e.get('app', {}).get('bundleID') or t.get('bundleID')
        fg = target_tokens(t, app)
        resolution = data.get('reason') == 'resolution_update'
        # Each later refinement replaces the prior arrival's representation only after observed.
        arrival_ref = data.get('resolvesContextEventID') if resolution else e['id']
        if resolution and not arrival_ref:
            arrival_ref = last_context
        if not resolution:
            last_context = e['id']
            arrival_counts[current] += 1
            arrivals_by_kind[t['kind']][current] += 1
            display[current] = dict(t)
            last_visits[current] = e['timestampMs']
        event_features[e['id']] = (current, fg, arrival_ref, not resolution, segment)
        s = anchors.get(e['id'])
        if not s:
            continue
        if s.get('schemaVersion') != 3:
            raise ValueError('Requires v3 samples')
        if s['status'] != 'observed_horizon':
            excluded['interrupted_or_export_boundary'] += 1
            continue
        coverage = s.get('inputCoverage', {}).get('status', 'unknown')
        oc = s.get('outcomeCoverage')
        strict = coverage == 'complete' and oc == 'observed'
        legacy = coverage == 'unknown' and oc == 'initial_coverage_unverified'
        if not (strict or legacy):
            excluded['explicit_partial_or_unreliable_outcome_coverage'] += 1
            continue
        if s.get('origin') != 'natural_observed':
            excluded['recommendation_related_anchor'] += 1
            continue
        if not s['outcomes']:
            excluded['no_observed_next_arrival_within_30s'] += 1
            continue
        outcome = s['outcomes'][0]
        oe = byid[outcome['eventID']]
        oi = indices[oe['id']]
        if oi <= i or oe['sessionID'] != session or oe['data'].get('reason') == 'resolution_update':
            raise ValueError('Invalid next-arrival label')
        if oe.get('decisionID'):
            excluded['recommendation_related_next_arrival'] += 1
            continue
        # Future target metadata is label-only; never put it into current candidates/features.
        h = outcome.get('hierarchy', {})
        rawid = outcome.get('targetID')
        # Find metadata using target updates only through the outcome, not end-of-export catalog.
        truth_target = catalog.get(rawid)
        for update in events[i + 1:oi + 1]:
            if update['kind'] == 'target.update' and update['data']['target']['id'] == rawid:
                truth_target = update['data']['target']
        if not truth_target or not admitted(truth_target) or synthetic(truth_target):
            excluded['unavailable_or_synthetic_label_target'] += 1
            continue
        truth = key_for(truth_target, session)
        if truth == current:
            excluded['next_arrival_same_destination'] += 1
            continue
        refs = list(dict.fromkeys(s['recentContextEventIDs'] + s['inventoryEventIDs']))
        for ref in refs:
            assert_asof(byid[ref], e, indices)
        if s.get('inputThroughSequence') != e['sequence']:
            raise ValueError('Mismatched cutoff')
        # Group background by source family. Missing/stale sources provide no empty-set evidence.
        sources = {x['source']: x for x in s['inputCoverage']['sources']}
        background, qualities, entries_asof = {}, defaultdict(list), []
        nbackground = 0
        for ref in s['inventoryEventIDs']:
            inv = byid[ref]
            src, invdata = inv['source'], inv['data']
            sc = sources.get(src, {})
            status = sc.get('status', 'unknown')
            entries = invdata.get('entrances', [])
            if status == 'ready':
                quality = 1.
            elif status in {'legacy_observed', 'unknown'} and entries:
                quality = .5
            else:
                continue
            f = family(src)
            if f == 'other':
                continue
            v = background.setdefault(f, {})
            qualities[f].append(quality)
            entry_targets = {}
            for entry in entries:
                target = catalog.get(entry.get('targetID'))
                if not target:
                    continue
                entries_asof.append(entry)
                ek = key_for(target, session)
                entry_targets[entry['id']] = target
                if ek == current:
                    continue
                nbackground += 1
                merge_max(v, target_tokens(target, entry.get('bundleID')))
            # Selected tab/pane is context even if its containing application is in the background.
            selected = [invdata.get('selectedEntranceID')]
            for window in invdata.get('browserState', {}).get('windows', []):
                selected.append(window.get('selectedEntranceID'))
                for tab in window.get('tabs', []):
                    st = entry_targets.get(tab.get('entranceID'))
                    if st and tab.get('pinned'):
                        merge_max(v, target_tokens(st), 'pinned:', .3)
            for sid in selected:
                st = entry_targets.get(sid)
                if st:
                    merge_max(v, target_tokens(st), 'selected:', .5)
        quality_map = {f: min(q) for f, q in qualities.items()}
        # Registry missing families lower reliability even when another source of that family works.
        for source in s['inputCoverage']['sources']:
            f = family(source['source'])
            if source['source'] in s['inputCoverage'].get('missingSources', []) and f in quality_map:
                quality_map[f] *= .5
        history_arrivals = []
        for ref in s['recentContextEventIDs']:
            ef = event_features.get(ref)
            if not ef or ef[4] != segment:
                continue
            k, tf, parent, independent, _ = ef
            if independent:
                history_arrivals.append([ref, k, tf])
            elif parent:
                for row in reversed(history_arrivals):
                    if row[0] == parent:
                        row[1], row[2] = k, tf
                        break
        hist = {}
        for pos, (_, k, tf) in enumerate(reversed(history_arrivals[-8:])):
            merge_max(hist, tf, 'position:' + str(min(pos, 3)) + ':', .7 ** pos)
            merge_max(hist, tf, 'recent:', .8 ** pos)
        # Dynamic candidates: historical admitted locators plus live, observed window entrances.
        live_ids = {x['targetID'] for x in entries_asof}
        candidates = {}
        for target in catalog.values():
            key = key_for(target, session)
            if key == current:
                continue
            if target['kind'] == 'app' or target.get('locator') or target['id'] in live_ids:
                bundles = {x['bundleID'] for x in entries_asof if x['targetID'] == target['id']}
                bundle = target.get('bundleID') or (sorted(bundles)[0] if bundles else None)
                candidates[key] = {**target, '_bundle': bundle}
        recency = {k: last_visits[k] for k in candidates if k in last_visits}
        deadline = s.get('completedAtMonotonicMs', e['monotonicMs'] + 30000)
        pos = bisect.bisect_left(session_times[session], deadline)
        if pos == len(session_times[session]):
            raise ValueError('Complete sample without maturity event')
        ready_index = session_indices[session][pos]
        if s.get('completedThroughSequence') is not None:
            completed = sequence_indices.get((session, s['completedThroughSequence']))
            if completed is None or events[completed]['monotonicMs'] != deadline:
                raise ValueError('Invalid sample completion evidence')
            ready_index = completed
        truth_meta = dict(truth_target)
        truth_meta['_bundle'] = oe.get('app', {}).get('bundleID') or truth_target.get('bundleID')
        if h.get('appTargetID'):
            truth_meta['_bundle'] = h['appTargetID'].removeprefix('app:')
        truth_meta['_known_at_cutoff'] = truth in seen
        truth_meta['_prior_arrivals'] = arrival_counts[truth]
        items.append(Observation(i, ready_index, e['timestampMs'], session, e['id'], s['id'], segment,
            current, app, fg, background, hist, tuple(r[1] for r in history_arrivals[-8:]),
            candidates, recency, dict(last_visits), truth, truth_meta, outcome['offsetMs'], coverage,
            strict, nbackground, max((indices[r] for r in refs), default=i), quality_map, oe['id']))
    audit = {
        'event_count': len(events), 'sample_count': len(samples), 'included': len(items),
        'exclusions': dict(excluded), 'strict_included': sum(x.strict for x in items),
        'legacy_exploratory_included': sum(not x.strict for x in items),
        'sessions_included': len({x.session for x in items}),
        'range_utc': [utc(events[0]['timestampMs']), utc(events[-1]['timestampMs'])],
        'max_input_reference_violations': sum(x.reference_max_index > x.index for x in items),
    }
    if sum(excluded.values()) + len(items) != len(samples):
        raise ValueError('Sample accounting mismatch')
    return items, audit, arrivals_by_kind, inventory_counts, display


def similarity(a, b):
    foreground = weighted_jaccard(a.foreground, b.foreground)
    history = weighted_jaccard(a.history, b.history)
    allfamilies = set(a.background) | set(b.background)
    denom = len(allfamilies)
    bg_sum, quality_sum = 0., 0.
    for f in set(a.background) & set(b.background):
        q = min(a.source_quality[f], b.source_quality[f])
        quality_sum += q
        bg_sum += q * weighted_jaccard(a.background[f], b.background[f])
    reliability = quality_sum / denom if denom else 0.
    bg = bg_sum / quality_sum if quality_sum else 0.
    # Return numerator and denominator separately so missing state lowers its effective weight.
    state_num = .4 * foreground + .6 * reliability * bg
    state_den = .4 + .6 * reliability
    return state_num, state_den, history, foreground, bg, reliability


def combined_similarity(parts, weight, no_background=False):
    sn, sd, hs, fg, _, _ = parts
    if no_background:
        sn, sd = .4 * fg, .4
    denominator = weight * sd + 1 - weight
    return (weight * sn + (1 - weight) * hs) / denominator if denominator else 0.


def compatible(a, b):
    if a['id'] == b['id']:
        return False
    if a.get('parentID') == b['id'] or b.get('parentID') == a['id']:
        return False
    if a['kind'] == 'app' and a.get('_bundle') and a.get('_bundle') == b.get('_bundle'):
        return False
    if b['kind'] == 'app' and b.get('_bundle') and b.get('_bundle') == a.get('_bundle'):
        return False
    return True


def select_four(ranked, candidates):
    selected = []
    for key in ranked:
        if key not in candidates:
            continue
        if all(compatible(candidates[key], candidates[other]) for other in selected):
            selected.append(key)
            if len(selected) == 4:
                break
    return selected


def rank_votes(votes, query):
    # Unseen/unvisited catalog items are not assigned alphabetical recommendation priority.
    keys = set(votes) | set(query.recency)
    ranked = sorted((k for k in keys if k in query.candidates),
                    key=lambda k: (-votes.get(k, 0.), -query.recency.get(k, -1), k))
    return ranked, select_four(ranked, query.candidates)


def predict_knn(items, i, pairs, weight=.55, k=15, half_life_hours=6., no_background=False):
    query = items[i]
    sims = [(combined_similarity(parts, weight, no_background), j)
            for j, parts in pairs[i] if items[j].truth in query.candidates]
    neighbors = heapq.nlargest(k, ((s, j) for s, j in sims if s > 0), key=lambda x: (x[0], items[x[1]].index))
    votes = Counter()
    for sim, j in neighbors:
        age_hours = max(0., (query.timestamp - items[j].timestamp) / 3600000)
        votes[items[j].truth] += sim ** 2 * 2 ** (-age_hours / half_life_hours)
    raw, shown = rank_votes(votes, query)
    return shown, {'raw': raw[:20], 'neighbors': len(neighbors),
        'best_similarity': neighbors[0][0] if neighbors else 0.,
        'neighbor_label_support': sum(items[j].truth == query.truth for _, j in neighbors),
        'neighbor_median_age_hours': statistics.median(
            [(query.timestamp-items[j].timestamp)/3600000 for _,j in neighbors]) if neighbors else None,
        'neighbor_examples': [items[j].sample_id for _, j in neighbors[:5]]}


def predict_baseline(items, i, pairs, markov=False):
    q = items[i]
    votes = Counter()
    if markov:
        for j, _ in pairs[i]:
            if items[j].current == q.current and items[j].truth in q.candidates:
                age = max(0., (q.timestamp - items[j].timestamp) / 3600000)
                votes[items[j].truth] += 2 ** (-age / 6)
    return rank_votes(votes, q)[1]


def metric(rows):
    n = len(rows)
    covered = sum(x['covered'] for x in rows)
    hits = sum(x['hit4'] for x in rows)
    return {'n': n, 'top1_hits': sum(x['hit1'] for x in rows), 'top4_hits': hits,
            'top1': sum(x['hit1'] for x in rows) / n if n else None,
            'top4': hits / n if n else None,
            'mrr4': sum(x['rr'] for x in rows) / n if n else None,
            'candidate_coverage': covered / n if n else None,
            'covered_n': covered, 'top4_given_covered': hits / covered if covered else None,
            'app_level_top4': sum(x['app_hit4'] for x in rows) / n if n else None}


def score_row(q, predictions):
    rank = predictions.index(q.truth) + 1 if q.truth in predictions else None
    truth_app = q.truth_meta.get('_bundle')
    return {'hit1': rank == 1, 'hit4': rank is not None, 'rr': 1 / rank if rank else 0.,
            'covered': q.truth in q.candidates, 'app_hit4': bool(truth_app) and
            any(q.candidates[p].get('_bundle') == truth_app for p in predictions)}


def evaluate(items, pairs, positions, config=None, baseline=None):
    rows, details, latencies = [], [], []
    for i in positions:
        start = time.perf_counter()
        if baseline:
            preds = predict_baseline(items, i, pairs, baseline == 'markov')
            extra = {}
        else:
            preds, extra = predict_knn(items, i, pairs, **config)
        latencies.append((time.perf_counter() - start) * 1000)
        row = score_row(items[i], preds)
        rows.append(row)
        details.append({'position': i, 'predictions': preds, **row, **extra})
    return metric(rows), details, latencies


def failure_reason(q, row):
    if row['hit4']:
        return 'hit'
    if not row['covered']:
        return 'first_seen_destination' if not q.truth_meta['_known_at_cutoff'] else 'known_but_not_in_candidate_set'
    if q.truth in row.get('raw', [])[:4] and q.truth not in row['predictions']:
        return 'hierarchy_dedup_suppressed'
    if row['app_hit4']:
        return 'right_application_wrong_destination'
    if q.truth_meta['_prior_arrivals'] < 3:
        return 'rare_destination_fewer_than_3_prior_arrivals'
    if row.get('neighbor_label_support', 0) == 0:
        return 'nearest_contexts_have_other_outcomes'
    return 'vote_ranking_error'


def percent(x):
    return '—' if x is None else f'{100*x:.1f}%'


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def describe_target(meta):
    value = meta.get('locator') or meta.get('name') or meta.get('id', '?')
    return str(value).replace('|', '\\|').replace('\n', ' ')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--exclude-session', action='append', default=[])
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if not output.is_relative_to(PRIVATE_ROOT.resolve()) or output == PRIVATE_ROOT.resolve():
        parser.error('Output must be a NEW directory under private SmartApp/training')
    os.umask(0o077)
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    hashes, manifest = verified_snapshot(args.snapshot)
    events = load_lines(args.snapshot / 'events.jsonl')
    samples = load_lines(args.snapshot / 'navigation-samples.jsonl')
    excluded_sessions = KNOWN_TEST_SESSIONS | set(args.exclude_session)
    items, audit, arrivals, inventory, display = make_observations(events, samples, excluded_sessions)
    if len(items) < 50:
        raise ValueError('Too few eligible samples for a three-way replay')
    write_json(output / 'data-audit.json', audit)
    print(json.dumps({'stage': 'features', **audit}, ensure_ascii=False), flush=True)
    # One expensive feature comparison per causal pair, reused across the predefined grid.
    pairs = []
    feature_times = []
    for i, q in enumerate(items):
        start = time.perf_counter()
        pairs.append([(j, similarity(q, items[j])) for j in eligible_neighbors(items, i)])
        feature_times.append((time.perf_counter() - start) * 1000)
        if i and i % 300 == 0:
            print(json.dumps({'stage': 'causal_pair_comparison', 'done': i, 'total': len(items)}), flush=True)
    n = len(items)
    train_end, validation_end = int(n * .6), int(n * .8)
    validation = list(range(train_end, validation_end))
    test = list(range(validation_end, n))
    # Hyperparameter selection cannot consume validation outcomes that mature in the test period.
    mature_validation = [i for i in validation if items[i].ready_index < items[validation_end].index]
    grid = []
    for weight in (0., .35, .55, .75, 1.):
        for k in (5, 15, 30):
            for half_life in (1., 6., 24.):
                config = {'weight': weight, 'k': k, 'half_life_hours': half_life}
                metrics, _, _ = evaluate(items, pairs, mature_validation, config)
                grid.append({'config': config, 'validation': metrics})
    # Tie-break choices are declared, not selected on held-out test results.
    best = max(grid, key=lambda x: (x['validation']['top4'], x['validation']['mrr4'],
                                  -abs(x['config']['weight']-.55), -abs(x['config']['k']-15),
                                  -abs(x['config']['half_life_hours']-6)))
    configs = {
        'knn_fixed_context_055': {'weight': .55, 'k': 15, 'half_life_hours': 6.},
        'knn_validation_selected': best['config'],
        'knn_history_only': {**best['config'], 'weight': 0.},
        'knn_no_background': {**best['config'], 'no_background': True},
    }
    results, predictions, latency = {}, {}, {}
    for name in ['recency', 'markov'] + list(configs):
        m, rows, times = evaluate(items, pairs, test, configs.get(name), name if name in {'recency','markov'} else None)
        predictions[name] = rows
        strict_rows = [r for r in rows if items[r['position']].strict]
        legacy_rows = [r for r in rows if not items[r['position']].strict]
        results[name] = {'test_all_exploratory': m, 'test_strict': metric(strict_rows),
                         'test_legacy_unknown_coverage': metric(legacy_rows)}
        latency[name] = {'cached_scoring_median_ms': statistics.median(times),
                         'cached_scoring_p95_ms': sorted(times)[int(.95*(len(times)-1))]}
    chosen_rows = predictions['knn_validation_selected']
    errors, categories, bykind, bysession = [], Counter(), defaultdict(list), defaultdict(list)
    strict_categories, fixed_categories = Counter(), Counter()
    for row in predictions['knn_fixed_context_055']:
        fixed_categories[failure_reason(items[row['position']], row)] += 1
    for row in chosen_rows:
        q = items[row['position']]
        reason = failure_reason(q, row)
        categories[reason] += 1
        if q.strict:
            strict_categories[reason] += 1
        bykind[q.truth_meta['kind']].append(row)
        bysession[q.session].append(row)
        if not row['hit4']:
            errors.append({'reason': reason, 'time_utc': utc(q.timestamp), 'sample_id': q.sample_id,
                'input_coverage': q.coverage, 'strict': q.strict,
                'current': display.get(q.current, {'id': q.current}),
                'actual': q.truth_meta,
                'predicted': [q.candidates[k] for k in row['predictions']],
                'history': [display.get(k, {'id': k}) for k in q.history_keys],
                'background_entrances': q.background_count, 'source_quality': q.source_quality,
                'next_arrival_after_ms': q.offset_ms,
                'best_similarity': row['best_similarity'],
                'neighbor_label_support': row['neighbor_label_support'],
                'neighbor_median_age_hours': row['neighbor_median_age_hours'],
                'nearest_sample_ids': row['neighbor_examples']})
    overlap = {}
    for other in ['recency', 'knn_history_only', 'knn_no_background']:
        both = list(zip(chosen_rows, predictions[other]))
        overlap[other] = {'selected_only_correct': sum(a['hit4'] and not b['hit4'] for a,b in both),
                          'other_only_correct': sum(b['hit4'] and not a['hit4'] for a,b in both)}
    weight_sensitivity = []
    for w in (0., .35, .55, .75, 1.):
        candidates = [x for x in grid if x['config']['weight'] == w]
        winner = max(candidates, key=lambda x: (x['validation']['top4'], x['validation']['mrr4']))
        weight_sensitivity.append(winner)
    metrics = {'version': VERSION, 'audit': audit, 'selected_config': best['config'],
        'split': {'train_n': train_end, 'validation_n': len(validation),
                  'validation_mature_n': len(mature_validation), 'test_n': len(test),
                  'train_end_utc': utc(items[train_end-1].timestamp),
                  'validation_start_utc': utc(items[train_end].timestamp),
                  'test_start_utc': utc(items[validation_end].timestamp),
                  'test_end_utc': utc(items[-1].timestamp),
                  'test_sessions': len({items[i].session for i in test})},
        'results': results, 'failure_counts': dict(categories),
        'strict_failure_counts': dict(strict_categories), 'fixed_055_failure_counts': dict(fixed_categories),
        'high_similarity_misses': sum(e['best_similarity'] >= .9 for e in errors),
        'old_neighborhood_misses_over_6h': sum((e['neighbor_median_age_hours'] or 0) > 6 for e in errors),
        'failure_by_label_kind': {k: metric(v) for k,v in bykind.items()},
        'per_test_session': {k: metric(v) for k,v in bysession.items()},
        'paired_top4_outcomes': overlap, 'validation_weight_sensitivity': weight_sensitivity,
        'latency': latency,
        'feature_comparison_median_ms': statistics.median(feature_times[validation_end:]),
        'feature_comparison_p95_ms': sorted(feature_times[validation_end:])[int(.95*(len(test)-1))],
    }
    write_json(output / 'metrics.json', metrics)
    write_json(output / 'validation-grid.json', grid)
    write_json(output / 'errors-private.json', errors)
    with (output / 'predictions-private.jsonl').open('w') as handle:
        for name, rows in predictions.items():
            for row in rows:
                q = items[row['position']]
                handle.write(json.dumps({'model': name, 'sample_id': q.sample_id,
                    'timestamp_ms': q.timestamp, 'truth': q.truth, 'strict': q.strict, **row}, ensure_ascii=False)+'\n')
    # A replay ledger makes each cutoff, label availability and source snapshot auditable.
    with (output / 'samples-private.jsonl').open('w') as handle:
        for q in items:
            handle.write(json.dumps({'sample_id': q.sample_id, 'anchor_id': q.anchor_id,
                'index': q.index, 'ready_index': q.ready_index, 'next_event_id': q.next_event_id,
                'max_input_reference_index': q.reference_max_index, 'strict': q.strict,
                'candidate_count': len(q.candidates), 'truth': q.truth}, ensure_ascii=False)+'\n')
    # kNN's trained model IS its feature/label memory. Every record is ready by this snapshot.
    with (output / 'model-memory-private.jsonl').open('w') as handle:
        for q in items:
            handle.write(json.dumps({'sample_id': q.sample_id, 'timestamp': q.timestamp,
                'foreground': q.foreground, 'background': q.background, 'history': q.history,
                'source_quality': q.source_quality, 'label': q.truth, 'label_meta': q.truth_meta}, ensure_ascii=False)+'\n')
    with (output / 'target-counts-private.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['kind','name','locator','independent_arrivals','inventory_snapshot_mentions'])
        for kind in sorted(KINDS):
            keys = set(arrivals[kind]) | set(inventory[kind])
            for key in sorted(keys, key=lambda k: -arrivals[kind][k]):
                t = display.get(key, {})
                writer.writerow([kind,t.get('name',''),t.get('locator',''),arrivals[kind][key],inventory[kind][key]])
    code = Path(__file__)
    (output / 'algorithm-source.py').write_text(code.read_text())
    (output / 'known_inactivity.py').write_text(code.with_name('known_inactivity.py').read_text())
    provenance = {'version': VERSION, 'snapshot': str(args.snapshot.resolve()), 'input_hashes': hashes,
        'script_sha256': sha(code), 'inactivity_sha256': sha(code.with_name('known_inactivity.py')), 'script_path': str(code.resolve()),
        'excluded_sessions': sorted(excluded_sessions), 'synthetic_markers': TEST_MARKERS,
        'config_grid': {'context_weight':[0,.35,.55,.75,1], 'k':[5,15,30], 'half_life_hours':[1,6,24]},
        'selected_config':best['config'], 'split':metrics['split'],
        'semantics': 'predict first independent next arrival within complete 30s horizon; exclude same-destination next arrival; update only after horizon maturity; online test adaptation; no test-set hyperparameter selection',
        'limitations': ['Legacy coverage is unknown, not ground truth completeness.',
            'No-change 30-second windows are excluded; no waiting-time or final-intent prediction.',
            'Window instance labels do not generalize across sessions.',
            'Historical local-path existence cannot be recovered; candidate coverage is an as-of known-locator proxy.',
            'Controls not documented in validation records may remain.',
            'Raw recommendations are not calibrated probabilities.',
            'Replay hit rate is not online menu selection rate or causal usefulness.']}
    write_json(output / 'run.json', provenance)
    render_report(output, metrics, errors, arrivals, inventory, display)
    artifact_hashes = {p.name:sha(p) for p in output.iterdir() if p.is_file()}
    write_json(output / 'artifact-hashes.json', artifact_hashes)
    print(json.dumps({'stage':'complete','output':str(output),'selected':best['config'],
                      'results':results,'failure_counts':dict(categories)}, ensure_ascii=False),flush=True)


def render_report(output, m, errors, arrivals, inventory, display):
    a, split = m['audit'], m['split']
    lines = ['# 状态与行为加权近邻：历史回放实验', '',
        '这是离线研究实现；没有修改正在运行的推荐菜单。完整地址与个人案例仅保存在这个私有 run。', '',
        '## 本轮结论', '',
        f"- 初始状态权重 0.55 的近邻模型，测试 Top-4 为 {percent(m['results']['knn_fixed_context_055']['test_all_exploratory']['top4'])}；验证选参后为 {percent(m['results']['knn_validation_selected']['test_all_exploratory']['top4'])}；近期访问基线为 {percent(m['results']['recency']['test_all_exploratory']['top4'])}。本轮没有支持替换近期访问基线的证据。",
        f"- 去掉后台特征、保留其余选定参数的对照 Top-4 为 {percent(m['results']['knn_no_background']['test_all_exploratory']['top4'])}；这次背景相似度的实现未显示净收益，不能推断后台状态本身没有价值。",
        f"- 错误中有 {m['high_similarity_misses']} 次最佳近邻相似度至少 0.9：可观察状态相似不保证后续目标相同。另有 {m['old_neighborhood_misses_over_6h']} 次近邻的中位年龄超过 6 小时。",
        '- 检索先按相似度取近邻，时间衰减仅影响投票；有微弱历史票数的目标仍排在近期访问补位前。这是当前原型的明确限制，后续应在新的验证周期检验近邻新鲜度、支持度与近期访问的融合。',
        '- 验证段选中了状态权重 1.0，只代表这一小段数据的网格结果；测试不支持把它固定为正式模型权重。', '',
        '## 数据与评估口径', '',
        f"- 原始事件 {a['event_count']:,} 条，原始预测截点 {a['sample_count']:,} 个；纳入 {a['included']:,} 个。",
        f"- 原始时间：{a['range_utc'][0]} 至 {a['range_utc'][1]}。",
        f"- 纳入的完整覆盖样本 {a['strict_included']} 个；覆盖未知的旧样本 {a['legacy_exploratory_included']} 个。",
        f"- 时间顺序划分：初始学习 {split['train_n']}，验证 {split['validation_n']}（成熟可调参 {split['validation_mature_n']}），测试 {split['test_n']}。",
        f"- 测试区间：{split['test_start_utc']} 至 {split['test_end_utc']}，共 {split['test_sessions']} 个采集会话。",
        '- 预测当前独立到达之后 30 秒内的第一次独立到达；未发生到达、同目的地重复、边界中断分别排除。',
        '- 输入只读截点前的目标和状态；补全仅在当时已可用时加入输入，不充作新到达。',
        '- 每个样本完整观察 30 秒后才进入记忆；测试阶段仍按实际时间增量学习，不重新调参。',
        '- 严格子集使用同一个历史训练模型，但测试输入与后续来源覆盖均完整；不能称为全部训练数据质量都已验证。',
        '- 这些结果不包含停留预测，不代表菜单点击率、最终意图准确率或推荐的因果收益。', '',
        '## 算法', '',
        '- 状态组：前台语义占 0.4，后台来源组占 0.6；来源缺失不编码为空，旧已观察库存可信权重 0.5。',
        '- 后台按应用、普通窗口、浏览器、Finder、QSpace 分组，以加权 Jaccard 比较；保留目标/应用/URL 路径层级、选中和固定标签信息。',
        '- 行为组：最近最多 8 次已观察到达的语义、位置和近期集合；无键盘/鼠标轨迹/正文。',
        '- 状态与行为按可调权重融合；在标签属于当前候选的已成熟记忆中选 k 个近邻，以相似度平方和时间衰减投票。',
        '- 未获近邻支持的位置按近期访问补足；父子目标与同应用的上层入口去重，各模型使用相同候选与后处理。',
        '- 仅在验证段比较 45 组配置；并列按 MRR、接近初始配置的次序决定。',
        f"- 验证选定：{json.dumps(m['selected_config'],ensure_ascii=False)}。", '',
        '## 保留测试段结果', '',
        '| 模型 | n | Top-1 | Top-4 | 候选覆盖 | 候选内 Top-4 | 同应用 Top-4 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name, result in m['results'].items():
        r=result['test_all_exploratory']
        lines.append(f"| {name} | {r['n']} | {percent(r['top1'])} | {percent(r['top4'])} | {percent(r['candidate_coverage'])} | {percent(r['top4_given_covered'])} | {percent(r['app_level_top4'])} |")
    lines += ['', 'Top-4 指四个展示位置中精确命中目的地；同应用命中是较宽松的诊断，不能替代具体目的地命中。', '',
              '### 测试输入与标签均完整的子集', '', '| 模型 | n | Top-1 | Top-4 |', '|---|---:|---:|---:|']
    for name,result in m['results'].items():
        r=result['test_strict']
        lines.append(f"| {name} | {r['n']} | {percent(r['top1'])} | {percent(r['top4'])} |")
    lines += ['', '### 权重敏感性：仅验证段', '', '| 状态权重 | k | 半衰期（小时） | Top-4 |', '|---:|---:|---:|---:|']
    for row in m['validation_weight_sensitivity']:
        c=row['config'];r=row['validation']
        lines.append(f"| {c['weight']} | {c['k']} | {c['half_life_hours']} | {percent(r['top4'])} |")
    lines += ['', '### 目标类型', '', '| 后续目标类型 | n | Top-4 | 候选覆盖 |', '|---|---:|---:|---:|']
    for kind,r in m['failure_by_label_kind'].items():
        lines.append(f"| {kind} | {r['n']} | {percent(r['top4'])} | {percent(r['candidate_coverage'])} |")
    lines += ['', '## 失误分解', '', '以下针对验证选定模型，按规则互斥分类；是可观察失误特征，不是对用户真实意图或错误因果的证明。初始 0.55 模型的分解另存 metrics.json。', '']
    names={'first_seen_destination':'当时尚未见过的新目标',
           'known_but_not_in_candidate_set':'已知目标但当时不在可用候选中',
           'hierarchy_dedup_suppressed':'层级去重挤掉目标',
           'right_application_wrong_destination':'应用方向正确，但具体目的地错误',
           'rare_destination_fewer_than_3_prior_arrivals':'稀少目标：此前独立到达少于 3 次',
           'nearest_contexts_have_other_outcomes':'找到的近邻后续都是其他目标',
           'vote_ranking_error':'近邻包含正确目标，但票数排序未进入前四'}
    total_errors=sum(v for k,v in m['failure_counts'].items() if k!='hit')
    for reason,count in sorted(m['failure_counts'].items(),key=lambda kv:-kv[1]):
        if reason!='hit':
            lines.append(f"- {names[reason]}：{count} 次，占失误 {percent(count/total_errors if total_errors else None)}。")
    lines += ['', '### 私有误判案例', '']
    for reason in names:
        subset=[e for e in errors if e['reason']==reason][:3]
        if not subset:continue
        lines.append(f'#### {names[reason]}')
        lines.append('')
        for e in subset:
            lines += [f"- 时间：{e['time_utc']}；覆盖：{e['input_coverage']}；后续间隔：{e['next_arrival_after_ms']/1000:.1f} 秒。",
                      f"  - 当前：{describe_target(e['current'])}",
                      f"  - 实际：{describe_target(e['actual'])}",
                      '  - 推荐：'+'；'.join(describe_target(t) for t in e['predicted']),
                      f"  - 背景入口 {e['background_entrances']} 个；最佳近邻相似度 {e['best_similarity']:.3f}；近邻中正确目标支持数 {e['neighbor_label_support']}。"]
        lines.append('')
    lines += ['## 数据排除', '']
    lines += [f'- {key}: {value}' for key,value in a['exclusions'].items()]
    lines += ['', '## 独立到达与背景库存排行', '',
              '每个类型按原始非补全到达计数。库存次数是被快照包含的次数，既不是访问次数，也不是独立行为标签；库存不计入到达排行。', '']
    for kind in sorted(KINDS):
        lines += [f'### {kind}', '', '| 目标 | 独立到达 | 库存快照包含次数 |', '|---|---:|---:|']
        top=arrivals[kind].most_common(10)
        for key,count in top:
            lines.append(f"| {describe_target(display.get(key,{'id':key}))} | {count} | {inventory[kind][key]} |")
        if not top:lines.append('| 无已确认独立到达 | 0 | — |')
        lines.append('')
    lines += ['## 限制与复现', '',
        '- 大部分历史输入覆盖未知；少量完整覆盖测试不足以证明跨天、跨任务稳定收益。',
        '- 相邻行为有关联，测试百分比是描述性结果；没有把样本当成相互独立来给出过窄置信区间。',
        '- 已排除文档明确的验收会话与合成标记；未被标记的受控行为可能仍在数据里。',
        '- 历史本地路径是否实际存在无法恢复，候选采用当时已知合法定位符；不使用今天的文件系统倒推过去。',
        '- 原始瞬态窗口 ID 不作跨会话特征；精确窗口标签限定会话，不能保证跨会话重识别。',
        '- 第一到达可能只有应用/窗口层级，后续补全不回填成当时输入，也不声称最终意图已确认。',
        '- run.json 保存快照校验值、算法代码摘要、排除规则与参数；algorithm-source.py 固定本轮源码；predictions-private.jsonl 保存逐样本预测。',
        '- model-memory-private.jsonl 是离线近邻记忆产物，未加载到应用。',
        f"- 单次全记忆特征比较中位数 {m['feature_comparison_median_ms']:.2f} ms；这是本机研究脚本时间，非应用端到端延迟。",'']
    (output / 'report-private.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
