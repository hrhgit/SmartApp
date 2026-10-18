"""One scoring contract for selection, comparison and diagnosis.

Evaluation-only metadata must never be passed to a predictor. Historical model
code may be frozen, but new experiments import this module before frozen code.
There is no scoring-version switch: every evaluation returns accepted and exact
scores together, after the same prediction-independent label audit.
"""
from collections import Counter
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
from urllib.parse import parse_qsl, urlsplit
from uuid import UUID

SERVICE = 'service:https://chatgpt.com/'
QUERY_KEYS = {'messageId', 'src', 'historySearchQuery', 'no_universal_links'}
FALLBACK_GAIN = 0.6
FALLBACK_LOSS = 1. - FALLBACK_GAIN
# A known parent application is a useful, but deliberately incomplete, way to
# recover a concrete page or directory.  It receives exactly the same fixed
# partial utility as the already-agreed ChatGPT home fallback.  It never
# changes the exact or service-accepted hit columns.
PARENT_APP_FALLBACK_GAIN = FALLBACK_GAIN
RANK_CUTOFF = 4
RULES = {
    'primary': 'prior-supported actionable-route leaderboard; all legal rows remain a required companion board',
    'diagnostic': 'exact emitted target, always reported separately',
    'eligibility': 'audited target change; exclude current target and its app-only label',
    'topk': 'first k emitted slots; no collapse, refill or reranking during scoring',
    'missing_prediction': 'error; an explicit empty prediction is a miss',
    'membership': 'independent of model outputs, history support and rankability',
    'decision_loss': {'exact': 0., 'correct_service_home_fallback': FALLBACK_LOSS,
                      'correct_prior_known_parent_app_fallback': FALLBACK_LOSS,
                      'wrong_specific_or_wrong_service': 1., 'home_when_home_is_truth': 0.},
    'prior_supported_leaderboard': {
        'membership': 'truth had an exact action already confirmed before the cutoff, or a confirmed, launchable parent app different from the current app',
        'uses_realized_truth_only_for_posthoc_scoring_group': True,
        'never_filters_model_outputs_or_changes_exact_hits': True,
        'parent_app_gain': PARENT_APP_FALLBACK_GAIN,
    },
    'actionable_topk': {
        'gain': 'exact=1; correct ChatGPT home or eligible prior-known parent app fallback=0.6; otherwise=0',
        'parent_app_eligibility': 'same foreground outcome app, previously confirmed launchable action, and different from the current foreground app',
        'topk_aggregation': 'best single action gain among the first K emitted slots; no duplicate gain for one need',
    },
    'path_success': {
        'meaning': 'exact target or a positive-utility, prediction-time executable ancestor of the realized target',
        'input': 'known_actions on the observation contains only metadata available at its cutoff',
        'current': 'the current destination and its owning application have no fallback value',
        'rankable_coverage': 'at least one positive-utility executable action is in the model supplied rankable list',
        'legacy': 'without known_actions, only existing exact, approved service-home, and confirmed parent-app gains apply',
        'partial_utility': 'ordinary URL and directory ancestors are reported under named experimental curves; no new fixed value is assumed',
    },
    'path_utility_curves': {
        'product0.6': {'mode': 'product', 'content_retention': 0.6},
        'product0.8': {'mode': 'product', 'content_retention': 0.8},
        'saturating0.8': {'mode': 'saturating', 'content_retention': 0.8, 'floor': 0.4},
    },
    'rank_score': {'cutoff': RANK_CUTOFF, 'gain': '1 - decision_loss',
                   'aggregation': 'mean of best gain in each emitted prefix from Top1 through TopK',
                   'binary_equivalent': '(K + 1 - exact rank) / K; zero beyond K; parent fallback gets its fixed partial gain',
                   'duplicates': 'one best gain per prefix; never add homepage, parent, and child gains',
                   'missing_or_unknown': 'zero gain, retained in denominator'},
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode()).hexdigest()


@lru_cache(maxsize=1)
def contract():
    return {'rules': RULES, 'rules_sha256': digest(RULES),
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def service(target):
    """Only the already agreed ChatGPT root/conversation hierarchy is equivalent."""
    if not target or target.get('kind') != 'page' or not target.get('locator'):
        return None
    try:
        u = urlsplit(target['locator'])
        if (u.scheme.lower() != 'https' or u.hostname != 'chatgpt.com'
                or u.port not in {None, 443} or u.username is not None or u.password is not None
                or any(k not in QUERY_KEYS for k, _ in parse_qsl(u.query, keep_blank_values=True))
                or u.fragment.startswith('/')):
            return None
        if u.path in {'', '/'}:
            return SERVICE
        parts = u.path.split('/')
        if len(parts) == 3 and parts[:2] == ['', 'c'] and len(parts[2]) == 36:
            UUID(parts[2])
            return SERVICE
    except (ValueError, TypeError):
        pass
    return None


def decision_hierarchy(targets):
    """Known metadata only. Decision callers must supply their as-of catalog.

    The root action is distinct from its child outcomes. Evaluation callers may
    use outcome metadata solely to score; it must never enter a predictor.
    """
    aliases = {SERVICE: SERVICE}; entries = {SERVICE}
    for target in targets:
        parent = service(target)
        if parent:
            aliases[target['id']] = parent
            if urlsplit(target['locator']).path in {'', '/'}:
                entries.add(target['id'])
    return aliases, entries


def decision_loss(action, truth, aliases, entries):
    """One directional loss: an ancestor may partially help, a wrong child does not."""
    if action == truth:
        return 0.
    if action in entries and aliases.get(action) is not None and aliases.get(action) == aliases.get(truth):
        return 0. if truth in entries else FALLBACK_LOSS
    return 1.


def parent_action_id(metadata):
    """Return only a well-formed app action identity from evaluation metadata.

    This is deliberately not a catalog lookup and does not discover an app.  A
    runner supplies it only from the same already-observed foreground event as
    the current or later destination.
    """
    if not isinstance(metadata, dict) or metadata.get('kind') != 'app':
        return None
    bundle = metadata.get('bundleID')
    identifier = metadata.get('id')
    if not isinstance(bundle, str) or not bundle or identifier != 'app:'+bundle:
        return None
    return identifier


def _prior_evidence(row, current_parent, truth_parent):
    """Normalize the prediction-independent membership facts supplied by a runner."""
    supplied = row.get('prior_evidence') or {}
    if not isinstance(supplied, dict):
        raise ValueError('Prior-evidence metadata must be an object')
    exact = bool(supplied.get('exact_target'))
    parent = bool(supplied.get('parent_action'))
    if parent and (truth_parent is None or current_parent is None):
        raise ValueError('Parent fallback requires foreground parent identities')
    if parent and truth_parent == current_parent:
        raise ValueError('A current-app parent cannot be counted as a fallback')
    return {
        'exact_target': exact,
        'parent_action': parent,
        'supported': exact or parent,
    }


def parent_fallback_gain(row, action):
    """Partial utility for one pre-known, different parent application only."""
    scoring = row.get('_scoring') or {}
    evidence = scoring.get('prior_evidence') or {}
    truth_parent = scoring.get('truth_parent_action')
    current_parent = scoring.get('current_parent_action')
    if (evidence.get('parent_action') and action != row.get('truth_action')
            and truth_parent and truth_parent != current_parent and action == truth_parent):
        return PARENT_APP_FALLBACK_GAIN
    return 0.


def _blocked_current(row, action):
    scoring = row.get('_scoring') or {}
    if action in {scoring.get('current_action'), scoring.get('current_parent_action')}:
        return True
    hierarchy = scoring.get('path_hierarchy')
    if hierarchy is not None:
        import hierarchical_fallback_policy as path_policy
        if action in path_policy.blocked_current_actions(
            hierarchy, current_action=scoring.get('current_action'),
            current_app=scoring.get('current_parent_action'),
        ):
            return True
        current = scoring.get('current_metadata') or {}
        proposed = hierarchy.metadata.get(action) or {}
        # Two IDs can describe the same clean URL/directory location (for
        # example, a recorded page and a derived path action). A spelling
        # variant of the place already reached is not a useful fallback.
        current_web = path_policy._web_location(current)
        proposed_web = path_policy._web_location(proposed)
        if (current_web is not None and proposed_web is not None
                and not any(current_web[2:]) and not any(proposed_web[2:])
                and current_web[:2] == proposed_web[:2]):
            return True
        current_file = path_policy._file_location(current)
        proposed_file = path_policy._file_location(proposed)
        if current_file is not None and current_file == proposed_file:
            return True
    # The abstract ChatGPT home action is the current place when the actual
    # foreground target is already that home. A different conversation is not.
    current = scoring.get('current_metadata') or {}
    return bool(action == SERVICE and current.get('id') in scoring.get('entries', ())
                and service(current) == SERVICE)


def action_gain(row, action):
    """Score one emitted action without changing binary accepted/exact hits."""
    if not action or _blocked_current(row, action):
        return 0.
    hierarchy = row['_scoring'].get('path_hierarchy')
    if (hierarchy is not None and action != row['truth_action']
            and action not in hierarchy.executable
            and not (action == SERVICE and any(
                service(hierarchy.metadata[known_id]) == SERVICE
                for known_id in hierarchy.executable
                if known_id in hierarchy.metadata))):
        return 0.
    service_gain = 1. - decision_loss(action, row['truth_action'],
                                      row['_scoring']['aliases'], row['_scoring']['entries'])
    return max(service_gain, parent_fallback_gain(row, action))


def _path_policy(name):
    import hierarchical_fallback_policy as path_policy
    config = RULES['path_utility_curves'][name]
    return path_policy.FallbackPolicy(name, **config)


def path_action_gain(row, action, policy_name='product0.8'):
    """One shared scorer for the old gains and proven multilevel routes."""
    hierarchy = row['_scoring'].get('path_hierarchy')
    if hierarchy is not None and action not in hierarchy.executable:
        return 0.
    gain = action_gain(row, action)
    if not action or hierarchy is None or _blocked_current(row, action):
        return gain
    import hierarchical_fallback_policy as path_policy
    truth_id = (row.get('truth_metadata') or {}).get('id')
    if not truth_id:
        return gain
    return max(gain, path_policy.action_utility(
        hierarchy, action, truth_id, _path_policy(policy_name),
        current_action=row['_scoring'].get('current_action'),
        current_app=row['_scoring'].get('current_parent_action'),
    ))


def success_hits(row, output):
    if not eligible(row):
        raise ValueError('Excluded label must not be scored')
    predictions = output['predictions']
    return {k: any(path_action_gain(row, action) > 0 for action in predictions[:k])
            for k in (1, 2, 4)}


def minimum_risk_action(posterior, actions, aliases, entries, outcome_cost=None):
    """Minimize expected loss under a supplied predictive distribution.

    This does not estimate or calibrate probabilities. Outcomes must be mutually
    exclusive (a homepage intent may be one outcome; an aggregated service mass
    overlapping its children must not be). Retain unknown-outcome mass instead
    of renormalizing only the pages the model happens to know.
    """
    if not posterior or any(not math.isfinite(p) or p < 0 for p in posterior.values()):
        raise ValueError('Expected finite nonnegative outcome probabilities')
    if not math.isclose(sum(posterior.values()), 1., abs_tol=1e-9, rel_tol=1e-9):
        raise ValueError('Probabilities must sum to one; do not silently normalize ranking scores')
    if SERVICE in posterior:
        raise ValueError('Aggregated service probability is not a disjoint outcome; use homepage or unknown-child outcomes')
    weights = outcome_cost if outcome_cost is not None else {y: 1. for y in posterior}
    if any(y not in weights or not math.isfinite(weights[y]) or weights[y] <= 0 for y in posterior):
        raise ValueError('Every outcome needs a finite positive cost weight')
    actions = list(dict.fromkeys(actions))
    if not actions:
        raise ValueError('No eligible action')
    risks = {a: math.fsum(p*weights[y]*decision_loss(a, y, aliases, entries) for y, p in posterior.items())
             for a in actions}
    # Equal risk prefers a precise action, then the caller's stable order.
    selected = min(actions, key=lambda a: (risks[a], a in entries, actions.index(a)))
    return {'action': selected, 'expected_loss': risks[selected], 'action_expected_losses': risks,
            'loss_definition': RULES['decision_loss'], 'outcome_weighted': outcome_cost is not None,
            'probability_calibration': 'caller_responsibility'}


def label_audit(row, observation):
    """Audit the label being scored, not an inferred intent or model prediction.

    A real window transition collapsed to the current app is also unscoreable
    under that coarse label. Preserve it for label repair, not as a model miss.
    Existing dataset construction supplies stable-arrival and admission evidence;
    this audit checks identity, chronology and actionable label granularity.
    """
    current, target = observation.get('target') or {}, row.get('truth_metadata') or {}
    label = row.get('truth_action')
    reason = None
    if not current.get('id') or not label or not target.get('id'):
        reason = 'missing_identity_evidence'
    elif current.get('kind') not in {'app', 'window', 'page', 'directory', 'file'} or target.get('kind') not in {'app', 'window', 'page', 'directory', 'file'}:
        reason = 'unsupported_target_kind'
    elif observation.get('ready_index', -1) <= observation.get('index', -1):
        reason = 'invalid_outcome_chronology'
    elif label == current['id']:
        reason = 'current_destination'
    elif current.get('bundleID') and label in {'app:'+current['bundleID'], 'window_class:'+current['bundleID']}:
        reason = 'same_application_coarse_label'
    elif label != target['id'] and not (target.get('kind') == 'window' and
                                      label == 'app:'+target.get('bundleID', '')):
        reason = 'label_identity_mismatch'
    return {'eligible': reason is None, 'reason': reason,
            'distinct_window_lost_in_app_label': bool(reason == 'same_application_coarse_label'
                and current.get('kind') == target.get('kind') == 'window'
                and current.get('id') != target.get('id'))}


def _path_hierarchy(observation, row):
    """Build score-only routes from the caller's cutoff catalog, never a future catalog."""
    known = observation.get('known_actions')
    if known is None:
        return None, None
    if not isinstance(known, list) or any(not isinstance(item, dict)
                                             or not isinstance(item.get('id'), str)
                                             or not item['id'] for item in known):
        raise ValueError('known_actions must be a list of as-of action metadata')
    items = {item['id']: item for item in known}
    if len(items) != len(known):
        raise ValueError('Duplicate known action identity at a prediction cutoff')
    current = observation.get('target') or {}
    if isinstance(current.get('id'), str) and current['id'] not in items:
        items[current['id']] = current
    import hierarchical_fallback_policy as path_policy
    base = path_policy.build_hierarchy(items.values(), executable=items)
    expanded = path_policy.add_outcome(base, row.get('truth_metadata') or {})
    return expanded, digest([sorted(items.values(), key=lambda item: item['id']),
                             sorted(base.executable)])


def prepare_evaluation(data, output=None):
    """Return audited copies; never edit the frozen dataset or model inputs."""
    observations = {r['position']: r for r in data['observations']}
    targets = [o['target'] for o in data['observations']]
    targets += [r['truth_metadata'] for r in data['evaluation']]
    for stream in data.get('streams', {}).values():
        targets += [r['record']['target'] for r in stream]
    aliases, entries = decision_hierarchy(targets)
    aliases_sha256 = digest([aliases, sorted(entries)])
    rows = []
    for row in data['evaluation']:
        if row['position'] not in observations:
            raise ValueError('Evaluation input observation missing')
        observation = observations[row['position']]
        audit = label_audit(row, observation)
        current_parent = parent_action_id(observation.get('parent_action'))
        truth_parent = parent_action_id(row.get('truth_parent_metadata'))
        prior_evidence = _prior_evidence(row, current_parent, truth_parent)
        path_hierarchy, path_catalog_sha256 = _path_hierarchy(observation, row)
        rows.append({**row, '_scoring': {**audit, 'contract': digest(RULES),
                                       'current_parent_action': current_parent,
                                       'current_action': (observation.get('target') or {}).get('id'),
                                       'current_metadata': observation.get('target') or {},
                                       'truth_parent_action': truth_parent,
                                       'prior_evidence': prior_evidence,
                                       'path_hierarchy': path_hierarchy,
                                       'path_catalog_sha256': path_catalog_sha256,
                                       'row_sha256': digest([row['position'], row['sample_id'],
                                           row['truth_action'], row['truth_metadata'],
                                           observation['target'], audit, current_parent,
                                           truth_parent, prior_evidence, aliases_sha256,
                                           path_catalog_sha256]),
                                       'aliases': aliases, 'entries': entries}})
    if len({r['position'] for r in rows}) != len(rows):
        raise ValueError('Duplicate evaluation cutoff')
    if output is not None:
        output = Path(output)
        payload = audit_summary(rows)
        payload['contract'] = contract()
        payload['membership_sha256'] = digest([
            [r['position'], r['sample_id'], r['truth_action'], r['truth_metadata'],
             observations[r['position']]['target'], r['_scoring']['eligible'], r['_scoring']['reason'],
             r['_scoring']['current_parent_action'], r['_scoring']['truth_parent_action'],
             r['_scoring']['prior_evidence'], r['_scoring']['path_catalog_sha256']]
            for r in rows])
        payload['rows'] = [
            {'position': r['position'], 'sample_id': r['sample_id'],
             **{k: v for k, v in r['_scoring'].items()
                if k not in {'aliases', 'entries', 'contract', 'current_parent_action',
                             'current_metadata', 'path_hierarchy', 'truth_parent_action'}}}
            for r in rows
        ]
        path = output/'evaluation-audit-private.json'
        if path.exists():
            raise ValueError('Scoring audit is immutable')
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2)+'\n')
        path.chmod(0o600)
        source = output/'evaluation_contract.py'
        if source.resolve() != Path(__file__).resolve():
            source.write_bytes(Path(__file__).read_bytes())
            source.chmod(0o600)
    return rows


def eligible(row):
    audit = row.get('_scoring')
    if not audit or audit.get('contract') != digest(RULES):
        raise ValueError('Run prepare_evaluation before scoring; no legacy fallback')
    return audit['eligible']


def audit_summary(rows):
    valid = [r for r in rows if eligible(r)]
    prior = [r['_scoring'].get('prior_evidence', {}) for r in valid]
    return {'candidate_n': len(rows), 'eligible_n': len(valid), 'excluded_n': len(rows)-len(valid),
            'exclusion_reasons': dict(Counter(r['_scoring']['reason'] for r in rows if not eligible(r))),
            'prior_exact_target_n': sum(bool(value.get('exact_target')) for value in prior),
            'prior_parent_action_n': sum(bool(value.get('parent_action')) for value in prior),
            'prior_supported_n': sum(bool(value.get('supported')) for value in prior)}


def prediction_at(outputs, position):
    try:
        return outputs[position] if position in outputs else outputs[str(position)]
    except KeyError as e:
        raise ValueError('Missing prediction for evaluation cutoff '+str(position)) from e


def hits(row, output):
    if not eligible(row):
        raise ValueError('Excluded label must not be scored')
    predicted = output['predictions']
    aliases = row['_scoring']['aliases']
    truth = row['truth_action']
    accepted_truth = aliases.get(truth, truth)
    accepted = [aliases.get(p, p) for p in predicted]
    return {'accepted': {k: accepted_truth in accepted[:k] for k in (1, 2, 4)},
            'exact': {k: truth in predicted[:k] for k in (1, 2, 4)}}


def observed_decision_loss(row, output):
    if not eligible(row): raise ValueError('Excluded label must not be scored')
    return 1. - action_gain(row, output['predictions'][0]) if output['predictions'] else 1.


def ranked_gain(row, output):
    """Average graded Top1..TopK utility, using actual emitted positions.

    For one exact target at rank r this equals 1-(r-1)/K, truncated to zero
    beyond K. A homepage can partly satisfy a prefix; a later exact target can
    improve subsequent prefixes. Each prefix counts the same need only once.
    This measures menu opportunity, not observed selection effort or time saved.
    """
    if not eligible(row):
        raise ValueError('Excluded label must not be scored')
    emitted = output['predictions'][:RANK_CUTOFF]
    best = 0.; prefixes = []; parent_improved = []
    for i in range(RANK_CUTOFF):
        before = best
        parent_gain = 0.
        if i < len(emitted):
            parent_gain = parent_fallback_gain(row, emitted[i])
            best = max(best, action_gain(row, emitted[i]))
        prefixes.append(best)
        parent_improved.append(parent_gain > before + 1e-12)
    return {'score': math.fsum(prefixes)/RANK_CUTOFF, 'prefix_gains': prefixes,
            'parent_fallback_improved': parent_improved}


def measure_weighted_loss(rows, outputs, row_weights):
    """One common, externally supplied weight per eligible row, for every policy.

    Use identical pre-cutoff frequency statistics across policies. This secondary
    preference analysis never changes primary membership or TopK scoring.
    """
    valid = [r for r in rows if eligible(r)]; total = mass = 0.
    for r in valid:
        w = row_weights[r['position']]
        if not math.isfinite(w) or w <= 0: raise ValueError('Invalid evaluation weight')
        total += w*observed_decision_loss(r, prediction_at(outputs, r['position'])); mass += w
    return {'n': len(valid), 'weighted_loss_total': total, 'weight_sum': mass,
            'mean': total/mass if mass else None,
            'comparison_key': digest([contract(), sorted((r['_scoring']['row_sha256'], row_weights[r['position']]) for r in valid)])}


def measure(rows, outputs):
    """Primary fields are accepted hits; exact is a mandatory sibling diagnostic."""
    valid = [r for r in rows if eligible(r)]
    counts = {level: Counter() for level in ('accepted', 'exact')}
    success_counts = Counter()
    path_totals = {name: Counter() for name in RULES['path_utility_curves']}
    success_rankable = 0
    covered = Counter(); reciprocal = Counter(); returned = 0
    actual_loss = 0.; loss_counts = Counter(); rank_total = 0.; prefix_totals = Counter()
    actionable_totals = Counter(); actionable_helpful = Counter(); parent_fallback_used = Counter()
    for r in valid:
        output = prediction_at(outputs, r['position'])
        h = hits(r, output); returned += bool(output['predictions'])
        successful = success_hits(r, output)
        success_counts.update({k: int(value) for k, value in successful.items()})
        success_rankable += any(path_action_gain(r, action) > 0
                                for action in output.get('rankable', output['predictions']))
        emitted = output['predictions'][:RANK_CUTOFF]
        for name, totals in path_totals.items():
            best = 0.; prefixes = []
            for index in range(RANK_CUTOFF):
                if index < len(emitted):
                    best = max(best, path_action_gain(r, emitted[index], name))
                prefixes.append(best)
            totals.update({k: prefixes[k - 1] for k in (1, 2, 4)})
            totals['rank_score'] += math.fsum(prefixes) / RANK_CUTOFF
        loss = observed_decision_loss(r, output)
        actual_loss += loss
        ranked = ranked_gain(r, output); rank_total += ranked['score']
        prefix_totals.update({k: gain for k, gain in enumerate(ranked['prefix_gains'], 1)})
        for k, gain in enumerate(ranked['prefix_gains'], 1):
            actionable_totals[k] += gain
            actionable_helpful[k] += int(gain > 0.)
            parent_fallback_used[k] += int(any(ranked['parent_fallback_improved'][:k]))
        loss_name = 'zero_cost' if math.isclose(loss, 0.) else (
            'known_parent_or_home_fallback' if math.isclose(loss, FALLBACK_LOSS) else 'wrong_or_empty')
        loss_counts[loss_name] += 1
        for level in counts:
            counts[level].update({k: int(v) for k, v in h[level].items()})
            alias = r['_scoring']['aliases'] if level == 'accepted' else {}
            truth = alias.get(r['truth_action'], r['truth_action'])
            candidates = [alias.get(p, p) for p in output.get('rankable', output['predictions'])]
            covered[level] += truth in candidates
            p = [alias.get(p, p) for p in output['predictions'][:4]]
            reciprocal[level] += 1/(p.index(truth)+1) if truth in p else 0
    n = len(valid)
    def result(level):
        c = counts[level]
        return {'n': n, **{f'top{k}_hits': c[k] for k in (1, 2, 4)},
                **{f'top{k}': c[k]/n if n else None for k in (1, 2, 4)},
                'returned_n': returned, 'prediction_coverage': returned/n if n else None,
                'rankable_truth_n': covered[level], 'rankable_truth_coverage': covered[level]/n if n else None,
                'mrr4': reciprocal[level]/n if n else None}
    return {**result('accepted'), 'exact': result('exact'),
            'success': {'n': n,
                        **{f'top{k}_hits': success_counts[k] for k in (1, 2, 4)},
                        **{f'top{k}': success_counts[k]/n if n else None for k in (1, 2, 4)},
                        'rankable_useful_action_n': success_rankable,
                        'rankable_useful_action_coverage': success_rankable/n if n else None,
                        'definition': RULES['path_success']},
            'path_utility': {name: {'n': n,
                                    **{f'top{k}_gain': totals[k]/n if n else None
                                       for k in (1, 2, 4)},
                                    'rank_score': totals['rank_score']/n if n else None,
                                    'policy': RULES['path_utility_curves'][name]}
                             for name, totals in path_totals.items()},
            'audit': audit_summary(rows),
            'actionable_topk': {
                'n': n,
                **{f'top{k}_gain_total': actionable_totals[k] for k in (1, 2, 4)},
                **{f'top{k}_gain': actionable_totals[k]/n if n else None for k in (1, 2, 4)},
                **{f'top{k}_helpful_n': actionable_helpful[k] for k in (1, 2, 4)},
                **{f'top{k}_helpful': actionable_helpful[k]/n if n else None for k in (1, 2, 4)},
                **{f'top{k}_parent_fallback_used_n': parent_fallback_used[k] for k in (1, 2, 4)},
                **{f'top{k}_parent_fallback_used': parent_fallback_used[k]/n if n else None for k in (1, 2, 4)},
                'definition': RULES['actionable_topk'],
            },
            'rank_score': {'n': n, 'k': RANK_CUTOFF, 'total': rank_total,
                           'mean': rank_total/n if n else None,
                           'mean_loss': 1.-rank_total/n if n else None,
                           'prefix_mean_gains': {str(k): prefix_totals[k]/n if n else None
                                                 for k in range(1, RANK_CUTOFF+1)},
                           'definition': RULES['rank_score']},
            'decision_loss': {'n': n, 'total': actual_loss, 'mean': actual_loss/n if n else None,
                              'outcomes': dict(loss_counts), 'definition': RULES['decision_loss'],
                              'basis': 'observed first emitted action; not predicted posterior risk'},
            'comparison_key': digest([contract(), sorted(r['_scoring']['row_sha256'] for r in rows)])}


def require_comparable(*scores):
    keys = {m.get('comparison_key') for m in scores}
    if None in keys or len(keys) != 1:
        raise ValueError('Re-score saved predictions together; do not compare different scoring contracts or samples')


def paired_topk(rows, before, after):
    counts = {k: Counter() for k in (1, 2, 4)}
    success = {k: Counter() for k in (1, 2, 4)}
    n = 0
    for r in rows:
        if not eligible(r):
            continue
        n += 1
        a = hits(r, prediction_at(before, r['position']))['accepted']
        b = hits(r, prediction_at(after, r['position']))['accepted']
        sa = success_hits(r, prediction_at(before, r['position']))
        sb = success_hits(r, prediction_at(after, r['position']))
        for k in counts:
            counts[k]['helped'] += b[k] and not a[k]
            counts[k]['hurt'] += a[k] and not b[k]
            counts[k]['reference_hits'] += a[k]
            counts[k]['alternative_hits'] += b[k]
            counts[k]['both_correct'] += a[k] and b[k]
            counts[k]['both_wrong'] += not a[k] and not b[k]
            success[k]['helped'] += sb[k] and not sa[k]
            success[k]['hurt'] += sa[k] and not sb[k]
            success[k]['reference_hits'] += sa[k]
            success[k]['alternative_hits'] += sb[k]
    return {'n': n, **{f'top{k}': {**v, 'net': v['helped']-v['hurt'],
        'net_hits': v['helped']-v['hurt'],
        'delta_percentage_points': 100*(v['helped']-v['hurt'])/n if n else None}
        for k, v in counts.items()},
        'success': {f'top{k}': {**v, 'net': v['helped']-v['hurt'],
                               'delta_percentage_points': 100*(v['helped']-v['hurt'])/n if n else None}
                    for k, v in success.items()}}


def groups(rows):
    result = {'all': rows, 'original': [r for r in rows if r['evidence_grade'] == 'original'],
              'expanded': [r for r in rows if r['evidence_grade'] != 'unsupported']}
    for grade in ('added_exact', 'semantic_strong', 'semantic_weak', 'unsupported'):
        result[grade] = [r for r in rows if r['evidence_grade'] == grade]
    result['late_original'] = [r for r in result['original'] if r['split'] == 'test']
    result['new_exact_history'] = [r for r in rows if r.get('new_exact_history')]
    return result


LEADERBOARD_ALL = 'all_legal'
LEADERBOARD_PRIOR = 'prior_supported'
PRIMARY_LEADERBOARD = LEADERBOARD_PRIOR


def leaderboard_groups(rows):
    """Two fixed, prediction-independent scoreboards for the same legal data."""
    legal = [row for row in rows if eligible(row)]
    prior = [row for row in legal if row['_scoring'].get('prior_evidence', {}).get('supported')]
    return {LEADERBOARD_ALL: legal, LEADERBOARD_PRIOR: prior}


def leaderboard_audit(rows):
    legal = [row for row in rows if eligible(row)]
    exact = sum(bool(row['_scoring'].get('prior_evidence', {}).get('exact_target')) for row in legal)
    parent = sum(bool(row['_scoring'].get('prior_evidence', {}).get('parent_action')) for row in legal)
    supported = sum(bool(row['_scoring'].get('prior_evidence', {}).get('supported')) for row in legal)
    return {
        'legal_n': len(legal),
        'prior_exact_target_n': exact,
        'prior_parent_action_n': parent,
        'prior_supported_n': supported,
        'prior_unsupported_n': len(legal) - supported,
        'membership_definition': RULES['prior_supported_leaderboard'],
    }


def score_leaderboards(rows, outputs_by_model):
    """Score every model on both leaderboards with one common contract."""
    result = {}
    for name, part in leaderboard_groups(rows).items():
        board = {model: measure(part, output) for model, output in outputs_by_model.items()}
        if board:
            require_comparable(*board.values())
        result[name] = board
    return result


def primary_leaderboard_metric(leaderboards, model_name):
    """Use the prior-supported board when it exists; retain a safe empty fallback."""
    primary = leaderboards.get(PRIMARY_LEADERBOARD, {}).get(model_name)
    if primary is not None and primary.get('n', 0):
        return PRIMARY_LEADERBOARD, primary
    return LEADERBOARD_ALL, leaderboards[LEADERBOARD_ALL][model_name]


def grouped_metrics(rows, outputs):
    return {name: measure(part, outputs) for name, part in groups(rows).items()}


def cohorts(rows):
    return {
        'evidence_total': [r for r in rows if r['evidence_grade'] == 'original' or r.get('new_exact_history')],
        'expanded_total': [r for r in rows if r['evidence_grade'] != 'unsupported'],
        'old_evidence': [r for r in rows if r['evidence_grade'] == 'original'],
        'late_old_evidence': [r for r in rows if r['split'] == 'test' and r['evidence_grade'] == 'original'],
        'new_evidence': [r for r in rows if r['split'] == 'new_test' and r.get('new_exact_history')],
        'old_all': [r for r in rows if r['split'] != 'new_test'],
        'new_all': [r for r in rows if r['split'] == 'new_test'], 'all': rows,
    }


def cohort_sizes(parts):
    return {name: sum(eligible(r) for r in rows) for name, rows in parts.items()}


def report_scope(rows):
    parts = cohorts(rows); sizes = cohort_sizes(parts); audit = audit_summary(rows)
    return (f"合法有证据样本合计 {sizes['evidence_total']} 条（旧 {sizes['old_evidence']}＋新增 {sizes['new_evidence']}），"
            f"占合法评测 {sizes['all']} 条的 {sizes['evidence_total']/sizes['all']:.1%}。"
            f"原始评测 {audit['candidate_n']} 条，合法性审查排除 {audit['excluded_n']} 条。"
            '主分按已约定服务归并接受规则，精确目标成绩另列；分母调整不代表模型进步。')
