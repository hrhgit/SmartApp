"""Causal, source-only cohorts for current concrete-page transition diagnosis.

This module deliberately does not define a new hit or loss function.  Callers
must score every returned cohort through ``evaluation_contract.measure``.  It
only answers a different membership question: at a prediction cutoff, has the
current *specific page* already been independently observed, and has it already
produced a mature next-step action?

The primary page-source cohorts are prospectively identifiable: they never read
the realized next target.  A page-to-page cohort is provided separately as an
after-outcome target-kind diagnostic.
"""

from collections import Counter, defaultdict


VERSION = 'current-page-source-history-v1'


def _observations(data):
    values = data.get('observations') or []
    by_position = {value.get('position'): value for value in values}
    if len(by_position) != len(values) or None in by_position:
        raise ValueError('Observations need unique positions')
    for value in values:
        target = value.get('target') or {}
        if not isinstance(value.get('index'), int) or not value.get('origin') or not target.get('id'):
            raise ValueError('Observation lacks chronological source identity')
    return by_position


def _rows_by_index(rows, observations):
    result = defaultdict(list)
    seen = set()
    for row in rows:
        position = row.get('position')
        if position in seen or position not in observations:
            raise ValueError('Diagnostic row has duplicate or missing observation')
        seen.add(position)
        result[observations[position]['index']].append(row)
    return result


def source_history(data, rows, training):
    """Return causal current-page history flags keyed by evaluation position.

    ``training`` must already be the common first-head, navigation-valid stream
    used by the model comparison.  Its mature actions are admitted only when
    ``ready_index < cutoff``.  Repeated refinements of one origin/outcome are
    deduplicated, so an arrival cannot create extra source-outgoing support.
    """
    observations = _observations(data)
    rows_by_index = _rows_by_index(rows, observations)
    history = {}

    prior_arrivals = Counter()
    all_observations = sorted(observations.values(), key=lambda value: (value['index'], value['position']))
    cursor = 0
    while cursor < len(all_observations):
        index = all_observations[cursor]['index']
        end = cursor
        while end < len(all_observations) and all_observations[end]['index'] == index:
            end += 1
        # A current observation at this index must not count as a prior visit to
        # itself or another simultaneous current observation.
        for row in rows_by_index.get(index, []):
            observation = observations[row['position']]
            target = observation['target']
            is_exact_page = target.get('kind') == 'page' and observation['origin'] == target['id']
            history[row['position']] = {
                'current_exact_page': is_exact_page,
                'prior_current_page_arrivals': prior_arrivals[observation['origin']] if is_exact_page else 0,
            }
        for observation in all_observations[cursor:end]:
            prior_arrivals[observation['origin']] += 1
        cursor = end

    if len(history) != len(rows):
        raise ValueError('Could not assign a pre-cutoff source history to every row')

    prepared = []
    for item in training:
        required = {'position', 'ready_index', 'head', 'label', 'outcome_id'}
        if not required <= set(item):
            raise ValueError('Training item lacks canonical transition fields')
        if item['head'] != 0:
            raise ValueError('Page-source diagnosis requires first retained next steps only')
        if item['position'] not in observations or not isinstance(item['ready_index'], int):
            raise ValueError('Training item lacks an observed source or maturity index')
        source = observations[item['position']]
        if source['index'] > item['ready_index']:
            raise ValueError('A transition matures before its source observation')
        prepared.append(item)

    outgoing = defaultdict(Counter)
    assigned = {}
    transition_cursor = 0
    prepared.sort(key=lambda item: (item['ready_index'], item['position'], item['outcome_id'], item['label']))
    for row in sorted(rows, key=lambda value: (observations[value['position']]['index'], value['position'])):
        observation = observations[row['position']]
        cutoff = observation['index']
        while transition_cursor < len(prepared) and prepared[transition_cursor]['ready_index'] < cutoff:
            item = prepared[transition_cursor]
            source = observations[item['position']]['origin']
            key = (source, item['outcome_id'])
            old_label = assigned.get(key)
            if old_label is not None and old_label != item['label']:
                raise ValueError('One independent source outcome has conflicting labels')
            if old_label is None:
                assigned[key] = item['label']
                outgoing[source][item['label']] += 1
            transition_cursor += 1

        detail = history[row['position']]
        if detail['current_exact_page']:
            counts = outgoing[observation['origin']]
            total = sum(counts.values())
            top = max(counts.values(), default=0)
            detail.update({
                'mature_outgoing_count': total,
                'mature_outgoing_destination_count': len(counts),
                'mature_outgoing_top_share': top / total if total else None,
            })
        else:
            detail.update({'mature_outgoing_count': 0, 'mature_outgoing_destination_count': 0,
                           'mature_outgoing_top_share': None})
        detail['seen_current_page_before'] = bool(detail['prior_current_page_arrivals'])
        detail['has_mature_current_page_outgoing'] = bool(detail['mature_outgoing_count'])

    return history


def cohorts(rows, history):
    """Return source-defined page cohorts plus clearly labeled target-kind views."""
    if {row['position'] for row in rows} != set(history):
        raise ValueError('Page-source history must cover exactly the scored rows')

    def selected(predicate):
        return [row for row in rows if predicate(history[row['position']])]

    page = lambda detail: detail['current_exact_page']
    seen = lambda detail: page(detail) and detail['seen_current_page_before']
    outgoing = lambda detail: seen(detail) and detail['has_mature_current_page_outgoing']
    result = {
        'current_exact_page': selected(page),
        'current_exact_page_seen_before': selected(seen),
        'current_exact_page_unseen_before': selected(lambda detail: page(detail) and not detail['seen_current_page_before']),
        'seen_current_page_with_mature_outgoing': selected(outgoing),
        'seen_current_page_without_mature_outgoing': selected(
            lambda detail: seen(detail) and not detail['has_mature_current_page_outgoing']),
        'seen_current_page_outgoing_1': selected(lambda detail: outgoing(detail) and detail['mature_outgoing_count'] == 1),
        'seen_current_page_outgoing_2_3': selected(lambda detail: outgoing(detail) and 2 <= detail['mature_outgoing_count'] <= 3),
        'seen_current_page_outgoing_4_plus': selected(lambda detail: outgoing(detail) and detail['mature_outgoing_count'] >= 4),
    }
    # The next-target kind is only known after the event.  Keep this out of the
    # source-defined primary cohorts, but make it available for diagnosing the
    # harder concrete page-to-page case the user sees in product behavior.
    result['seen_current_page_to_page'] = [
        row for row in result['current_exact_page_seen_before']
        if (row.get('truth_metadata') or {}).get('kind') == 'page'
    ]
    return result


def fixed_split_page_history(data, rows, boundary_index):
    """Classify post-split source pages using only arrivals before one fixed cut.

    This is deliberately different from :func:`source_history`: membership is
    frozen at the common validation boundary, rather than growing as the model
    learns from earlier rows in the validation tail.  A page is "seen" only
    when its exact page identity was an independently confirmed page arrival
    strictly before that boundary.  It never consults the realized next target
    or any post-boundary observation.
    """
    if not isinstance(boundary_index, int):
        raise ValueError('Fixed split boundary must be an integer event index')
    observations = _observations(data)
    _rows_by_index(rows, observations)

    pre_split_arrivals = Counter()
    for observation in observations.values():
        target = observation['target']
        if observation['index'] < boundary_index and target.get('kind') == 'page':
            pre_split_arrivals[target['id']] += 1

    result = {}
    for row in rows:
        observation = observations[row['position']]
        if observation['index'] < boundary_index:
            raise ValueError('A fixed-split page diagnostic row precedes the validation boundary')
        target = observation['target']
        is_exact_page = target.get('kind') == 'page' and observation['origin'] == target['id']
        count = pre_split_arrivals[observation['origin']] if is_exact_page else 0
        result[row['position']] = {
            'current_exact_page': is_exact_page,
            'source_page_id': observation['origin'] if is_exact_page else None,
            'pre_split_current_page_arrivals': count,
            'seen_current_page_before_split': bool(count),
        }
    return result


def fixed_split_page_cohorts(rows, history):
    """Group a common post-split validation tail by pre-split page exposure.

    The first four groups are all source-defined before prediction.  The final
    page-to-page group is intentionally marked as an after-outcome diagnostic.
    """
    if {row['position'] for row in rows} != set(history):
        raise ValueError('Fixed-split page history must cover exactly the scored rows')

    def selected(predicate):
        return [row for row in rows if predicate(history[row['position']])]

    page = lambda detail: detail['current_exact_page']
    seen = lambda detail: page(detail) and detail['seen_current_page_before_split']
    result = {
        'all_complete_validation': list(rows),
        'current_exact_page': selected(page),
        'current_exact_page_seen_before_split': selected(seen),
        'current_exact_page_not_seen_before_split': selected(
            lambda detail: page(detail) and not detail['seen_current_page_before_split']),
        'seen_current_page_arrivals_1': selected(
            lambda detail: seen(detail) and detail['pre_split_current_page_arrivals'] == 1),
        'seen_current_page_arrivals_2_3': selected(
            lambda detail: seen(detail) and 2 <= detail['pre_split_current_page_arrivals'] <= 3),
        'seen_current_page_arrivals_4_plus': selected(
            lambda detail: seen(detail) and detail['pre_split_current_page_arrivals'] >= 4),
    }
    result['seen_current_page_to_page'] = [
        row for row in result['current_exact_page_seen_before_split']
        if (row.get('truth_metadata') or {}).get('kind') == 'page'
    ]
    return result


def source_page_rows(rows, history):
    """Return source-page subgroups for the pre-split-seen page cohort."""
    result = defaultdict(list)
    for row in rows:
        detail = history[row['position']]
        if detail['current_exact_page'] and detail['seen_current_page_before_split']:
            result[detail['source_page_id']].append(row)
    return {key: result[key] for key in sorted(result)}


UNIT_KINDS = frozenset({'app', 'window', 'page', 'directory', 'file'})


def fixed_split_unit_history(data, rows, boundary_index):
    """Classify every current model unit using only a common pre-split history.

    A unit is the observation's canonical ``origin`` -- the same identity the
    current-only predictor receives.  This intentionally keeps application,
    window-class, page, directory, and file units in one population.  In
    particular, a window is represented by its stable window class when that
    is the model input, rather than by an ephemeral raw window instance.
    """
    if not isinstance(boundary_index, int):
        raise ValueError('Fixed split boundary must be an integer event index')
    observations = _observations(data)
    _rows_by_index(rows, observations)

    pre_split_arrivals = Counter()
    for observation in observations.values():
        if observation['index'] < boundary_index and observation['target'].get('kind') in UNIT_KINDS:
            pre_split_arrivals[observation['origin']] += 1

    result = {}
    for row in rows:
        observation = observations[row['position']]
        if observation['index'] < boundary_index:
            raise ValueError('A fixed-split unit diagnostic row precedes the validation boundary')
        kind = observation['target'].get('kind')
        is_unit = kind in UNIT_KINDS
        count = pre_split_arrivals[observation['origin']] if is_unit else 0
        result[row['position']] = {
            'current_unit': is_unit,
            'current_unit_id': observation['origin'] if is_unit else None,
            'current_unit_kind': kind if is_unit else None,
            'pre_split_current_unit_arrivals': count,
            'seen_current_unit_before_split': bool(count),
        }
    return result


def fixed_split_unit_cohorts(rows, history):
    """Group one complete validation tail by pre-split current-unit exposure."""
    if {row['position'] for row in rows} != set(history):
        raise ValueError('Fixed-split unit history must cover exactly the scored rows')

    def selected(predicate):
        return [row for row in rows if predicate(history[row['position']])]

    unit = lambda detail: detail['current_unit']
    seen = lambda detail: unit(detail) and detail['seen_current_unit_before_split']
    result = {
        'all_complete_validation': list(rows),
        'current_unit': selected(unit),
        'current_unit_seen_before_split': selected(seen),
        'current_unit_not_seen_before_split': selected(
            lambda detail: unit(detail) and not detail['seen_current_unit_before_split']),
        'seen_current_unit_arrivals_1': selected(
            lambda detail: seen(detail) and detail['pre_split_current_unit_arrivals'] == 1),
        'seen_current_unit_arrivals_2_3': selected(
            lambda detail: seen(detail) and 2 <= detail['pre_split_current_unit_arrivals'] <= 3),
        'seen_current_unit_arrivals_4_plus': selected(
            lambda detail: seen(detail) and detail['pre_split_current_unit_arrivals'] >= 4),
    }
    for kind in sorted(UNIT_KINDS):
        result['seen_current_unit_kind_'+kind] = selected(
            lambda detail, kind=kind: seen(detail) and detail['current_unit_kind'] == kind)
    result['seen_current_unit_to_page'] = [
        row for row in result['current_unit_seen_before_split']
        if (row.get('truth_metadata') or {}).get('kind') == 'page'
    ]
    return result


def source_unit_rows(rows, history):
    """Return source-unit subgroups for the pre-split-seen unit cohort."""
    result = defaultdict(list)
    for row in rows:
        detail = history[row['position']]
        if detail['current_unit'] and detail['seen_current_unit_before_split']:
            result[detail['current_unit_id']].append(row)
    return {key: result[key] for key in sorted(result)}


def fixed_test_cohorts(rows):
    """Group the frozen ordered-current test rows without changing membership.

    The historical ordered-current audit stores an exact current-page identity
    and the mature, deduplicated outgoing actions from that same current
    context in ``pre_prediction.historical_outcomes``.  A nonempty list is a
    conservative, prediction-time proof that this concrete page was already
    observed as a source and has a learned next-step distribution.  It is more
    useful than treating a catalog or background-tab appearance as a visit.
    """
    required = {'sample_id', 'display_only', 'pre_prediction', 'observed_outcome',
                'diagnosis_after_outcome', 'evaluation'}
    if len({row.get('sample_id') for row in rows}) != len(rows):
        raise ValueError('Frozen test rows need unique sample IDs')
    if any(not required <= set(row) for row in rows):
        raise ValueError('Frozen test row lacks page-transition fields')

    def page(row):
        display_page = row['display_only'].get('current_identity_level') == 'page'
        current = row['pre_prediction'].get('current_key')
        if display_page and (not isinstance(current, str) or not current.startswith('page:')):
            raise ValueError('A page diagnostic row lacks an exact current-page identity')
        return display_page

    def outgoing(row):
        values = row['pre_prediction'].get('historical_outcomes')
        if not isinstance(values, list):
            raise ValueError('Frozen test row lacks its mature outgoing history')
        if page(row):
            current = row['pre_prediction']['current_key']
            if any(witness.get('context') != [current] for witness in values):
                raise ValueError('Page source history is not tied to the exact current page')
        return len(values)

    source_seen = lambda row: page(row) and outgoing(row) > 0
    result = {
        'all_fixed_test': list(rows),
        'current_page': [row for row in rows if page(row)],
        'current_page_with_mature_outgoing': [row for row in rows if source_seen(row)],
        'current_page_without_mature_outgoing': [row for row in rows if page(row) and not source_seen(row)],
        'current_page_outgoing_1': [row for row in rows if page(row) and outgoing(row) == 1],
        'current_page_outgoing_2_3': [row for row in rows if page(row) and 2 <= outgoing(row) <= 3],
        'current_page_outgoing_4_plus': [row for row in rows if page(row) and outgoing(row) >= 4],
        # This is a target-kind diagnostic, not a prediction-time acceptance
        # group: the next target's kind is known only after scoring.
        'current_page_to_page': [row for row in rows if page(row)
                                  and (row['observed_outcome'].get('metadata') or {}).get('kind') == 'page'],
        'current_page_to_page_with_mature_outgoing': [row for row in rows if source_seen(row)
                                                       and (row['observed_outcome'].get('metadata') or {}).get('kind') == 'page'],
        # This is the old, realized-next-target-conditioned metric restricted to
        # pages.  Keep it separate from the new prospective source cohort.
        'current_page_with_realized_transition_support': [row for row in rows if page(row)
                                                           and row['diagnosis_after_outcome'].get('eligible')],
    }
    return result


def frozen_test_metrics(rows, population_n):
    """Aggregate the frozen audit's exact Top1/Top4 scores without rescoring."""
    if population_n < len(rows):
        raise ValueError('A diagnostic subset cannot exceed its fixed population')
    for row in rows:
        evaluation = row['evaluation']
        diagnosis = row['diagnosis_after_outcome']
        if not isinstance(evaluation.get('hit1'), bool) or not isinstance(evaluation.get('hit4'), bool):
            raise ValueError('Frozen row has no exact Top1/Top4 score')
        if not isinstance(diagnosis.get('candidate_present'), bool):
            raise ValueError('Frozen row has no candidate-coverage audit')
    n = len(rows)
    top1 = sum(row['evaluation']['hit1'] for row in rows)
    top4 = sum(row['evaluation']['hit4'] for row in rows)
    covered = sum(row['diagnosis_after_outcome']['candidate_present'] for row in rows)
    return {
        'n': n,
        'population_n': population_n,
        'population_coverage': n / population_n if population_n else None,
        'top1_hits': top1,
        'top1': top1 / n if n else None,
        'top4_hits': top4,
        'top4': top4 / n if n else None,
        'candidate_present_n': covered,
        'candidate_present': covered / n if n else None,
        'candidate_missing_n': n - covered,
    }
