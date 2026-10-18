#!/usr/bin/env python3
"""Score a complete late-test submission against frozen evidence cohorts.

Submission JSON: {"3": [{"sample_id": "...", "predictions": ["target-key"]}], "5": [...]}.
Abstention is [] and remains a failure. All original test cutoffs are required.
"""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import evaluate_context_knn as core
from diagnose_dwell_errors import frozen_core
from extract_current_evidence import summarize_groups
from summarize_tuned_history import verify_run


def score_submission(rows, submitted, scoring_core=core):
    by_id = {r['sample_id']: r for r in submitted}
    expected = {r['sample_id'] for r in rows}
    if len(by_id) != len(submitted) or set(by_id) != expected:
        raise ValueError('Submit each original test cutoff exactly once, including abstentions')
    scored = []
    for original in rows:
        r = deepcopy(original); keys = by_id[r['sample_id']]['predictions']
        candidates = r['pre_prediction']['candidates']
        if not isinstance(keys, list) or not all(isinstance(k, str) for k in keys):
            raise ValueError('Predictions must be a list of target keys')
        if len(keys) > 4 or len(keys) != len(set(keys)) or any(k not in candidates for k in keys):
            raise ValueError('At most four distinct observed candidate keys are allowed')
        if scoring_core.select_four(keys, candidates) != keys:
            raise ValueError('Prediction violates the frozen destination hierarchy rule')
        outcome = r['observed_outcome']
        q = SimpleNamespace(truth=outcome['key'], truth_meta=outcome['metadata'], candidates=candidates)
        score = scoring_core.score_row(q, keys)
        r['evaluation'].update(score)
        r['evaluation']['miss_reason'] = 'hit' if score['hit4'] else (
            'not_in_candidates' if not score['covered'] else 'submitted_prediction_missed')
        scored.append(r)
    result = summarize_groups(scored)
    # Gates describe the frozen baseline, not an unprovided confidence score for
    # the submitted model. Do not accidentally attribute them to the new model.
    result.pop('prospective_first_candidate_grades')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark-run', required=True, type=Path)
    parser.add_argument('--predictions', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    run = args.benchmark_run.expanduser().resolve(); out = args.output.expanduser().resolve()
    if not out.is_relative_to(core.PRIVATE_ROOT.resolve()) or out == core.PRIVATE_ROOT.resolve():
        parser.error('Use a new private training run')
    _, benchmark_hashes = verify_run(run)
    submitted = json.loads(args.predictions.read_text())
    scoring_core = frozen_core(run)
    results = {t: score_submission(core.load_lines(run / f'all-test-{t}s-private.jsonl'), submitted[t], scoring_core) for t in ('3', '5')}
    os.umask(0o077); out.mkdir(mode=0o700, parents=True, exist_ok=False)
    core.write_json(out / 'metrics.json', results)
    core.write_json(out / 'submission-private.json', submitted)
    (out / 'core-source.py').write_bytes((run / 'core-source.py').read_bytes())
    core.write_json(out / 'run.json', {'benchmark_run': str(run), 'benchmark_hashes': benchmark_hashes,
        'submission_sha256': core.sha(args.predictions), 'candidate_policy': 'frozen_asof_candidates',
        'selection': 'fixed historical evidence cohorts, never prediction correctness', 'test_previously_seen': True})
    for name in ('score_current_evidence.py', 'extract_current_evidence.py', 'evaluate_context_knn.py', 'summarize_tuned_history.py'):
        (out / ('source-' + name)).write_text(Path(__file__).with_name(name).read_text())
    core.write_json(out / 'artifact-hashes.json', {p.name: core.sha(p) for p in out.iterdir() if p.is_file()})
    print(json.dumps(results, ensure_ascii=False))


if __name__ == '__main__':
    main()
