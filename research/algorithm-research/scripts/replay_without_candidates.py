"""Frozen-model control with no inventory whitelist or same-origin output gate."""
import argparse
import gzip
import json
import os
from pathlib import Path
from types import SimpleNamespace

import compare_action_cleaning as models
import evaluate_context_knn as io
from reopenable_actions import truth_action_key
from summarize_tuned_history import verify_run
from unrestricted_prediction import UnrestrictedPrediction, remove_evidence_gate


def metric(rows, predictions, items):
    ranks = []
    for row in rows:
        shown = predictions[row['position']]['predictions']
        actual = truth_action_key(row, items)
        ranks.append(shown.index(actual)+1 if actual in shown else 0)
    n = len(rows)
    return {'n': n, 'top4_hits': sum(bool(r) for r in ranks),
            'top1_hits': ranks.count(1), 'top4': sum(bool(r) for r in ranks)/n if n else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    _, source_hashes = verify_run(args.source_run)
    source = json.loads((args.source_run/'selection.json').read_text())['best_current_only']
    with gzip.open(args.source_run/'worker-input-private.json.gz', 'rt') as f:
        payload = json.load(f)
    inputs = {int(i): {k: q[k] for k in ('origin', 'app', 'timestamp', 'x')}
              for i, q in payload['inputs'].items()}
    items = {int(i): SimpleNamespace(**q) for i, q in payload['query_items'].items()}
    stream = payload['streams'][source['training_cleaning']]
    config = remove_evidence_gate(source['config'])
    model = UnrestrictedPrediction(models.build(config))
    predictions = {}
    cursor = 0
    for index in sorted(inputs):
        query = items[index]
        while cursor < len(stream) and stream[cursor]['ready_index'] < query.index:
            row = stream[cursor]
            assert row['position'] < index and row['outcome_index'] < query.index
            model.learn(row['record'])
            cursor += 1
        predictions[index] = {**model.predict(inputs[index]), 'sample_id': query.sample_id,
                              'matured_training_updates': cursor}
    rows = payload['cohorts']['5']
    supported = [r for r in rows if r['diagnosis_after_outcome']['eligible']]
    groups = {'all': rows, 'supported': supported,
              'late_supported': [r for r in supported if r['split'] == 'test']}
    scores = {k: metric(v, predictions, items) for k, v in groups.items()}
    os.umask(0o077)
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    io.write_json(args.output/'predictions-private.json', predictions)
    io.write_json(args.output/'model-inputs-private.json', inputs)
    io.write_json(args.output/'result.json', {'config': config, 'cleaning': source['training_cleaning'],
        'scores': scores, 'former_restricted_scores': {k: source['scores']['5'][k] for k in groups},
        'external_candidate_restriction': False, 'same_origin_gate': False,
        'new_parameter_search': False, 'prediction_uses_scoring_answers': False})
    for name in ('unrestricted_prediction.py', 'replay_without_candidates.py'):
        (args.output/('source-'+name)).write_bytes(Path(__file__).with_name(name).read_bytes())
    io.write_json(args.output/'run.json', {'complete': True, 'source_run': str(args.source_run),
        'source_artifact_hashes': source_hashes, 'test_previously_seen': True,
        'model_output_vocabulary': 'only matured historical training labels; no inventory whitelist',
        'collection_or_installation_changed': False})
    io.write_json(args.output/'artifact-hashes.json', {p.name: io.sha(p) for p in args.output.iterdir() if p.is_file()})
    print(json.dumps(scores))


if __name__ == '__main__':
    main()
