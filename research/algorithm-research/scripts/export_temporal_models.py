"""Preserve each validation-selected time-context model without deployment."""
import argparse
import gzip
import json
import os
from pathlib import Path

import evaluate_context_knn as io
from summarize_tuned_history import verify_run
from temporal_context import TemporalNavigationModel, VERSION


def restore(payload):
    if payload['version'] != VERSION or payload.get('background_features') is not False:
        raise ValueError('Unsupported temporal model contract')
    model = TemporalNavigationModel(payload['config'])
    for row in payload['records']:
        q = row['input']
        if (q['index'] != q['temporal']['through_index'] or q['index'] > row['ready_index']
                or row['ready_index'] > payload['asof_event_index']):
            raise ValueError('Saved model contains unavailable features or labels')
        model.learn(q, row['record'])
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.expanduser().resolve()
    if not out.is_relative_to(io.PRIVATE_ROOT.resolve()) or out == io.PRIVATE_ROOT.resolve() or out.exists():
        parser.error('Use a new private training directory')
    provenance, hashes = verify_run(args.run)
    for name, digest in provenance['source_hashes'].items():
        if io.sha(Path(__file__).with_name(name)) != digest:
            raise ValueError('Replay implementation differs from the trained run')
    selections = json.loads((args.run/'selection.json').read_text())['selected_by_mode']
    with gzip.open(args.run/'model-private.json.gz', 'rt') as f:
        source = json.load(f)
    os.umask(0o077)
    out.mkdir(mode=0o700, parents=True, exist_ok=False)
    exported = {}
    for mode, result in selections.items():
        payload = {**source, 'config': result['config'], 'source_run': str(args.run.resolve()),
                   'selection_basis': 'frozen early validation only', 'deployment': False}
        model = restore(payload)
        filename = f'model-{mode}-private.json.gz'
        with gzip.open(out/filename, 'wt') as f:
            json.dump(payload, f, ensure_ascii=False)
        with gzip.open(out/filename, 'rt') as f:
            restored = restore(json.load(f))
        # Serialization probes use final-state memory and do not rescore the past.
        probes = source['records'][-3:]
        if any(model.scores(r['input']) != restored.scores(r['input']) for r in probes):
            raise ValueError('Saved model scores differ after reload')
        exported[mode] = {'file': filename, 'config': result['config'],
                          'training_records': len(payload['records']),
                          'navigation_records': len(model.statistics.records[0]), 'reload_score_parity': True}
    if verify_run(args.run)[1] != hashes:
        raise ValueError('Source experiment changed during export')
    io.write_json(out/'models.json', exported)
    (out/'source-export_temporal_models.py').write_bytes(Path(__file__).read_bytes())
    io.write_json(out/'run.json', {'version':VERSION, 'complete':True,
        'source_run':str(args.run.resolve()), 'source_hashes':hashes,
        'models':exported, 'runtime_changed':False, 'collection_changed':False})
    io.write_json(out/'artifact-hashes.json', {p.name:io.sha(p) for p in out.iterdir() if p.is_file()})
    print(json.dumps({'output':str(out), 'models':exported}, ensure_ascii=False))


if __name__ == '__main__':
    main()
