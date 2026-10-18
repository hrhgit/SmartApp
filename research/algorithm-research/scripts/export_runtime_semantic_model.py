"""Export the accepted single-step winner and native parity fixtures privately."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path

from compare_semantic_trajectory import training_rows
from summarize_tuned_history import verify_run
import evaluate_context_knn as io


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, default=Path.home()/'Library/Application Support/SmartApp')
    args = parser.parse_args()
    if (args.data_root/'training').resolve() not in args.output.resolve().parents:
        raise ValueError('Deployment artifacts must remain in private training storage')
    provenance, hashes = verify_run(args.run)
    dataset = Path(provenance['dataset'])
    dataset_run, dataset_hashes = verify_run(dataset)
    if dataset_hashes != provenance['dataset_hashes']:
        raise ValueError('Dataset provenance mismatch')
    selected = json.loads((args.run/'selection.json').read_text())['single']['config']
    required = {'family':'semantic_statistics','policy':'hard_kind','semantic_strength':16.,
                'half_life_hours':0.,'global_strength':1.,'recency_mix':.25,'baseline_rrf_mix':0.}
    if any(selected.get(k) != v for k,v in required.items()):
        raise ValueError('Native model implementation does not match selected configuration')
    with gzip.open(dataset/'dataset-private.json.gz','rt') as f:
        data = json.load(f)
    updates = []
    for row in training_rows(data,'hard_kind'):
        if row['head'] != 0:
            continue
        q = data['observations'][row['position']]
        updates.append({'readyIndex':row['ready_index'],'record':{'origin':q['origin'],
            'annotation':q['annotation'],'label':row['label'],'outcomeID':row['outcome_id'],
            'outcomeTimestamp':row['outcome_timestamp'],'weight':row['weight'],'source':'research_seed'}})
    predictions = json.loads((args.run/'selected-predictions-private.json').read_text())['single']
    queries = []
    for row in data['evaluation']:
        q = data['observations'][row['position']]
        queries.append({'position':row['position'],'index':q['index'],'origin':q['origin'],
                        'target':q['target'],'expected':predictions[str(row['position'])]['single']})
    snapshot = Path(dataset_run['snapshot'])
    current_hash = io.sha(snapshot/'events.jsonl')
    if current_hash != dataset_run['migration']['current']['events.jsonl']:
        raise ValueError('Source export changed since dataset creation')
    last = None
    with (snapshot/'events.jsonl').open() as f:
        for line in f:
            last = json.loads(line)
    source_run = json.loads((Path(dataset_run['source_run'])/'run.json').read_text())
    cleaning = json.loads((Path(source_run['cleaning_run'])/'run.json').read_text())
    seed = {'version':'semantic-statistics-hard-kind-v1','records':[r['record'] for r in updates],
            'identityKeySHA256':hashlib.sha256((args.data_root/'identity-v2.key').read_bytes()).hexdigest(),
            'seedThroughEventID':last['id'],'excludedSessions':cleaning['excluded_sessions']}
    os.umask(0o077)
    args.output.mkdir(parents=True,mode=0o700,exist_ok=False)
    io.write_json(args.output/'seed.json',seed)
    io.write_json(args.output/'native-parity-private.json',{'updates':updates,'queries':queries,
        'annotations':[{'target':q['target'],'expected':q['annotation']} for q in data['observations']]})
    io.write_json(args.output/'run.json',{'complete':True,'source_run':str(args.run),'source_hashes':hashes,
        'dataset_hashes':dataset_hashes,'model_version':seed['version'],'records':len(updates),
        'queries':len(queries),'annotations':len(data['observations']),'configuration':required,
        'seed_source_cutoff_event':last['id'],'installation_performed':False})
    (args.output/'source-export_runtime_semantic_model.py').write_bytes(Path(__file__).read_bytes())
    io.write_json(args.output/'artifact-hashes.json',{p.name:io.sha(p) for p in args.output.iterdir()})
    print(json.dumps({'records':len(updates),'queries':len(queries),'annotations':len(data['observations'])}))


if __name__ == '__main__':
    main()
