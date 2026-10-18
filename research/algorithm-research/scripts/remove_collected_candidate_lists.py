"""Explicit one-off redaction of collected candidateTargetIDs; no collector edits.

JSONL events and SQLite events retain every other field and every row. An audit
contains counts and hashes, never removed identities. User-authorized redaction
supersedes export immutability; updated checksums do not disguise the redaction.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile


def scrub(value):
    count = 0
    if isinstance(value, dict):
        result = {}
        for key, child in value.items():
            if key == 'candidateTargetIDs':
                count += 1
            else:
                result[key], found = scrub(child)
                count += found
        return result, count
    if isinstance(value, list):
        result = []
        for child in value:
            cleaned, found = scrub(child)
            result.append(cleaned)
            count += found
        return result, count
    return value, 0


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic(path, data):
    fd, name = tempfile.mkstemp(dir=path.parent, prefix='.redaction-')
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def clean_db(path):
    c = sqlite3.connect(path, timeout=30)
    try:
        c.execute('PRAGMA secure_delete=ON')
        c.execute('BEGIN IMMEDIATE')
        total = c.execute('select count(*) from events').fetchone()[0]
        rows = c.execute("select seq,json from events where json like '%candidateTargetIDs%'").fetchall()
        changed = 0
        for seq, text in rows:
            original = json.loads(text)
            cleaned, n = scrub(original)
            if n:
                c.execute('update events set json=? where seq=?',
                          (json.dumps(cleaned, ensure_ascii=False, separators=(',', ':')), seq))
                assert json.loads(c.execute('select json from events where seq=?', (seq,)).fetchone()[0]) == cleaned
                changed += 1
        assert c.execute('select count(*) from events').fetchone()[0] == total
        assert c.execute("select count(*) from events where json like '%candidateTargetIDs%'").fetchone()[0] == 0
        assert c.execute('PRAGMA quick_check').fetchone()[0] == 'ok'
        c.commit()
        checkpoint = c.execute('PRAGMA wal_checkpoint(TRUNCATE)').fetchone()
        return {'path': str(path), 'rows_before': total, 'rows_after': total,
                'changed_rows': changed, 'remaining_candidate_rows': 0,
                'checkpoint_result': checkpoint}
    finally:
        c.close()


def clean_jsonl(path):
    before = sha(path)
    lines = []
    changed = 0
    for line in path.read_bytes().splitlines(keepends=True):
        if b'candidateTargetIDs' not in line:
            lines.append(line)
            continue
        value, count = scrub(json.loads(line))
        if count:
            line = (json.dumps(value, ensure_ascii=False, separators=(',', ':')) + '\n').encode()
            changed += 1
        lines.append(line)
    if changed:
        atomic(path, b''.join(lines))
    return {'path': str(path), 'rows': len(lines), 'changed_rows': changed,
            'sha256_before': before, 'sha256_after': sha(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--audit-directory', type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    args.audit_directory.mkdir(mode=0o700, parents=True, exist_ok=False)
    report = {'operation': 'remove-collected-candidate-lists-v1', 'complete': False,
              'preserve_background_and_arrivals': True, 'databases': [], 'jsonl': []}
    report_path = args.audit_directory / 'redaction.json'
    def save():
        atomic(report_path, json.dumps(report, ensure_ascii=False, indent=2).encode())
    save()
    try:
        for path in [args.root / 'context-v2.sqlite', *sorted((args.root / 'backups').rglob('*.sqlite'))]:
            report['databases'].append(clean_db(path))
            save()
        for path in sorted((args.root / 'exports').rglob('*.jsonl')):
            result = clean_jsonl(path)
            report['jsonl'].append(result)
            if result['changed_rows']:
                folder = path.parent
                marker = folder / 'candidate-list-redaction.json'
                atomic(marker, json.dumps({'operation': report['operation'],
                    'user_authorized': True, 'changed_file': result,
                    'old_training_provenance_requires_explicit_redaction_migration': True}, indent=2).encode())
                checksum = folder / 'SHA256SUMS'
                if checksum.exists():
                    names = [line.split(maxsplit=1)[1].strip().lstrip('*') for line in checksum.read_text().splitlines() if line]
                    if marker.name not in names:
                        names.append(marker.name)
                    atomic(checksum, ''.join(f'{sha(folder/name)}  {name}\n' for name in names).encode())
            save()
        report['complete'] = True
        save()
    except BaseException as error:
        report['error'] = type(error).__name__ + ': ' + str(error)
        save()
        raise
    print(json.dumps({'complete': True, 'database_rows_changed': sum(x['changed_rows'] for x in report['databases']),
                      'export_rows_changed': sum(x['changed_rows'] for x in report['jsonl']),
                      'audit': str(report_path)}))


if __name__ == '__main__':
    main()
