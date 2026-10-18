import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from remove_collected_candidate_lists import scrub, clean_db, clean_jsonl


class Tests(unittest.TestCase):
    def test_only_candidate_list_removed(self):
        v = {'data': {'candidateTargetIDs': ['a'], 'items': [{'targetID': 'a'}],
                      'inventoryEventIDs': ['b'], 'background': ['a'], 'selected': 'a'}}
        cleaned, count = scrub(v)
        self.assertEqual(count, 1)
        self.assertEqual(cleaned, {'data': {k: value for k, value in v['data'].items() if k != 'candidateTargetIDs'}})
        self.assertIn('candidateTargetIDs', v['data'])

    def test_sql_and_jsonl_preserve_records_and_are_idempotent(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            c = sqlite3.connect(p/'events.sqlite')
            c.execute('create table events(seq integer primary key, json text)')
            values = [{'data': {'candidateTargetIDs': ['a'], 'items': [1]}}, {'background': ['a']}]
            c.executemany('insert into events(json) values(?)', [(json.dumps(v),) for v in values])
            c.commit(); c.close()
            self.assertEqual(clean_db(p/'events.sqlite')['changed_rows'], 1)
            self.assertEqual(clean_db(p/'events.sqlite')['changed_rows'], 0)
            f = p/'events.jsonl'; f.write_text(''.join(json.dumps(v)+'\n' for v in values))
            self.assertEqual(clean_jsonl(f)['changed_rows'], 1)
            self.assertEqual(clean_jsonl(f)['changed_rows'], 0)
            self.assertEqual([json.loads(l) for l in f.read_text().splitlines()], [scrub(v)[0] for v in values])


if __name__ == '__main__':
    unittest.main()
