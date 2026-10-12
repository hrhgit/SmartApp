"""Synthetic checks using the existing local runtime, without loading model weights."""
import importlib.util
import queue
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

path = Path(__file__).resolve().parents[2] / "Resources" / "KevSmartAppServer.py"
spec = importlib.util.spec_from_file_location("smartapp_kev", path)
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)
from kev.model import encode


class Tokenizer:
    def __call__(self, text, **_):
        return SimpleNamespace(input_ids=[1] * len(text))

    def convert_tokens_to_ids(self, _):
        return 2


class StrictServerTests(unittest.TestCase):
    def server(self):
        return SimpleNamespace(
            stopping=threading.Event(), tok=Tokenizer(), queue=queue.Queue(),
            model=SimpleNamespace(encode=encode),
        )

    def test_oversized_state_is_rejected_before_any_model_work(self):
        server = self.server()
        record = {"state": "x" * adapter.serve.SERVE_MAX_STATE, "questions": []}
        with self.assertRaises(adapter.serve.HTTPException) as error:
            adapter.strict_submit(server, record)
        self.assertEqual(error.exception.status_code, 422)
        self.assertTrue(server.queue.empty())

    def test_complete_small_state_is_queued_without_truncation(self):
        server = self.server()
        future = adapter.strict_submit(server, {"state": "synthetic", "questions": []})
        encoded, queued_future = server.queue.get_nowait()
        self.assertFalse(encoded["state_truncated"])
        self.assertIs(future, queued_future)
        self.assertEqual(adapter.serve.PREFIX_CACHE_SIZE, 0)


if __name__ == "__main__":
    unittest.main()
