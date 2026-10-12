"""Serve the installed Kev runtime with complete-input validation and no prompt cache.

This adapter changes this child process only; it never edits the shared FileRecall runtime.
"""
import os

os.environ["KEV_PREFIX_CACHE"] = "0"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from kev import serve


def strict_submit(self, record):
    if self.stopping.is_set():
        raise serve.HTTPException(503, "the server is stopping")
    try:
        encoded = self.model.encode(
            self.tok, record, max_state=serve.SERVE_MAX_STATE,
            max_branch=serve.SERVE_MAX_BRANCH, strict=True,
        )
    except ValueError as error:
        raise serve.HTTPException(422, str(error)) from None
    future = serve.Future()
    self.queue.put((encoded, future))
    return future


serve.Server.submit = strict_submit


@serve.app.middleware("http")
async def context_policy(request, call_next):
    response = await call_next(request)
    response.headers["X-SmartApp-Context-Policy"] = "complete-input-v1"
    return response


if __name__ == "__main__":
    serve.main()
