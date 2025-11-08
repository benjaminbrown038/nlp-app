import asyncio
from typing import Any, List, Tuple


class PipelineBatcher:
    def __init__(self, pipeline, batch_size: int = 8, max_wait_ms: int = 20):
        self.pipeline = pipeline
        self.batch_size = batch_size
        self.max_wait = max_wait_ms / 1000.0

        self.queue: List[Tuple[str, asyncio.Future]] = []
        self.lock = asyncio.Lock()

    async def submit(self, text: str):
        """
        Add a request to the batch queue.
        Returns an awaitable future that will receive the prediction.
        """
        future: asyncio.Future = asyncio.get_event_loop().create_future()

        async with self.lock:
            self.queue.append((text, future))

            # If we hit batch size early, process immediately
            if len(self.queue) >= self.batch_size:
                asyncio.create_task(self.process_batch())

        # Otherwise, process after timeout
        asyncio.create_task(self._delayed_process())
        return await future

    async def _delayed_process(self):
        await asyncio.sleep(self.max_wait)
        async with self.lock:
            if self.queue:
                asyncio.create_task(self.process_batch())

    async def process_batch(self):
        async with self.lock:
            batch = self.queue
            self.queue = []

        texts = [t for t, _ in batch]
        futures = [f for _, f in batch]

        try:
            results = self.pipeline(texts)
            for fut, res in zip(futures, results):
                fut.set_result(res)
        except Exception as e:
            for fut in futures:
                fut.set_exception(e)
