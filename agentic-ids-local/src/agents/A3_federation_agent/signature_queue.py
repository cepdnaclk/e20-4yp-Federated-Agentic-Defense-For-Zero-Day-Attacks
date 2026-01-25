# federation/signature_queue.py

import threading
from typing import List, Dict


class SignatureQueue:
    def __init__(self, max_size: int = 100):
        self._queue: List[Dict] = []
        self._lock = threading.Lock()
        self._max_size = max_size

    def enqueue(self, signature: Dict):
        with self._lock:
            if len(self._queue) >= self._max_size:
                # drop oldest if full (non-blocking design)
                self._queue.pop(0)
            self._queue.append(signature)

    def dequeue_batch(self, batch_size: int) -> List[Dict]:
        with self._lock:
            if not self._queue:
                return []

            batch = self._queue[:batch_size]
            self._queue = self._queue[batch_size:]
            return batch

    def requeue_front(self, signatures: List[Dict]):
        with self._lock:
            self._queue = signatures + self._queue

    def size(self) -> int:
        with self._lock:
            return len(self._queue)
