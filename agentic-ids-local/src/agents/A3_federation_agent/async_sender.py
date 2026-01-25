# federation/async_sender.py

import threading
import time
from typing import Dict

from .signature_queue import SignatureQueue
from .fl_client import FLClient


class AsyncSignatureSender:
    def __init__(
        self,
        server_url: str,
        send_interval: int = 5,
        batch_size: int = 5
    ):
        self.queue = SignatureQueue()
        self.client = FLClient(server_url)
        self.send_interval = send_interval
        self.batch_size = batch_size
        self._running = True

        self._thread = threading.Thread(
            target=self._run,
            daemon=True
        )
        self._thread.start()

    def enqueue(self, signature: Dict):
        self.queue.enqueue(signature)

    def _run(self):
        while self._running:
            batch = self.queue.dequeue_batch(self.batch_size)

            if batch:
                success = self.client.send_signatures(batch)
                if not success:
                    # re-queue if send failed
                    self.queue.requeue_front(batch)

            time.sleep(self.send_interval)

    def stop(self):
        self._running = False
