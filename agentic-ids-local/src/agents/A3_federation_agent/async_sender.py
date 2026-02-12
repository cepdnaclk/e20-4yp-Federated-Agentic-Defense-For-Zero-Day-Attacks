# federation/async_sender.py

import threading
import time
import logging
from typing import Dict, List

from .signature_queue import SignatureQueue
from .fl_client import FLClient


class AsyncSignatureSender:
    def __init__(
        self,
        server_url: str,
        send_interval: int = 5,
        batch_size: int = 5,
        max_retries: int = 3
    ):
        self.queue = SignatureQueue()
        self.client = FLClient(server_url)
        self.send_interval = send_interval
        self.batch_size = batch_size
        self.max_retries = max_retries
        self._running = True
        self.logger = logging.getLogger(__name__)

        self._thread = threading.Thread(
            target=self._run,
            daemon=True
        )
        self._thread.start()

    def enqueue(self, signature: Dict):
        # Add retry count to signature metadata
        signature_with_meta = {
            "data": signature,
            "retry_count": 0
        }
        self.queue.enqueue(signature_with_meta)

    def _run(self):
        while self._running:
            batch = self.queue.dequeue_batch(self.batch_size)

            if batch:
                # Extract signature data for sending
                signature_data = [item["data"] for item in batch]
                success = self.client.send_signatures(signature_data)
                
                if not success:
                    # Handle retry logic
                    retry_batch = []
                    dropped_count = 0
                    
                    for item in batch:
                        item["retry_count"] += 1
                        
                        if item["retry_count"] <= self.max_retries:
                            retry_batch.append(item)
                        else:
                            dropped_count += 1
                            self.logger.warning(
                                f"Dropping signature after {self.max_retries} failed attempts"
                            )
                    
                    # Re-queue items that haven't exceeded retry limit
                    if retry_batch:
                        self.queue.requeue_front(retry_batch)
                        
                    if dropped_count > 0:
                        self.logger.info(f"Dropped {dropped_count} signatures due to max retry limit")

            time.sleep(self.send_interval)

    def stop(self):
        self._running = False
