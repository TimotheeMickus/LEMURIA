import numpy as np
import torch
import itertools
import random

import multiprocessing
import queue

# Class for datasets that by default work sequentially but can be switched (with `turnAsynchronous`) to asynchronous mode.
class SeqAsyncDataset:
    # `resources` describes the structures that the workers need to work.
    def __init__(self, **resources):
        self.asynchronous = False

        self._resources = resources

        self.close = self._synchronous_close
        self._get_batch = self._synchronous_get_batch

    # nb_workers: int
    # nb_prefetch: int (number of batches that are prefetched, per request type)
    def turnAsynchronous(self, nb_workers, nb_prefetch):
        assert (not self.asynchronous), "Dataset already asynchronous."
        assert (nb_workers >= 1)
        
        self.asynchronous = True
        self.nb_prefetch = nb_prefetch
        self.nb_requests = dict() # dict[tuple[(str, ?)], ?]
        self.task_queue = multiprocessing.Queue() # multiprocessing.Queue[?]
        self.result_queue = multiprocessing.Queue() # multiprocessing.Queue[(tuple[(str, ?)], ?)]
        self.results = dict() # dict[tuple[(str, ?)], queue.SimpleQueue[?]]
        
        self.workers = [multiprocessing.Process(target=self._worker, args=(self.task_queue, self.result_queue, self.__class__._generate_batch, self.__class__._additional_resources), kwargs=self._resources) for _ in range(nb_workers)]
        for worker in self.workers: worker.start()

        self.close = self._asynchronous_close
        self._get_batch = self._asynchronous_get_batch    

    # `kwargs` describes both the request and the structures needed.
    @staticmethod
    def _generate_batch(**kwargs):
        raise NotImplementedError("Method `_generate_batch` must be overriden.")
    
    def _synchronous_close(self):
        self._close()
    
    def _asynchronous_close(self):
        # Requests the workers to stop.
        for _ in self.workers: self.task_queue.put(None)

        # Waits for the workers to stop.
        for worker in self.workers: worker.join()

        self._close()

    # RMK: This method is replaced during initialisation.
    def _get_batch(self, **kwargs):
        raise RuntimeError("SeqAsyncDataset has not been properly initialised.")

    def _synchronous_get_batch(self, **kwargs):
        return self._generate_batch(**self._resources, **kwargs) 

    # task_queue: multiprocessing.Queue[?]
    # result_queue = multiprocessing.Queue[(tuple[(str, ?)], ?)]
    # `resources` describes the structures that the worker needs to work.
    @staticmethod
    def _worker(task_queue, result_queue, fn, gen_fn, **resources):
        print("Dataset worker started.")
        result_queue.cancel_join_thread() # So that the worker can really stop even if there is data in the queue.

        resources.update(gen_fn(**resources))
        
        while(True):
            request = task_queue.get() # tuple[(str, ?)]; blocking
            #print(f"Request ({request}) received.") # DEBUG

            if(request is None): break # Stopping request.
            result_queue.put((request, fn(**dict(request), **resources)))
        
        print("Dataset worker stopped.")

    # nb_prefetch: int
    def _asynchronous_get_batch(self, nb_prefetch=None, **kwargs):
        if(nb_prefetch is None): nb_prefetch = self.nb_prefetch
        request = tuple(sorted(kwargs.items())) # tuple[(str, ?)]

        # For requests of a new type.
        if(request not in self.results):
            self.results[request] = queue.SimpleQueue()
            self.nb_requests[request] = 0

        # Requests results.
        for _ in range(nb_prefetch + 1 - self.nb_requests[request]): # +1 for the current request
            self.task_queue.put(request)
            self.nb_requests[request] += 1

        # Waits for a result.
        results = self.results[request]
        while(results.empty()):
            (request2, result2) = self.result_queue.get() # blocking
            self.results[request2].put(result2)
        
        result = results.get()
        self.nb_requests[request] -= 1

        return result # ?
