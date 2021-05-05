from collections import deque

import numpy as np


class History(dict):
    """
    A memory that retains the track of histories.
    """
    def __init__(self, *keys, maxlen):
        """
        Initialize queues with the given keys and max length.
        """
        super().__init__()
        for key in keys:
            self[key] = deque(maxlen=maxlen)
        self.length = 0

    def __len__(self):
        return self.length

    def append(self, *values):
        """
        Append new history to the queues.
        """
        for queue, value in zip(self.values(), values):
            queue.append(value)
        self.length = len(queue)

    def replay(self, batch_size):
        """
        Mini-batch replay method.
        """
        items = [np.array(queue) for queue in self.values()]
        indices = np.arange(len(self))
        np.random.shuffle(indices)
        for start in range(0, len(self), batch_size):
            end = start + batch_size
            index = indices[start:end]
            yield [item[index] for item in items]
