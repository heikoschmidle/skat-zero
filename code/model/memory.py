import numpy as np
from collections import deque


class Memory:
    def __init__(self, memory_size):
        self.size = memory_size
        self.ltmemory = deque(maxlen=self.size)
        self.stmemory = deque(maxlen=self.size)

    def commit_stmemory(self, state, value, action_values, current_position):
        self.stmemory.append({
            'state': state,
            'value': value,
            'action_values': action_values,
            'current_pos': current_position
        })

    def commit_ltmemory(self):
        for i in self.stmemory:
            self.ltmemory.append(i)
        self.clear_stmemory()

    def clear_stmemory(self):
        self.stmemory = deque(maxlen=self.size)

    def clear_ltmemory(self):
        self.ltmemory = deque(maxlen=self.size)
