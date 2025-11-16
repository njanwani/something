import numpy as np
import pandas as pd
from pathlib import Path

class Primitive:
    
    def __init__(self, trajectory, duration):
        self.traj     = trajectory
        self.length   = self.traj.shape[0]
        self.duration = duration
    
    def move(self, t):
        idx  = np.round(t / self.duration) * self.length
        idx  = np.clip(idx, 0, self.length)
        prev = self.traj[np.floor(idx)]
        next = self.traj(np.ceil(idx))
        
        prog = idx - np.floor(idx)
        cmd  = next * prog + prev * (1 - prev)
        
        return cmd
    
    @classmethod
    def description(self):
        raise NotImplementedError()
    
    @property
    def first_position(self):
        return self.traj[0]
    
    @property
    def last_position(self):
        return self.traj[-1]
    
class Rest(Primitive):
    
    def __init__(self, duration):
        super().__init__(
            Path(r'primitives/data/rest.csv'),
            duration
        )
    
    @classmethod
    def description(cls):
        return 'TBD'

class Transition(Primitive):
    
    def __init__(
        self,
        prev: Primitive, 
        next: Primitive,
        duration: float
    ):
        trajectory = np.array([
            prev.last_position,
            next.first_position
        ])
        super().__init__(trajectory, duration)
    
    @classmethod
    def description(self):
        return 'Transitions between one primitive to the next'
    
class Wave(Primitive):
    raise NotImplementedError()

class Nod(Primitive):
    raise NotImplementedError()

class Guide(Primitive):
    raise NotImplementedError()