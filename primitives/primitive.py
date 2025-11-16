import numpy as np
import pandas as pd
from pathlib import Path

class Primitive:
    
    def __init__(self, trajectory, duration):
        self.traj     = trajectory
        self.length   = self.traj.shape[0]
        self.duration = duration
    
    def move(self, t):
        idx  = t / self.duration * (self.length - 1)
        idx  = np.clip(idx, 0, self.length - 1)
        prev = self.traj.iloc[np.floor(idx).astype(int)]
        next = self.traj.iloc[np.ceil(idx).astype(int)]
        
        prog = idx - np.floor(idx)
        cmd  = next * prog + prev * (1 - prog)
        
        return cmd
    
    @classmethod
    def description(self):
        raise NotImplementedError()
    
    @property
    def first_position(self):
        return self.traj.iloc[0]
    
    @property
    def last_position(self):
        return self.traj.iloc[-1]
    
class Rest(Primitive):
    
    def __init__(self, duration):
        super().__init__(
            pd.read_csv(Path(r'primitives/data/Rest.csv'), index_col=0),
            duration
        )
    
    @classmethod
    def description(cls):
        return 'TBD'
    
class Wave(Primitive):
    
    def __init__(self, duration):
        super().__init__(
            pd.read_csv(Path(r'primitives/data/Wave.csv'), index_col=0),
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
    
# class Wave(Primitive):
#     raise NotImplementedError()

# class Nod(Primitive):
#     raise NotImplementedError()

# class Guide(Primitive):
#     raise NotImplementedError()


if __name__ == '__main__':
    p = Rest(duration=1)
    print(p.traj)
    print()
    print(p.move(0))
    print()
    print(p.move(0.5))
    print()
    print(p.move(1))