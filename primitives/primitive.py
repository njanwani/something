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
        trajectory = pd.concat([
            prev.last_position.to_frame().T,
            next.first_position.to_frame().T
        ])
        super().__init__(trajectory, duration)
    
    @classmethod
    def description(self):
        return 'Transitions between one primitive to the next'

# class Nod(Primitive):
#     raise NotImplementedError()
# NodYes
# NodNo
# NodAcknowledge
# class Guide(Primitive):
#     raise NotImplementedError()

class Trajectory:
    
    def __init__(self, *primitives: list[Primitive]):
        self.primitives = primitives
        self.num_primitives = len(primitives)
        print(self.num_primitives)
        dur = 0
        for p in self.primitives:
            dur += p.duration
        
        self.duration = dur
            
    def __call__(self, t):
        t0 = 0
        idx = 0
        while True:
            if idx < self.num_primitives - 2 and t > t0 + self.primitives[idx + 1].duration:
                idx += 1
                t0 += self.primitives[idx].duration
            else:
                cmd = self.primitives[idx + 1].move(t - t0)
                break
        
        return cmd
    
        
if __name__ == '__main__':
    p = Trajectory(
        Rest(duration=1),
        Wave(duration=2),
        Rest(duration=1)
    )

    for t in range(4):
        print(p(t))