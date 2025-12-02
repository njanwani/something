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
    def get_description(cls):
        raise NotImplementedError()
    
    @classmethod
    def get_name(cls):
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
    def get_name(cls):
        return 'Rest'
    
    @classmethod
    def get_description(cls):
        return 'Robot rests at an idle position.'
    
class Wave(Primitive):
    
    def __init__(self, duration):
        super().__init__(
            pd.read_csv(Path(r'primitives/data/Wave.csv'), index_col=0),
            duration
        )
    
    @classmethod
    def get_name(cls):
        return 'Wave'
    
    @classmethod
    def get_description(cls):
        return 'Robot waves casually with right hand.'

class FranticWave(Primitive):
    
    def __init__(self, duration):
        super().__init__(
            pd.read_csv(Path(r'primitives/data/Frantic_Wave.csv'), index_col=0),
            duration
        )
    
    @classmethod
    def get_name(cls):
        return 'FranticWave'
    
    @classmethod
    def get_description(cls):
        return 'Robot waves frantically with right hand.'
    

class DoubleWave(Primitive):
    
    def __init__(self, duration):
        super().__init__(
            pd.read_csv(Path(r'primitives/data/Double_Wave.csv'), index_col=0),
            duration
        )
    
    @classmethod
    def get_name(cls):
        return 'DoubleWave'
    
    @classmethod
    def get_description(cls):
        return 'Robot waves both hands rapidly.'

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
    def get_name(cls):
        return 'Transition'
    
    @classmethod
    def get_description(self):
        return 'Transitions between one primitive to the next'
    
def add_transitions_to_list(
    primitives: list[Primitive]
):
    ret = []
    for idx in range(len(primitives) - 1):
        ret.append(primitives[idx])
        ret.append(
            Transition(
                prev     = primitives[idx],
                next     = primitives[idx + 1],
                duration = 0.1
            )
        )
    ret.append(primitives[-1])
    return ret

PRIMITIVES = [
    Rest,
    Wave,
    FranticWave,
    DoubleWave,
    # Transition
]
    
class Trajectory:
    
    def __init__(self, *primitives: list[Primitive]):
        self.primitives = primitives
        self.num_primitives = len(primitives)
        dur = 0
        for p in self.primitives:
            dur += p.duration
        
        self.duration = dur
            
    def __call__(self, t):
        t0 = 0
        idx = 0
        while True:
            if idx < self.num_primitives - 1 and t > t0 + self.primitives[idx].duration:
                t0 += self.primitives[idx].duration
                idx += 1
            else:
                print(idx, t)
                cmd = self.primitives[idx].move(t - t0)
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