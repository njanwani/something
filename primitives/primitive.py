import numpy as np
import pandas as pd
from pathlib import Path


class Primitive:
    def __init__(self, trajectory, duration):
        self.traj = trajectory
        self.length = self.traj.shape[0]
        self.duration = duration

    def move(self, t):
        idx = t / self.duration * (self.length - 1)
        idx = np.clip(idx, 0, self.length - 1)
        prev = self.traj.iloc[np.floor(idx).astype(int)]
        next = self.traj.iloc[np.ceil(idx).astype(int)]

        prog = idx - np.floor(idx)
        cmd = next * prog + prev * (1 - prog)

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
            pd.read_csv(Path(r"primitives/data/Rest.csv"), index_col=0), duration
        )

    @classmethod
    def description(cls):
        return "Robot stays still"


class Wave(Primitive):
    def __init__(self, duration):
        super().__init__(
            pd.read_csv(Path(r"primitives/data/Wave.csv"), index_col=0), duration
        )

    @classmethod
    def description(cls):
        return "Robot waves with right arm"


class Frantic_Wave(Primitive):
    def __init__(self, duration):
        super().__init__(
            pd.read_csv(Path(r"primitives/data/Frantic_Wave.csv"), index_col=0),
            duration,
        )

    @classmethod
    def description(cls):
        return "Robot waves frantically with one arm"


class Double_Wave(Primitive):
    def __init__(self, duration):
        super().__init__(
            pd.read_csv(Path(r"primitives/data/Double_Wave.csv"), index_col=0), duration
        )

    @classmethod
    def description(cls):
        return "Robot waves with both arms"


class Transition(Primitive):
    def __init__(self, prev: Primitive, next: Primitive, duration: float):
        trajectory = np.array([prev.last_position, next.first_position])
        super().__init__(trajectory, duration)

    @classmethod
    def description(self):
        return "Robot transitions between one primitive to the next"


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

    @property
    def duration(self):
        """Total duration of all primitives in the trajectory."""
        return sum(p.duration for p in self.primitives)

    def __call__(self, t):
        t0 = 0
        idx = 0
        while True:
            if t > t0 + self.primitives[idx].duration and idx < self.num_primitives - 1:
                idx += 1
                t0 += self.primitives[idx].duration
            else:
                cmd = self.primitives[idx].move(t - t0)
                break

        return cmd


PRIMITIVE_REGISTRY = {
    # Basic motions
    "rest": Rest,
    "idle": Rest,
    "stay_still": Rest,
    # Wave variations
    "wave": Wave,
    "wave_hello": Wave,
    "wave_greeting": Wave,
    "frantic_wave": Frantic_Wave,
    "double_wave": Double_Wave,
    # Add new primitives here following this pattern:
    # 'primitive_name': PrimitiveClass,
}


def get_primitive_descriptions():
    """Get descriptions of all available primitives."""
    descriptions = {}
    seen_classes = set()
    for name, cls in PRIMITIVE_REGISTRY.items():
        if cls not in seen_classes:
            descriptions[name] = cls.description()
            seen_classes.add(cls)
    return descriptions


def create_motion_function(primitive_name, duration=3.0, name2idx=None):
    """Create a motion function from a primitive that matches LLM motion signature.

    Args:
        primitive_name: Name of primitive from PRIMITIVE_REGISTRY
        duration: Duration of the motion in seconds
        name2idx: Joint name to qpos index mapping (optional, can be provided later)

    Returns:
        A function with signature motion_fn(t, qpos) -> qpos
    """
    if primitive_name not in PRIMITIVE_REGISTRY:
        raise ValueError(f"Unknown primitive: {primitive_name}")

    primitive = PRIMITIVE_REGISTRY[primitive_name](duration=duration)

    def motion_fn(t, qpos):
        from utils.print_joints import apply_named_cmd

        cmd = primitive.move(t)

        # If name2idx not provided, we need the model to create it
        # For now, apply the command if we have the mapping
        if name2idx is not None:
            qpos = apply_named_cmd(name2idx, qpos, cmd)

        return qpos

    return motion_fn


if __name__ == "__main__":
    p = Trajectory(Rest(duration=1), Wave(duration=2), Rest(duration=1))

    for t in range(4):
        print(p(t))
