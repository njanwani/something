from robosuite.models import MujocoWorldBase

world = MujocoWorldBase()

from robosuite.models.robots import GR1

mujoco_robot = GR1()

# from robosuite.models.grippers import gripper_factory

# gripper = gripper_factory('PandaGripper')
# mujoco_robot.add_gripper(gripper)

mujoco_robot.set_base_xpos([0, 0, 0])
world.merge(mujoco_robot)

from robosuite.models.arenas import TableArena

mujoco_arena = TableArena()
mujoco_arena.set_origin([0.8, 0, 0])
world.merge(mujoco_arena)

from robosuite.models.objects import BallObject
from robosuite.utils.mjcf_utils import new_joint

sphere = BallObject(
    name="sphere",
    size=[0.04],
    rgba=[0, 0.5, 0.5, 1]).get_obj()
sphere.set('pos', '1.0 0 1.0')
world.worldbody.append(sphere)

model = world.get_model(mode="mujoco")

import mujoco, time
HZ = 50
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        time_start = data.time
        real_time  = time.time()
        while data.time - time_start < 1.0 / HZ:
            mujoco.mj_step(model, data)
            
        viewer.sync()
        left_over_time = time.time() - real_time
        time.sleep(max(1.0 / HZ - left_over_time, 0.0))

viewer.close()