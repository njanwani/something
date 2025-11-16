# from eval.scenario1 import wave_motion, interpolate_pose
from eval.scenario2 import point_motion
import mujoco
import mujoco_viewer
from pathlib import Path
from eval.motion import Wave, Point
from utils.print_joints import create_name2idx, apply_named_cmd
import primitives.primitive as pm
import time

G1_XYZ_ROOT = 'floating_base_joint_xyz'
HZ = 50

path = Path('xmls/scene.xml')
model = mujoco.MjModel.from_xml_path(path.as_posix())
data = mujoco.MjData(model)
viewer = mujoco_viewer.MujocoViewer(model, data, hide_menus=True)
viewer.cam = mujoco.MjvCamera()
viewer.cam.distance = 5
viewer.cam.azimuth = 210
viewer.cam.elevation = -45

name2idx = create_name2idx(model)

hz = 50
dt = 1.0 / hz
# motion = pm.Rest(duration=4)
motion = pm.Trajectory(
    
)
# motion = Point(speed_scale=3.0)
cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "side_view")

while viewer.is_alive:
    start = time.time()
    data.qpos[:] = 0
    data.qvel[:] = 0
    data.qacc[:] = 0
    
    # Human movement
    t = data.time
    g1_cmd = motion.move(t % DURATION)
    qpos = apply_named_cmd(name2idx, data.qpos, g1_cmd)
    
    # Robot movement
    data.qpos[name2idx[G1_XYZ_ROOT][2]] = 0.793
    
    # Step + render
    t0 = data.time
    while data.time - t0 < 1 / HZ:
        mujoco.mj_step(model, data)
    viewer.render()
    time.sleep(max(1 / HZ - (time.time() - start), 0))

viewer.close()
