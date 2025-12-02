import mujoco
import mujoco_viewer
from pathlib import Path
from scenarios.motion import Wave, Point
from utils.print_joints import create_name2idx

G1_XYZ_ROOT = 'floating_base_joint_xyz'

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
motion = Wave(speed_scale=3.0, name2idx=name2idx)
# motion = Point(speed_scale=3.0)
cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "side_view")

while viewer.is_alive:
    data.qpos[:] = 0
    data.qvel[:] = 0
    data.qacc[:] = 0
    
    # Human movement
    t = data.time
    pos, quat = motion.interpolate_pose(t)
    data.qpos[0:3] = pos
    data.qpos[3:7] = quat
    data.qpos = motion.motion(t, data.qpos.copy())
    
    # Robot movement
    data.qpos[name2idx[G1_XYZ_ROOT][2]] = 0.793
    
    # Step + render
    mujoco.mj_step(model, data)
    viewer.render()

viewer.close()
