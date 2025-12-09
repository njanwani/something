# from eval.scenario1 import wave_motion, interpolate_pose
from eval.scenario2 import point_motion
import mujoco
import mujoco_viewer
from pathlib import Path
from scenarios.motion import Wave, Point
from utils.print_joints import create_name2idx, apply_named_cmd
import primitives.primitive as pm
import time
from scenarios.motion import Wave, Point
import mediapy as mp

G1_XYZ_ROOT = 'floating_base_joint_xyz'
HZ = 50

path = Path('xmls/scene.xml')
model = mujoco.MjModel.from_xml_path(path.as_posix())
data = mujoco.MjData(model)
viewer_mode = ['window', 'offscreen'][0]
viewer = mujoco_viewer.MujocoViewer(model, data, hide_menus=True, mode=viewer_mode)
viewer.cam = mujoco.MjvCamera()
viewer.cam.distance = 5
viewer.cam.azimuth = 210
viewer.cam.elevation = -45

name2idx = create_name2idx(model)

one   = pm.Rest(duration=2)
# two   = pm.Mix(pm.NodHead(duration=2), pm.Wave(duration=2))
two = pm.Wave(duration=4)
three = pm.Rest(duration=20)

g1_motion = pm.Trajectory(
    one,
    pm.Transition(one, two, duration=0.5),
    two,
    two,
    two,
    two,
    two,
    two,
    two,
)

# human_motion = Wave(speed_scale=1.0, name2idx=name2idx)
human_motion = Point(speed_scale=1.0, name2idx=name2idx)
# motion = Point(speed_scale=3.0)
cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "side_view")
frames = []
while viewer.is_alive:
    if data.time > g1_motion.duration:
        break
    start = time.time()
    data.qpos[:] = 0
    data.qvel[:] = 0
    data.qacc[:] = 0
    
    # Human movement
    t = data.time
    g1_cmd = g1_motion(t)
    
    pos, quat = human_motion.interpolate_pose(t)
    data.qpos[0:3] = pos
    data.qpos[3:7] = quat
    qpos = human_motion.motion(t, data.qpos)
    qpos = apply_named_cmd(name2idx, data.qpos, g1_cmd)
    
    # Robot movement
    data.qpos[name2idx[G1_XYZ_ROOT][2]] = 0.793
    
    # Step + render
    t0 = data.time
    while data.time - t0 < 1 / HZ:
        mujoco.mj_step(model, data)
    
    if viewer_mode == 'window':
        viewer.render()
    else:
        frames.append(viewer.read_pixels())
    time.sleep(max(1 / HZ - (time.time() - start), 0))
    

if viewer_mode == 'offscreen':
    mp.write_video(f"primitives/show.mp4", frames, fps=30)

viewer.close()
