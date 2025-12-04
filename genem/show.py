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
from genem import agents as agi

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

hz = 50
dt = 1.0 / hz
mode = ['genem', 'oracle'][0]
# human_motion = Wave(speed_scale=1.0, name2idx=name2idx)
human_motion = Point(speed_scale=1.0, pickup=True, name2idx=name2idx)
if mode == 'oracle':
    begin = pm.Rest(duration=2)
    wave  = pm.Wave(duration=4)
    end   = pm.Rest(duration=2)

    g1_motion = pm.Trajectory(
        begin,
        pm.Transition(begin, wave, duration=0.5),
        wave,
        pm.Transition(wave, end, duration=0.5),
        end
    )
elif mode == 'genem':
    print('generating trajectory...')
    se = agi.SocialExpression()
    tg = agi.TrajectoryGenerator(pm.PRIMITIVES)
    
    expressive_description      = se.query(human_motion.generate_motion_description())
    primitives                  = tg.query(expressive_description)
    primitives_with_transitions = pm.add_transitions_to_list(primitives)
    g1_motion                   = pm.Trajectory(*primitives_with_transitions)
    
    with open('genem/log.txt', 'w') as f:
        f.write(str(se.get_history_as_string()))
        f.write(str(tg.get_history_as_string()))

cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "side_view")
frames = []
while viewer.is_alive:
    if data.time > 13:
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
    mp.write_video(f"{mode}.mp4", frames, fps=30)

viewer.close()
