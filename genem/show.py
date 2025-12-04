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
from tqdm import tqdm

def create_viewer(model, data, viewer_mode):
    viewer = mujoco_viewer.MujocoViewer(model, data, hide_menus=True, mode=viewer_mode)
    viewer.cam = mujoco.MjvCamera()
    viewer.cam.distance = 5
    viewer.cam.azimuth = 210
    viewer.cam.elevation = -45
    return viewer

def make_scenario(scenario):
    if scenario == 'wave':
        path = Path('xmls/scene.xml')
    elif scenario == 'good-pickup':
        path = Path('xmls/scene_mug.xml')
    elif scenario == 'bad-pickup':
        path = Path('xmls/scene_knife.xml')
    else:
        raise Exception('Bad scenario')
    model = mujoco.MjModel.from_xml_path(path.as_posix())
    data = mujoco.MjData(model)
    return model, data

def create_human_motion(scenario, name2idx):
    if scenario == 'wave':
        return Wave(speed_scale=1.0, name2idx=name2idx)
    elif scenario == 'good-pickup':
        return Point(speed_scale=1.0, pickup=True, name2idx=name2idx)
    elif scenario == 'bad-pickup':
        return Point(speed_scale=1.0, pickup=False, name2idx=name2idx)
    else:
        raise Exception('Bad scenario')
    
def create_g1_motion(scenario, ctrl_mode, human_motion, name2idx):
    if ctrl_mode == 'oracle':
        if scenario == 'wave':
            begin = pm.Rest(duration=2.2)
            hi    = pm.Mix(pm.Wave(duration=3), pm.NodHead(duration=0.8))
            end   = pm.Rest(duration=2)

            g1_motion = pm.Trajectory(
                begin,
                pm.Transition(begin, hi, duration=0.3),
                hi,
                pm.Transition(hi, end, duration=0.3),
                end
            )
        elif scenario == 'good-pickup':
            begin = pm.Rest(duration=2.2)
            yes   = pm.Mix(pm.NodHead(duration=0.6), pm.Rest(duration=0.1))
            end   = pm.Rest(duration=2)

            g1_motion = pm.Trajectory(
                begin,
                pm.Transition(begin, yes, duration=0.3),
                yes,
                yes,
                yes,
                pm.Transition(yes, end, duration=0.3),
                end
            )
        elif scenario == 'bad-pickup':
            begin = pm.Rest(duration=2.2)
            no   =  pm.Mix(pm.ShakeHead(duration=3), pm.DoubleWave(duration=3))
            end   = pm.Rest(duration=2)

            g1_motion = pm.Trajectory(
                begin,
                pm.Transition(begin, no, duration=0.3),
                no,
                pm.Transition(no, end, duration=0.3),
                end
            )
        else:
            raise Exception('Bad scenario')
    elif ctrl_mode == 'genem':
        print('generating trajectory...')
        se = agi.SocialExpression(pm.PRIMITIVES)
        tg = agi.TrajectoryGenerator(pm.PRIMITIVES)
        
        expressive_description      = se.query(human_motion.generate_motion_description())
        primitives                  = tg.query(expressive_description)
        primitives_with_transitions = pm.add_transitions_to_list(primitives)
        g1_motion                   = pm.Trajectory(*primitives_with_transitions)
        
        with open(f'genem/logs/{scenario}_log.txt', 'w') as f:
            f.write(str(se.get_history_as_string()))
            f.write(str(tg.get_history_as_string()))
    else:
        raise Exception('Bad control mode')
    
    return g1_motion

def simulate(model, data, viewer, human_motion, g1_motion, name2idx, G1_XYZ_ROOT, VIDEO_MODE, HZ):
    frames = []
    for _ in tqdm(range(HZ * 13)):
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
        
        if VIDEO_MODE == 'window':
            viewer.render()
            time.sleep(max(1 / HZ - (time.time() - start), 0))
        else:
            frames.append(viewer.read_pixels())
    
    return frames

def main():
    G1_XYZ_ROOT   = 'floating_base_joint_xyz'
    HZ            = 50
    SCENARIO      = ['wave', 'good-pickup', 'bad-pickup'][0]
    CTRL_MODE     = ['genem', 'oracle'][0]
    VIDEO_MODE    = ['offscreen', 'window'][0]
    
    model, data   = make_scenario(SCENARIO)
    name2idx      = create_name2idx(model)
    viewer        = create_viewer(model, data, VIDEO_MODE)
    
    human_motion  = create_human_motion(SCENARIO, name2idx)
    g1_motion     = create_g1_motion(SCENARIO, CTRL_MODE, human_motion, name2idx)
    
    frames        = simulate(
        model           = model,
        data            = data,
        viewer          = viewer,
        human_motion    = human_motion,
        g1_motion       = g1_motion,
        name2idx        = name2idx,
        G1_XYZ_ROOT     = G1_XYZ_ROOT,
        VIDEO_MODE      = VIDEO_MODE,
        HZ              = HZ
    )

    if VIDEO_MODE == 'offscreen':
        print('saving to video')
        mp.write_video(f"videos/{SCENARIO}_{CTRL_MODE}.mp4", frames, fps=HZ)

    viewer.close()
    
if __name__ == '__main__':
    main()
