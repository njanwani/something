import numpy as np
old_opts = np.get_printoptions()
from kinematics.observe import PoseObserver2D
from kinematics.solver import imitate
from kinematics.chain import Chain
np.set_printoptions(**old_opts)
import pandas as pd
from utils.print_joints import create_name2idx
import mujoco
import cv2
import sys
from primitives.utils import G1_JOINTS, G1_LEFT_ARM, G1_RIGHT_ARM

model = mujoco.MjModel.from_xml_path('xmls/scene_only_g1.xml')
name2idx = create_name2idx(model)

def rest():
    rest_pos = np.zeros(32)
    rest_pos[-7 + name2idx['left_shoulder_pitch_joint']]  =  0.28
    rest_pos[-7 + name2idx['left_shoulder_roll_joint']]   =  0.121
    rest_pos[-7 + name2idx['left_elbow_joint']]           =  0.916
    rest_pos[-7 + name2idx['right_shoulder_pitch_joint']] =  0.28
    rest_pos[-7 + name2idx['right_shoulder_roll_joint']]  = -0.121
    rest_pos[-7 + name2idx['right_elbow_joint']]          =  0.916
    traj     = np.array([rest_pos, rest_pos]).astype(float)
    df = pd.DataFrame(data=traj, columns=G1_JOINTS)
    return df

def nodhead():
    NODS = 1
    t = np.linspace(0, 2 * np.pi * NODS, num=50)
    nodding_traj = np.sin(t) * 0.2
    traj = np.zeros((t.shape[0], len(G1_JOINTS)))
    traj[:, -7 + name2idx['head_pitch_joint']] = nodding_traj
    df = pd.DataFrame(data=traj, columns=G1_JOINTS)
    return df

def shakehead():
    NODS = 4
    t = np.linspace(0, 2 * np.pi * NODS, num=50)
    nodding_traj = np.sin(t) * 0.4
    traj = np.zeros((t.shape[0], len(G1_JOINTS)))
    traj[:, -7 + name2idx['head_yaw_joint']] = nodding_traj
    df = pd.DataFrame(data=traj, columns=G1_JOINTS)
    return df

def wave():
    NJOINTS = name2idx['left_wrist_yaw_joint'] - name2idx['left_shoulder_pitch_joint'] + 1
    left_q  = np.zeros(NJOINTS)
    right_q = np.zeros(NJOINTS)
    left_arm_chain = Chain.from_mujoco(
        base_body = 'left_shoulder_start',
        end_body  = 'left_hand',
        model=model,
    )
    right_arm_chain = Chain.from_mujoco(
        base_body = 'right_shoulder_start',
        end_body  = 'right_hand',
        model=model,
    )
    po2D = PoseObserver2D(0.4, 'accurate', 'primitives/videos/wave.mp4')
    ret = True
    left_arm_traj  = []
    right_arm_traj = []
    while ret:
        ret = po2D.read_pose()
        if po2D.frame is None:
            break
        left_arm_sites = np.concatenate(
            [np.zeros((3, 1)), po2D.left_arm], axis=1
        ) + np.array([0.0, 0.1, 1.1])
        right_arm_sites = np.concatenate(
            [np.zeros((3, 1)), po2D.right_arm], axis=1
        ) + np.array([0.0, -0.1, 1.1])
        sol = imitate(
            reference_sites = left_arm_sites,
            robot_chain     = left_arm_chain,
            q_init          = left_q,
            density=20
        )
        left_q = sol.x.copy()
        
        sol = imitate(
            reference_sites = right_arm_sites,
            robot_chain     = right_arm_chain,
            q_init          = right_q,
            density=20
        )
        right_q = sol.x.copy()
        po2D.show_keypoints()
        left_arm_traj.append(left_q)
        right_arm_traj.append(right_q)
    
    left_arm_traj = np.array(left_arm_traj)
    right_arm_traj = np.array(right_arm_traj)
    traj = np.zeros((left_arm_traj.shape[0], 32))
    
    for idx, name in enumerate(G1_LEFT_ARM):
        traj[:, -7 + name2idx[name]] = left_arm_traj[:, idx]
    
    for idx, name in enumerate(G1_RIGHT_ARM):
        traj[:, -7 + name2idx[name]] = right_arm_traj[:, idx]
        
    # post process
    post_traj = np.zeros_like(traj)
    post_traj[0] = traj[0]
    SMOOTH = 3
    for i in range(1, traj.shape[0]):
        post_traj[i] = post_traj[i - 1] + 1 / SMOOTH * (traj[i] - post_traj[i - 1])
        
    df = pd.DataFrame(data=post_traj, columns=G1_JOINTS)
    return df


PRIMITIVE_FUNC = {
    'Rest': rest,
    'Wave': wave,
    'NodHead': nodhead,
    'ShakeHead': shakehead
}

if __name__ == '__main__':
    PRIMITIVE = sys.argv[1]
    df = PRIMITIVE_FUNC[PRIMITIVE]()
    df.to_csv(f'primitives/data/{PRIMITIVE}.csv')