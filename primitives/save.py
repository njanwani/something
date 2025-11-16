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

G1_LEFT_ARM = [
    'left_shoulder_pitch_joint',
    'left_shoulder_roll_joint',
    'left_shoulder_yaw_joint',
    'left_elbow_joint',
    'left_wrist_roll_joint',
    'left_wrist_pitch_joint',
    'left_wrist_yaw_joint',
]
G1_RIGHT_ARM = [
    'right_shoulder_pitch_joint',
    'right_shoulder_roll_joint',
    'right_shoulder_yaw_joint',
    'right_elbow_joint',
    'right_wrist_roll_joint',
    'right_wrist_pitch_joint',
    'right_wrist_yaw_joint',
]
G1_JOINTS = [
    'left_hip_pitch_joint',
    'left_hip_roll_joint',
    'left_hip_yaw_joint',
    'left_knee_joint',
    'left_ankle_pitch_joint',
    'left_ankle_roll_joint',
    'right_hip_pitch_joint',
    'right_hip_roll_joint',
    'right_hip_yaw_joint',
    'right_knee_joint',
    'right_ankle_pitch_joint',
    'right_ankle_roll_joint',
    'waist_yaw_joint',
    'waist_roll_joint',
    'waist_pitch_joint',
    'head_pitch_joint',
    'head_yaw_joint',
    'head_roll_joint',
] + G1_LEFT_ARM + G1_RIGHT_ARM

model = mujoco.MjModel.from_xml_path('xmls/scene.xml')
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

def wave():
    NJOINTS = name2idx['left_wrist_yaw_joint'] - name2idx['left_shoulder_pitch_joint'] + 1
    imitate_q = np.zeros(NJOINTS)
    g1_chain = Chain.from_mujoco(
        base_body = 'left_shoulder_start',
        end_body  = 'left_hand',
        model=model,
    )
    po2D = PoseObserver2D(0.4, 'accurate', 'primitives/videos/wave-flipped.mp4')
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
        sol = imitate(
            reference_sites = left_arm_sites,
            robot_chain     = g1_chain,
            q_init          = imitate_q,
            density=20
        )
        imitate_q = sol.x
        po2D.show_keypoints()
        left_arm_traj.append(imitate_q)
    
    left_arm_traj = np.array(left_arm_traj)
    traj = np.zeros((left_arm_traj.shape[0], 32))
    
    for idx, name in enumerate(G1_LEFT_ARM):
        traj[:, -7 + name2idx[name]] = left_arm_traj[:, idx]
        
    df = pd.DataFrame(data=traj, columns=G1_JOINTS)
    return df


PRIMITIVE_FUNC = {
    'Rest': rest,
    'Wave': wave
}

if __name__ == '__main__':
    PRIMITIVE = 'Wave'
    df = PRIMITIVE_FUNC[PRIMITIVE]()
    df.to_csv(f'primitives/data/{PRIMITIVE}.csv')