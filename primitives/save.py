import numpy as np
old_opts = np.get_printoptions()
from kinematics.observe import PoseObserver2D
np.set_printoptions(**old_opts)
import pandas as pd
from utils.print_joints import create_name2idx
import mujoco

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
    'left_shoulder_pitch_joint',
    'left_shoulder_roll_joint',
    'left_shoulder_yaw_joint',
    'left_elbow_joint',
    'left_wrist_roll_joint',
    'left_wrist_pitch_joint',
    'left_wrist_yaw_joint',
    'right_shoulder_pitch_joint',
    'right_shoulder_roll_joint',
    'right_shoulder_yaw_joint',
    'right_elbow_joint',
    'right_wrist_roll_joint',
    'right_wrist_pitch_joint',
    'right_wrist_yaw_joint',
]
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
    


PRIMITIVE_FUNC = {
    'Rest': rest
}

if __name__ == '__main__':
    PRIMITIVE = 'Rest'
    df = PRIMITIVE_FUNC[PRIMITIVE]()
    df.to_csv(f'primitives/data/{PRIMITIVE}.csv')