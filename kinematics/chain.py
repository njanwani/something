import numpy as np
import mujoco
from pathlib import Path

def rot_axis(axis, angle):
    """Return 3x3 rotation matrix for rotation about 'axis' by 'angle'."""
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    C = 1 - c
    return np.array([
        [c + x*x*C, x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s, c + y*y*C, y*z*C - x*s],
        [z*x*C - y*s, z*y*C + x*s, c + z*z*C]
    ])

class Chain:
    
    def __init__(self, link_transforms, joint_axes):
        """Initialize chain with link translations and joint axes.
        
        Args:
            link_transforms: (N, 3) array, local translations between joints.
            joint_axes: (N, 3) array, joint rotation axes in local frames.
        """
        self.link_transforms = np.array(link_transforms)
        self.joint_axes      = np.array(joint_axes)
        self.n = len(link_transforms)
    
    @classmethod
    def from_mujoco(cls, base_body, end_body, model):
        """Build Chain from a MuJoCo model between two bodies."""
        body_id_base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, base_body)
        body_id_end  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, end_body)
        
        # Walk up the tree from end_body to base_body
        body_chain = []
        jnt_chain  = []
        bid = body_id_end
        while bid != body_id_base and bid != -1:
            body_chain.append(bid)
            jid = model.body_jntadr[bid]
            if jid != -1:  # body has a joint
                jnt_chain.append(jid)
            bid = model.body_parentid[bid]
        
        body_chain.reverse()
        jnt_chain.reverse()
        
        link_transforms = []
        joint_axes = []
        
        # Compute relative transforms and joint axes in local frames
        for jid in jnt_chain:
            axis = model.jnt_axis[jid]
            body_id = model.jnt_bodyid[jid]
            parent_id = model.body_parentid[body_id]
            
            # Compute local translation from parent to this body
            parent_pos = model.body_pos[parent_id]
            body_pos = model.body_pos[body_id]
            link_T = body_pos - parent_pos
            
            link_transforms.append(link_T)
            joint_axes.append(axis)
        
        return cls(link_transforms, joint_axes)
    
    def forward_kinematics(self, q):
        """Compute 4x4 transforms of each joint frame in world coordinates."""
        assert(len(q) == self.n)
        T = np.eye(4)
        transforms = [T.copy()]
        
        for i in range(self.n):
            R = rot_axis(self.joint_axes[i], q[i])
            T_rot = np.eye(4)
            T_trans = np.eye(4)
            T_rot[:3, :3] = R
            T_trans[:3, 3]  = self.link_transforms[i]
            T = T @ T_rot @ T_trans
            transforms.append(T.copy())
        
        return transforms
    
    
    def compute_path(self, q, s):
        """Return SE(3) transforms along the chain at fractions s∈[0,1],
        with constant rotation along each link.
        """
        transforms = self.forward_kinematics(q)
        joint_positions = np.array([T[:3, 3] for T in transforms])
        
        # Compute cumulative distances
        segment_lengths = np.linalg.norm(np.diff(joint_positions, axis=0), axis=1)
        cum_lengths = np.concatenate([[0], np.cumsum(segment_lengths)])
        total_length = cum_lengths[-1]
        
        s = np.clip(np.array(s), 0, 1)
        target_lengths = s * total_length
        path_transforms = []
        
        for L in target_lengths:
            i = np.searchsorted(cum_lengths, L) - 1
            i = np.clip(i, 0, self.n - 1)
            t = (L - cum_lengths[i]) / (cum_lengths[i+1] - cum_lengths[i] + 1e-8)
            
            # Linear position interpolation along link i
            p0 = joint_positions[i]
            p1 = joint_positions[i+1]
            p = (1 - t) * p0 + t * p1
            
            # Constant rotation: orientation of link i
            R_const = transforms[i + 1][:3, :3]
            
            # Construct SE(3)
            T_interp = np.eye(4)
            T_interp[:3, :3] = R_const
            T_interp[:3, 3] = p
            path_transforms.append(T_interp)
        
        return np.array(path_transforms)
    
def log_SO3_batch(Rs):
    """Vectorized log map for SO(3) rotation matrices of shape (..., 3, 3)."""
    traces = np.trace(Rs, axis1=-2, axis2=-1)
    cos_theta = np.clip((traces - 1) / 2, -1, 1)
    theta = np.arccos(cos_theta)
    
    wx = np.stack([
        Rs[..., 2, 1] - Rs[..., 1, 2],
        Rs[..., 0, 2] - Rs[..., 2, 0],
        Rs[..., 1, 0] - Rs[..., 0, 1]
    ], axis=-1) / 2
    
    small = np.isclose(theta, 0)
    w = np.zeros_like(wx)
    w[~small] = (theta[~small, None] / np.sin(theta[~small, None])) * wx[~small]
    return w


def log_SE3_batch(Ts, reg=0.01):
    """Vectorized SE(3) log map for transforms of shape (..., 4, 4)."""
    Rs = Ts[..., :3, :3]
    ps = Ts[..., :3, 3]
    ws = log_SO3_batch(Rs)
    thetas = np.linalg.norm(ws, axis=-1)
    
    wx = np.zeros(Ts[..., :3, :3].shape)
    wx[..., 0, 1] = -ws[..., 2]; wx[..., 0, 2] =  ws[..., 1]
    wx[..., 1, 0] =  ws[..., 2]; wx[..., 1, 2] = -ws[..., 0]
    wx[..., 2, 0] = -ws[..., 1]; wx[..., 2, 1] =  ws[..., 0]
    
    V_inv = np.eye(3) - 0.5 * wx
    nonzero = ~np.isclose(thetas, 0)
    thetas[~nonzero] = np.pi / 2 # to avoid div by 0
    theta = thetas[..., None, None]
    term = (1/(theta**2)) * (1 - (theta*np.sin(theta)) / (2*(1 - np.cos(theta))))
    V_inv[nonzero] += term[nonzero] * (wx[nonzero] @ wx[nonzero])
    
    v = np.einsum('...ij,...j->...i', V_inv, ps)
    return np.concatenate([v, ws], axis=-1)


def compute_path_error(path1, path2):
    """Vectorized SE(3) geodesic error between two paths of equal length."""
    path1 = np.asarray(path1)
    path2 = np.asarray(path2)
    assert path1.shape == path2.shape
    
    # Compute ΔT = inv(T1) @ T2 for all pairs
    R1 = path1[..., :3, :3]
    p1 = path1[..., :3, 3]
    R2 = path2[..., :3, :3]
    p2 = path2[..., :3, 3]
    
    R_delta = np.einsum('...ij,...jk->...ik', np.transpose(R1, (0, 2, 1)), R2)
    p_delta = np.einsum('...ij,...j->...i',   np.transpose(R1, (0, 2, 1)), p2 - p1)
    
    delta_T = np.zeros_like(path1)
    delta_T[..., :3, :3] = R_delta
    delta_T[..., :3, 3] = p_delta
    delta_T[..., 3, 3] = 1.0
    
    xi = log_SE3_batch(delta_T)
    return np.linalg.norm(xi, axis=-1)

if __name__ == '__main__':
    # 3-link chain: each link 1m, each axis z, y, x
    links = [
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1]
    ]
    axes = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ]

    chain = Chain(links, axes)
    q = np.zeros(3)
    q[0] = np.pi / 2
    q[1] = np.pi / 2
    q[2] = np.pi / 2
    s = np.linspace(0, 1, 9)

    path = chain.compute_path(q, s)
    
    # Load the Unitree G1 model
    path = Path('xmls/generic_arm/arm6DOF.xml')
    model = mujoco.MjModel.from_xml_string(path.read_text())
    data = mujoco.MjData(model)

    chain = Chain.from_mujoco(
        base_body = 'base',
        end_body  = 'end_effector',
        model=model,
    )
    print(chain.joint_axes)
    print('--')
    print(chain.link_transforms)