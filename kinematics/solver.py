import numpy as np
from kinematics.chain import Chain, compute_path_error
from scipy import optimize

def overall_error(chain1: Chain, chain2: Chain, q1, q2, num, alpha):
    s = np.linspace(0, 1, num=num)
    path1 = chain1.compute_path(q1, s)
    path2 = chain2.compute_path(q2, s)

    pose_error = np.mean(compute_path_error(path1, path2))
    ee_error   = compute_path_error(path1[[-1]], path2[[-1]])
    error = pose_error + alpha * ee_error
    return error, {
        'pose_error': pose_error,
        'ee_error':   ee_error
    }

def imitate(
    reference_chain: Chain,
    reference_q: np.ndarray,
    actual_chain: Chain, 
    q_init: np.ndarray,
    density=100,
    alpha=0.0
):
    def cost(x: np.ndarray):
        return overall_error(
            chain1 = reference_chain,
            chain2 = actual_chain,
            q1     = reference_q,
            q2     = x,
            num    = density,
            alpha  = alpha
        )[0]
    print('Initial error', cost(q_init))
        
    sol = optimize.minimize(
        fun    = cost,
        x0     = q_init,
        method = 'Powell',
        options = dict(maxiter=500, xtol=1e-5, ftol=1e-5)
    )
    return sol

if __name__ == '__main__':
    # 3-link chain: each link 1m, each axis z, y, x
    links = [
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1]
    ]
    axes = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ]
    reference_chain = Chain(links, axes)
    
    links = [
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 0.5],
        [0, 0, 0.5],
        [0, 0, 1]
    ]
    axes = [
        [0, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 1]
    ]
    actual_chain = Chain(links, axes)
    
    reference_q    = np.zeros(3)
    reference_q[1] = np.pi / 2
    actual_q       = np.ones(4) * 0.9
    
    s = np.linspace(0, 1, 9)
    
    sol = imitate(
        reference_chain,
        reference_q,
        actual_chain,
        actual_q,
    )
    print(sol)
