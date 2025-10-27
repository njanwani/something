import numpy as np
from kinematics.chain import Chain, compute_path_error
from scipy import optimize

# TODO: change to add in reference as task positions
def overall_imitation_error(
    reference_sites: np.ndarray,
    robot_chain: Chain,
    robot_q: np.ndarray,
    num: int,
    alpha: float = 0
):
    """Computes the overall imitation error for a robot kinematic chain with 
    respect to a set of reference positions in task space.

    Args:
        reference_sites (np.ndarray): a list of reference positions (x,y,z)
        robot_chain (Chain): robot kinematic chain
        robot_q (np.ndarray): current robot joint configuration
        num (int): # of points sampled from the reference/robot trajectory 
        alpha (float, optional): Weight on imitating the end effector. Defaults to 0.

    Returns:
        tuple (float, dict): the error and a dict of each error component
    """
    s = np.linspace(0, 1, num=num)
    reference_path = Chain.compute_path_from_sites(reference_sites, s)
    robot_path = robot_chain.compute_path(robot_q, s)

    pose_error = np.mean(compute_path_error(reference_path, robot_path))
    ee_error   = compute_path_error(reference_path[[-1]], robot_path[[-1]])
    error = pose_error + alpha * ee_error
    return error, {
        'pose_error': pose_error,
        'ee_error':   ee_error
    }

def imitate(
    reference_sites: Chain,
    robot_chain: np.ndarray,
    q_init: np.ndarray,
    density=100,
    alpha=0.0
):
    def cost(x: np.ndarray):
        return overall_imitation_error(
            reference_sites,
            robot_chain,
            x,
            density,
            alpha
        )[0]
        
    sol = optimize.minimize(
        fun    = cost,
        x0     = q_init,
        method = 'Powell',
        options = dict(maxiter=500, xtol=1e-2, ftol=1e-2)
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
        reference_chain.compute_path(reference_q, s),
        actual_chain,
        actual_q,
    )
    print(sol)
