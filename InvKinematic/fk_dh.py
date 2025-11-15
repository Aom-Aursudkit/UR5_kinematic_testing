import numpy as np
np.set_printoptions(precision=3, suppress=True, floatmode='fixed', linewidth=120)

# DH parameters: [a, s, alpha]
# alpha in radians, distances in meters
dh_params = [
    [0, 0.1625, np.pi/2],   # Joint 1
    [-0.425, 0, 0],         # Joint 2
    [-0.3922, 0, 0],        # Joint 3
    [0, 0.1333, np.pi/2],   # Joint 4
    [0, 0.0997, -np.pi/2],  # Joint 5
    [0, 0.0996, 0]          # Joint 6
]

# DH Transformation Matrix
def dh_matrix(theta, a, d, alpha):
    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    return np.array([
        [ct, -st*ca, st*sa, a*ct],
        [st, ct*ca, -ct*sa, a*st],
        [0, sa, ca, d],
        [0, 0, 0, 1]
    ])

# Forward Kinematics
def fk_ur5(thetas, dh_params):
    """
    thetas: list of 6 joint angles [θ1, θ2, ..., θ6]
    dh_params: list of 6 DH parameters [d, a, alpha]
    """
    T = np.eye(4)
    for i in range(6):
        a, d, alpha = dh_params[i]
        T_i = dh_matrix(thetas[i], a, d, alpha)
        T = T @ T_i  # Multiply successive transforms
    
    R = T[:3, :3]  # Rotation
    p = T[:3, 3]   # Position
    return R, p

# Example
thetas = [2.8015,  1.1737, -2.1724, -2.1430,  1.9109,  0.0000]
R, p = fk_ur5(thetas, dh_params)
print("FK position:", p)
print("FK rotation:\n", R)
