import numpy as np

np.set_printoptions(precision=4, suppress=True, floatmode='fixed', linewidth=120)

# ====== Helper functions ======
def rot(k, theta):
    k = k / np.linalg.norm(k)
    k_hat = np.array([
        [0, -k[2], k[1]],
        [k[2], 0, -k[0]],
        [-k[1], k[0], 0]
    ])
    return np.eye(3) + np.sin(theta)*k_hat + (1-np.cos(theta))*(k_hat @ k_hat)

def wrapToPi(a):
    return (a + np.pi) % (2*np.pi) - np.pi

# Subproblems
def sp_1(p1, p2, k):
    KxP = np.cross(k, p1)
    A = np.column_stack([KxP, -np.cross(k, KxP)])
    x = A.T @ p2
    theta = np.arctan2(x[0], x[1])
    is_LS = abs(np.linalg.norm(p1)-np.linalg.norm(p2))>1e-8 or abs(np.dot(k,p1)-np.dot(k,p2))>1e-8
    return theta, is_LS

def sp_4(h, p, k, d):
    A_11 = np.cross(k,p)
    A_1 = np.column_stack([A_11, -np.cross(k,A_11)])
    A = h.T @ A_1
    b = d - h.T @ k * (k.T @ p)
    norm_A_2 = A @ A
    x_ls_tilde = A_1.T @ (h*b)
    
    if norm_A_2 > b**2:
        xi = np.sqrt(norm_A_2 - b**2)
        x_N_prime_tilde = np.array([A[1], -A[0]])
        sc_1 = x_ls_tilde + x_N_prime_tilde*xi
        sc_2 = x_ls_tilde - x_N_prime_tilde*xi
        theta = np.array([np.arctan2(sc_1[0], sc_1[1]), np.arctan2(sc_2[0], sc_2[1])])
        is_LS = False
    else:
        theta = np.arctan2(x_ls_tilde[0], x_ls_tilde[1])
        is_LS = True
    return theta, is_LS

def sp_3(p1, p2, k, d):
    return sp_4(p2, p1, k, 0.5*(np.dot(p1,p1)+np.dot(p2,p2)-d**2))

# ====== Forward Kinematics ======
def fwdkin(kin, theta):
    p = kin.P[:,0].copy()
    R = np.eye(3)
    for i in range(len(kin.joint_type)):
        if kin.joint_type[i]==0 or kin.joint_type[i]==2:
            R = R @ rot(kin.H[:,i], theta[i])
        elif kin.joint_type[i]==1 or kin.joint_type[i]==3:
            p = p + R @ kin.H[:,i]*theta[i]
        p = p + R @ kin.P[:,i+1]
    return R, p

# ====== IK Function ======
def IK_UR(R_06, p_0T, kin):
    P = kin.P
    H = kin.H
    Q = []
    is_LS_vec = []
    p_06 = p_0T - P[:,0] - R_06 @ P[:,6]

    theta1, theta1_is_ls = sp_4(H[:,1], p_06, -H[:,0], H[:,1].T @ np.sum(P[:,1:5], axis=1))

    for q_1 in np.atleast_1d(theta1):
        R_01 = rot(H[:,0], q_1)
        theta5, theta5_is_ls = sp_4(H[:,1], H[:,5], H[:,4], H[:,1].T @ R_01.T @ R_06 @ H[:,5])
        for q_5 in np.atleast_1d(theta5):
            R_45 = rot(H[:,4], q_5)
            theta_14, theta_14_is_LS = sp_1(R_45 @ H[:,5], R_01.T @ R_06 @ H[:,5], H[:,1])
            q_6, q_6_is_LS = sp_1(R_45.T @ H[:,1], R_06.T @ R_01 @ H[:,1], -H[:,5])

            d_inner = R_01.T @ p_06 - P[:,1] - rot(H[:,1], theta_14) @ P[:,4]
            d = np.linalg.norm(d_inner)
            theta_3, theta_3_is_LS = sp_3(-P[:,3], P[:,2], H[:,1], d)

            for q_3 in np.atleast_1d(theta_3):
                q_2, q_2_is_LS = sp_1(P[:,2] + rot(H[:,1], q_3) @ P[:,3], d_inner, H[:,1])
                q_4 = wrapToPi(theta_14 - q_2 - q_3)

                q_i = np.array([q_1, q_2, q_3, q_4, q_5, q_6])
                Q.append(q_i)
                is_LS_vec.append(theta1_is_ls or theta5_is_ls or theta_14_is_LS or theta_3_is_LS or q_2_is_LS or q_6_is_LS)

    return np.column_stack(Q), is_LS_vec

# ====== Define Robot ======
class Kin:
    pass

# Unit vectors
zv = np.array([0.0, 0.0, 0.0])
ex = np.array([1.0, 0.0, 0.0])
ey = np.array([0.0, 1.0, 0.0])
ez = np.array([0.0, 0.0, 1.0])

kin = Kin()
kin.H = np.column_stack([ez, -ey, -ey, -ey, -ez, -ey])
kin.P = np.column_stack([0.1625*ez, zv, -0.425*ex, -0.3922*ex, -0.1333*ey-0.0997*ez, zv, -0.0996*ey])
kin.joint_type = np.zeros(6)

# ====== INPUT TARGET ======
p_target = np.array([0.3, 0.0, 0.2])
R_target = np.array([[0,1,0],[1,0,0],[0,0,-1]])

# ====== Compute IK ======
Q_solutions, is_LS_vec = IK_UR(R_target, p_target, kin)
print("Number of IK solutions found:", Q_solutions.shape[1])

# ====== Check FK ======
for i in range(Q_solutions.shape[1]):
    q_sol = Q_solutions[:,i]
    R_check, p_check = fwdkin(kin, q_sol)
    print(f"\nSolution {i+1}:")
    print("Joint angles (rad):", q_sol)
    print("FK position:", p_check)
    print("FK rotation:\n", R_check)
    print("Position error:", p_check - p_target)
    print("Rotation error:\n", R_check - R_target)
