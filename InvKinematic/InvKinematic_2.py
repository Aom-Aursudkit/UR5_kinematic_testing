import numpy as np

# --------------------- Helper math functions ---------------------

def hat(k):
    return np.array([
        [0, -k[2], k[1]],
        [k[2], 0, -k[0]],
        [-k[1], k[0], 0]
    ])

def rot(k, theta):
    k = k / np.linalg.norm(k)
    K = hat(k)
    return np.eye(3) + np.sin(theta)*K + (1 - np.cos(theta))*(K @ K)

def wrapToPi(a):
    return (a + np.pi) % (2*np.pi) - np.pi

def rand_angle(n=1):
    return np.random.rand(n) * 2*np.pi - np.pi

# --------------------- Subproblem solvers ---------------------

def sp_1(p1, p2, k):
    KxP = np.cross(k, p1)
    A = np.column_stack((KxP, -np.cross(k, KxP)))
    x = A.T @ p2
    theta = np.arctan2(x[0], x[1])
    is_LS = abs(np.linalg.norm(p1) - np.linalg.norm(p2)) > 1e-8 or abs(np.dot(k,p1) - np.dot(k,p2)) > 1e-8
    return theta, is_LS

def sp_3(p1, p2, k, d):
    return sp_4(p2, p1, k, 0.5 * (np.dot(p1,p1) + np.dot(p2,p2) - d**2))

def sp_4(h, p, k, d):
    A_11 = np.cross(k, p)
    A_1 = np.column_stack((A_11, -np.cross(k, A_11)))
    A = h.T @ A_1
    b = d - h.T @ k * (k.T @ p)
    norm_A_2 = np.dot(A, A)
    x_ls_tilde = A_1.T @ (h * b)

    if norm_A_2 > b**2:
        xi = np.sqrt(norm_A_2 - b**2)
        x_N_prime_tilde = np.array([A[1], -A[0]])
        sc_1 = x_ls_tilde + xi * x_N_prime_tilde
        sc_2 = x_ls_tilde - xi * x_N_prime_tilde
        theta = np.array([np.arctan2(sc_1[0], sc_1[1]), np.arctan2(sc_2[0], sc_2[1])])
        is_LS = False
    else:
        theta = np.array([np.arctan2(x_ls_tilde[0], x_ls_tilde[1])])
        is_LS = True
    return theta, is_LS

# --------------------- Forward Kinematics ---------------------

def fwdkin(kin, theta):
    p = kin['P'][:,0]
    R = np.eye(3)
    for i, jt in enumerate(kin['joint_type']):
        if jt == 0 or jt == 2:  # rotational
            R = R @ rot(kin['H'][:,i], theta[i])
        elif jt == 1 or jt == 3:  # prismatic
            p = p + R @ kin['H'][:,i] * theta[i]
        p = p + R @ kin['P'][:,i+1]
    return R, p

# --------------------- Inverse Kinematics ---------------------

def IK_UR(R_06, p_0T, kin):
    P = kin['P']
    H = kin['H']
    Q = []
    is_LS_vec = []

    p_06 = p_0T - P[:,0] - R_06 @ P[:,6]
    theta1, ls1 = sp_4(H[:,1], p_06, -H[:,0], H[:,1].T @ np.sum(P[:,1:5], axis=1))

    for q1 in theta1:
        R_01 = rot(H[:,0], q1)
        theta5, ls5 = sp_4(H[:,1], H[:,5], H[:,4], H[:,1].T @ R_01.T @ R_06 @ H[:,5])

        for q5 in theta5:
            R_45 = rot(H[:,4], q5)
            theta14, ls14 = sp_1(R_45 @ H[:,5], R_01.T @ R_06 @ H[:,5], H[:,1])
            q6, ls6 = sp_1(R_45.T @ H[:,1], R_06.T @ R_01 @ H[:,1], -H[:,5])

            d_inner = R_01.T @ p_06 - P[:,1] - rot(H[:,1], theta14) @ P[:,4]
            d = np.linalg.norm(d_inner)
            theta3, ls3 = sp_3(-P[:,3], P[:,2], H[:,1], d)

            for q3 in theta3:
                q2, ls2 = sp_1(P[:,2] + rot(H[:,1], q3) @ P[:,3], d_inner, H[:,1])
                q4 = wrapToPi(theta14 - q2 - q3)

                q_sol = np.array([q1, q2, q3, q4, q5, q6])
                Q.append(q_sol)
                is_LS_vec.append(ls1 or ls5 or ls14 or ls3 or ls2 or ls6)

    return np.array(Q).T, np.array(is_LS_vec)

# --------------------- Example usage ---------------------

if __name__ == "__main__":
    ex = np.array([1,0,0])
    ey = np.array([0,1,0])
    ez = np.array([0,0,1])

    kin = {
        'H': np.column_stack((ez, -ey, -ey, -ey, -ez, -ey)),
        'P': np.column_stack((
            0.1625*ez, np.zeros(3), -0.425*ex, -0.3922*ex,
            -0.1333*ey-0.0997*ez, np.zeros(3), -0.0996*ey
        )),
        'joint_type': np.zeros(6)
    }

    q_test = np.deg2rad([10, 20, 30, 40, 50, 60])
    R_06, p_0T = fwdkin(kin, q_test)
    Q, is_ls = IK_UR(R_06, p_0T, kin)
    print("Original q:", q_test)
    print("IK solutions:\n", Q)
    print("Is LS:", is_ls)
