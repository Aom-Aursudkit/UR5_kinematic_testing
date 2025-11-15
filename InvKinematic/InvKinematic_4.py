import numpy as np

# ======================================================================
# Basic vector utilities
# ======================================================================
zv = np.zeros((3,1))
ex = np.array([[1],[0],[0]])
ey = np.array([[0],[1],[0]])
ez = np.array([[0],[0],[1]])

# ======================================================================
# Rotation utilities
# ======================================================================
def hat(k):
    k = k.flatten()
    return np.array([
        [0, -k[2], k[1]],
        [k[2], 0, -k[0]],
        [-k[1], k[0], 0]
    ])

def rot(k, theta):
    k = k.flatten() / np.linalg.norm(k)
    K = hat(k)
    return np.eye(3) + np.sin(theta)*K + (1-np.cos(theta))*(K @ K)

def wrapToPi(a):
    return (a + np.pi) % (2*np.pi) - np.pi

def rand_angle(n):
    return (np.random.rand(n)*2*np.pi - np.pi).reshape((n,1))

# ======================================================================
# Subproblems (SP1, SP3, SP4)
# ======================================================================
def sp_1(p1, p2, k):
    KxP = np.cross(k, p1, axis=0)
    A = np.hstack([KxP, -np.cross(k, KxP, axis=0)])
    x = A.T @ p2
    theta = np.arctan2(x[0], x[1])

    is_LS = (abs(np.linalg.norm(p1)-np.linalg.norm(p2)) > 1e-8 or
             abs((k.T @ p1).item() - (k.T @ p2).item()) > 1e-8)
    return theta.item(), is_LS

def sp_3(p1, p2, k, d):
    return sp_4(p2, p1, k, 0.5*((p1.T@p1).item() + (p2.T@p2).item() - d**2))

def sp_4(h, p, k, d):
    A11 = np.cross(k, p, axis=0)
    A1 = np.hstack([A11, -np.cross(k, A11, axis=0)])
    A = h.T @ A1
    b = d - (h.T @ k * (k.T @ p)).item()
    normA2 = (A @ A.T).item()
    x_ls = A1.T @ (h*b)

    if normA2 > b**2:
        xi = np.sqrt(normA2 - b**2)
        xN = np.array([[A[0,1]], [-A[0,0]]])

        sc1 = x_ls + xi*xN
        sc2 = x_ls - xi*xN

        th1 = np.arctan2(sc1[0].item(), sc1[1].item())
        th2 = np.arctan2(sc2[0].item(), sc2[1].item())
        return np.array([th1, th2]), False
    else:
        th = np.arctan2(x_ls[0].item(), x_ls[1].item())
        return np.array([th]), True

# ======================================================================
# Forward kinematics
# ======================================================================
def fwdkin(kin, theta):
    P = kin['P']
    H = kin['H']
    joint_type = kin['joint_type']

    p = P[:,[0]]
    R = np.eye(3)

    for i in range(len(joint_type)):
        if joint_type[i] in [0,2]:      # rotational
            R = R @ rot(H[:,[i]], theta[i])
        else:                           # prismatic
            p = p + R @ (H[:,[i]]*theta[i])
        p = p + R @ P[:,[i+1]]

    return R, p

# ======================================================================
# UR5 analytic IK
# ======================================================================
def IK_UR(R_06, p_0T, kin):
    P = kin['P']
    H = kin['H']

    Q = []
    is_LS_vec = []

    p_06 = p_0T - P[:,[0]] - R_06 @ P[:,[6]]

    # q1 using SP4
    theta1, ls1 = sp_4(H[:,[1]], p_06, -H[:,[0]], H[:,[1]].T @ np.sum(P[:,1:5], axis=1, keepdims=True))

    for q1 in theta1:
        R_01 = rot(H[:,[0]], q1)

        # q5
        theta5, ls5 = sp_4(H[:,[1]], H[:,[5]], H[:,[4]], H[:,[1]].T @ (R_01.T @ R_06 @ H[:,[5]]))

        for q5 in theta5:
            R_45 = rot(H[:,[4]], q5)

            # theta_14
            theta14, ls14 = sp_1(R_45 @ H[:,[5]], R_01.T @ R_06 @ H[:,[5]], H[:,[1]])

            # q6
            q6, ls6 = sp_1(R_45.T @ H[:,[1]], R_06.T @ R_01 @ H[:,[1]], -H[:,[5]])

            # q3 using SP3
            d_inner = R_01.T @ p_06 - P[:,[1]] - rot(H[:,[1]], theta14) @ P[:,[4]]
            d = np.linalg.norm(d_inner)
            q3_list, ls3 = sp_3(-P[:,[3]], P[:,[2]], H[:,[1]], d)

            for q3 in q3_list:
                # q2
                q2, ls2 = sp_1(P[:,[2]] + rot(H[:,[1]], q3) @ P[:,[3]], d_inner, H[:,[1]])

                # q4
                q4 = wrapToPi(theta14 - q2 - q3)

                q = np.array([float(q1), float(q2), float(q3), float(q4), float(q5), float(q6)]).reshape((6,1))
                Q.append(q)
                is_LS_vec.append(ls1 or ls5 or ls14 or ls3 or ls2 or ls6)

    return Q, is_LS_vec

# ======================================================================
# Define UR5 kinematics
# ======================================================================
kin = {
    "H": np.hstack([ez, -ey, -ey, -ey, -ez, -ey]),
    "P": np.hstack([
        0.1625*ez, zv, -0.425*ex,
        -0.3922*ex, -0.1333*ey - 0.0997*ez,
        zv, -0.0996*ey
    ]),
    "joint_type": np.zeros((6,), dtype=int)
}

# ======================================================================
# User
# ======================================================================
# Random joint angles
# q = rand_angle(6)
# R_06, p_0T = fwdkin(kin, q)

p_0T = np.array([[0.3], [0.0], [0.2]])

R_06 = np.array([
    [0,  1,  0],
    [1,  0,  0],
    [0,  0, -1]
])

# --------------------------
# Compute IK
# --------------------------
Q_solutions, ls_flags = IK_UR(R_06, p_0T, kin)

# Print IK solutions
print("IK joint solutions (radians):")
for i, q_sol in enumerate(Q_solutions):
    print(f"Solution {i+1}: {q_sol.flatten()}")

# --------------------------
# Verify by FK
# --------------------------
print("\nFK verification (should match input p and R):")
for i, q_sol in enumerate(Q_solutions):
    R_check, p_check = fwdkin(kin, q_sol)
    print(f"Solution {i+1}:")
    print("FK position:\n", p_check.flatten())
    print("FK rotation:\n", R_check)
