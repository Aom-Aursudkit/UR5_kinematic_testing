import numpy as np

# --- UR5e parameters ---
d1 = 0.089159
a2 = -0.42500
a3 = -0.39225
d4 = 0.10915
d5 = 0.09465
d6 = 0.0823

# --- Your input ---
translation = np.array([0.3, 0.0, 0.2])
rotation_input = np.array([
    [0,  1,  0],
    [1,  0,  0],
    [0,  0, -1]
])

# Joint limits (example, adjust if needed)
joint_limits = [(-2*np.pi, 2*np.pi)]*6

# --- Helper functions ---
def normalize_angle(a):
    return (a + np.pi) % (2*np.pi) - np.pi

def rot_x(theta):
    c,s = np.cos(theta), np.sin(theta)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]])

def rot_z(theta):
    c,s = np.cos(theta), np.sin(theta)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])

# --- DH frame offset ---
# This rotates your input rotation into the robot's DH frame
R_offset = np.array([
    [-1, 0, 0],
    [0, -1, 0],
    [0, 0, 1]
])

# Adjust rotation to DH frame
rotation_DH = R_offset @ rotation_input

# --- Forward kinematics ---
def fk_from_thetas(t):
    th1,th2,th3,th4,th5,th6 = t
    def A(theta, d, a, alpha):
        c,s = np.cos(theta), np.sin(theta)
        ca,sa = np.cos(alpha), np.sin(alpha)
        return np.array([
            [c, -s*ca,  s*sa, a*c],
            [s,  c*ca, -c*sa, a*s],
            [0,   sa,    ca,    d],
            [0,    0,     0,    1],
        ])
    A1 = A(th1,d1,0,-np.pi/2)
    A2 = A(th2,0,a2,0)
    A3 = A(th3,0,a3,0)
    A4 = A(th4,d4,0,-np.pi/2)
    A5 = A(th5,d5,0,np.pi/2)
    A6 = A(th6,d6,0,0)
    return A1 @ A2 @ A3 @ A4 @ A5 @ A6

# --- Inverse kinematics ---
def inverse_kinematics(translation, rotation_matrix):
    R = rotation_matrix.copy()
    p = translation.copy()

    # Compute wrist center
    pwc = p - d6 * R[:,2]  # z-axis along tool
    xw, yw, zw = pwc

    sols = []

    # --- theta1 candidates ---
    r = np.hypot(xw, yw)
    if r < 1e-9:
        return sols
    gamma = np.arctan2(yw, xw)
    arg = d4 / r
    if abs(arg) <= 1.0:
        delta = np.arccos(arg)
        theta1_candidates = [gamma + delta + np.pi/2, gamma - delta + np.pi/2]
    else:
        theta1_candidates = [gamma + np.pi/2]

    for th1 in theta1_candidates:
        c1 = np.cos(th1)
        s1 = np.sin(th1)
        x1 = c1*xw + s1*yw
        z1 = zw - d1
        px = x1 - d4
        pz = z1

        # --- Solve theta2, theta3 using planar triangle ---
        D = (px**2 + pz**2 - a2**2 - a3**2)/(2*a2*a3)
        if abs(D) > 1+1e-9:
            continue
        D = np.clip(D,-1.0,1.0)
        th3_candidates = [np.arctan2(np.sqrt(1-D**2),D), np.arctan2(-np.sqrt(1-D**2),D)]

        for th3 in th3_candidates:
            k1 = a2 + a3*np.cos(th3)
            k2 = a3*np.sin(th3)
            th2 = np.arctan2(pz, px) - np.arctan2(k2, k1)

            # --- Compute R0_3 ---
            R03 = rot_z(th1) @ rot_x(-np.pi/2) @ rot_z(th2) @ rot_z(th3)
            R36 = R03.T @ R

            r11,r12,r13 = R36[0,0], R36[0,1], R36[0,2]
            r21,r22,r23 = R36[1,0], R36[1,1], R36[1,2]
            r31,r32,r33 = R36[2,0], R36[2,1], R36[2,2]

            th5 = np.arctan2(np.hypot(r13,r23), r33)
            if abs(np.sin(th5)) < 1e-6:
                th4 = 0.0
                th6 = np.arctan2(-r12, r11)
            else:
                th4 = np.arctan2(r23,r13)
                th6 = np.arctan2(-r32,r31)

            sol = np.array([th1,th2,th3,th4,th5,th6])
            sol = np.vectorize(normalize_angle)(sol)
            # filter by joint limits
            if all(joint_limits[i][0]-1e-6 <= sol[i] <= joint_limits[i][1]+1e-6 for i in range(6)):
                sols.append(sol)

    # remove duplicates
    unique=[]
    for s in sols:
        if not any(np.allclose(s,u,atol=1e-6) for u in unique):
            unique.append(s)
    return unique

# --- Example usage ---

sols = inverse_kinematics(translation, rotation_DH)
print(f"Found {len(sols)} solutions:")
for i, s in enumerate(sols):
    print(i, np.round(s,4))

# Verify FK of first solution
if sols:
    T_fk = fk_from_thetas(sols[0])
    print("FK from first solution:")
    np.set_printoptions(precision=4, suppress=True)
    print(T_fk)
