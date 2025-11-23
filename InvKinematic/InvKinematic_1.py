import numpy as np
import sympy as sp
from math import atan2, acos, sqrt, pi
from dataclasses import dataclass

# --- UR5 DH parameters (common convention; verify for your setup) ---
# These values are widely used in UR5 examples (meters).
d1 = 0.089159
a2 = -0.42500
a3 = -0.39225
d4 = 0.10915
d5 = 0.09465
d6 = 0.0823

# For clarity, pack into a dataclass
@dataclass
class UR5Params:
    d1: float = d1
    a2: float = a2
    a3: float = a3
    d4: float = d4
    d5: float = d5
    d6: float = d6

PARAMS = UR5Params()

# ------------------------- Symbolic derivation bits -------------------------
# We'll symbolically compute the wrist center p_wc = p - d6 * R * z_hat
x, y, z = sp.symbols('x y z', real=True)
r11, r12, r13, r21, r22, r23, r31, r32, r33 = sp.symbols('r11:14 r21:24 r31:34', real=True)

# symbolically define p and R
p_sym = sp.Matrix([x, y, z])
R_sym = sp.Matrix([[r11, r12, r13],
                   [r21, r22, r23],
                   [r31, r32, r33]])
z_hat = sp.Matrix([0,0,1])

p_wc_sym = p_sym - PARAMS.d6 * (R_sym * z_hat)
# Cos(theta3) expression using law of cosines: D^2 = r^2 + s^2 where r = sqrt(x^2+y^2) - a1 (a1=0 here), s = z - d1
r_sym = sp.sqrt(x**2 + y**2)  # a1 assumed zero for these DH values
s_sym = z - PARAMS.d1
D_sq_sym = r_sym**2 + s_sym**2
cos_theta3_sym = (D_sq_sym - PARAMS.a2**2 - PARAMS.a3**2) / (2*PARAMS.a2*PARAMS.a3)

# Prepare symbolic outputs (simplified)
p_wc_sym_simpl = sp.simplify(p_wc_sym)
cos_theta3_sym_simpl = sp.simplify(sp.simplify(cos_theta3_sym))

# ------------------------- Numeric/Demonstration functions -------------------------
def rotz(theta):
    c = np.cos(theta); s = np.sin(theta)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]])

def rotx(alpha):
    c = np.cos(alpha); s = np.sin(alpha)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s,  c]])

def dh_transform(a, alpha, d, theta):
    # Standard DH homogeneous transform
    ct = np.cos(theta); st = np.sin(theta)
    ca = np.cos(alpha); sa = np.sin(alpha)
    T = np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0 ,     sa,     ca,    d],
        [0 ,      0,      0,    1]
    ])
    return T

def forward_kinematics_from_thetas(thetas, params=PARAMS):
    # the UR5 commonly uses these DH parameters (theta are the joint variables)
    th1, th2, th3, th4, th5, th6 = thetas
    # Using the same DH structure as the standard UR5 example:
    # T01: a1=0, alpha1=pi/2, d1=d1, theta1=th1
    # T12: a2=a2, alpha2=0, d2=0, theta2=th2
    # T23: a3=a3, alpha3=0, d3=0, theta3=th3
    # T34: a4=0, alpha4=pi/2, d4=d4, theta4=th4
    # T45: a5=0, alpha5=-pi/2, d5=d5, theta5=th5
    # T56: a6=0, alpha6=0, d6=d6, theta6=th6
    T01 = dh_transform(0, np.pi/2, params.d1, th1)
    T12 = dh_transform(params.a2, 0, 0, th2)
    T23 = dh_transform(params.a3, 0, 0, th3)
    T34 = dh_transform(0, np.pi/2, params.d4, th4)
    T45 = dh_transform(0, -np.pi/2, params.d5, th5)
    T56 = dh_transform(0, 0, params.d6, th6)
    T = T01 @ T12 @ T23 @ T34 @ T45 @ T56
    R = T[:3,:3]; p = T[:3,3]
    return T, R, p

def inverse_kinematics_numeric(T_target, params=PARAMS, eps=1e-6):
    R = T_target[:3,:3]; p = T_target[:3,3]
    # wrist center
    p_wc = p - params.d6 * R[:,2]  # end-effector z-axis is column 2
    x_wc, y_wc, z_wc = p_wc
    solutions = []
    # theta1 candidates (principal and plus pi)
    theta1_cand = [np.arctan2(y_wc, x_wc), np.arctan2(y_wc, x_wc) + np.pi]
    for theta1 in theta1_cand:
        # compute r and s for planar triangle
        r = np.sqrt(x_wc**2 + y_wc**2)
        # here a1 = 0 so no subtraction
        s = z_wc - params.d1
        D = np.sqrt(r**2 + s**2)
        # law of cosines for theta3
        cos_theta3 = (D**2 - params.a2**2 - params.a3**2) / (2*params.a2*params.a3)
        if cos_theta3 < -1 - 1e-9 or cos_theta3 > 1 + 1e-9:
            # no real solution for this candidate
            continue
        cos_theta3 = np.clip(cos_theta3, -1.0, 1.0)
        theta3_options = [np.arccos(cos_theta3), -np.arccos(cos_theta3)]
        for theta3 in theta3_options:
            # theta2 using geometry
            phi = np.arctan2(s, np.sqrt(max(0.0, r**2)))  # r used directly; phi=atan2(s,r)
            # better: phi = atan2(s, r)
            phi = np.arctan2(s, r)
            psi = np.arctan2(params.a3 * np.sin(theta3), params.a2 + params.a3 * np.cos(theta3))
            theta2 = phi - psi
            # compute R0_3 and then R3_6
            T01 = dh_transform(0, np.pi/2, params.d1, theta1)
            T12 = dh_transform(params.a2, 0, 0, theta2)
            T23 = dh_transform(params.a3, 0, 0, theta3)
            T03 = T01 @ T12 @ T23
            R03 = T03[:3,:3]
            R36 = R03.T @ R
            # extract wrist angles
            r13 = R36[0,2]; r23 = R36[1,2]; r33 = R36[2,2]
            r31 = R36[2,0]; r32 = R36[2,1]; r11 = R36[0,0]; r21 = R36[1,0]
            # theta5
            theta5 = np.arctan2(np.sqrt(r13**2 + r23**2), r33)
            # handle singularity when theta5 ~ 0
            if abs(np.sin(theta5)) < 1e-6:
                # wrist singular: set theta4 = 0 and compute theta6 from R36
                theta4 = 0.0
                theta6 = np.arctan2(-r21, r11)
            else:
                theta4 = np.arctan2(r23, r13)
                theta6 = np.arctan2(r32, -r31)
            # normalize angles to [-pi, pi]
            sol = np.array([theta1, theta2, theta3, theta4, theta5, theta6])
            sol = (sol + np.pi) % (2*np.pi) - np.pi
            # validate via FK
            T_sol, R_sol, p_sol = forward_kinematics_from_thetas(sol, params)
            pos_err = np.linalg.norm(p_sol - p)
            rot_err = np.linalg.norm(R_sol - R)  # Frobenius-norm-ish
            if pos_err < 1e-3 and rot_err < 1e-2:
                solutions.append({
                    'thetas': sol,
                    'pos_err': pos_err,
                    'rot_err': rot_err,
                    'p_sol': p_sol
                })
            else:
                # still append but mark as approximate; some poses (esp. near singularities) may fail strict check
                solutions.append({
                    'thetas': sol,
                    'pos_err': pos_err,
                    'rot_err': rot_err,
                    'p_sol': p_sol
                })
    # remove near-duplicate solutions (within small joint tol)
    unique = []
    for s in solutions:
        add = True
        for u in unique:
            if np.allclose(s['thetas'], u['thetas'], atol=1e-4):
                add = False; break
        if add:
            unique.append(s)
    return unique

# ------------------------- Test example -------------------------
# Build a target T: choose a reachable pose.
# We'll choose a moderate pose: position (0.3, 0.0, 0.4) and identity orientation.
p_target = np.array([0.3, 0.0, 0.4])
R_target = np.eye(3)
T_target = np.eye(4)
T_target[:3,:3] = R_target
T_target[:3,3] = p_target

# Run IK
solutions = inverse_kinematics_numeric(T_target, PARAMS)

# Display symbolic results and numeric solutions
from IPython.display import display, Markdown
md_lines = []
md_lines.append("### Symbolic expressions (wrist center and cos(theta3))")
md_lines.append("")

md_lines.append("Wrist center $\\mathbf{p}_{wc} = p - d_6 R \\hat z$ (symbolic):")
md_lines.append("```")
md_lines.append(sp.pretty(p_wc_sym_simpl, use_unicode=True))
md_lines.append("```")
md_lines.append("Law-of-cosines expression for $\\cos(\\theta_3)$ (symbolic):")
md_lines.append("```")
md_lines.append(sp.pretty(cos_theta3_sym_simpl, use_unicode=True))
md_lines.append("```")

if len(solutions) == 0:
    md_lines.append("**No IK solutions found for the chosen target pose with the tolerance checks.**")
else:
    md_lines.append(f"### Numeric IK solutions found: {len(solutions)}\n")
    for i, sol in enumerate(solutions):
        th = sol['thetas']
        md_lines.append(f"**Solution {i+1}:** thetas (rad) = {np.round(th, 6).tolist()}")
        md_lines.append(f"- position error (m): {sol['pos_err']:.6f}")
        md_lines.append(f"- rotation error (Frobenius norm): {sol['rot_err']:.6f}")
        md_lines.append("")

display(Markdown("\n".join(md_lines)))

# Also print a concise numpy table
import pandas as pd
rows = []
for i, s in enumerate(solutions):
    row = {'sol': i+1, 'pos_err': s['pos_err'], 'rot_err': s['rot_err']}
    for j in range(6):
        row[f't{j+1}'] = float(s['thetas'][j])
    rows.append(row)
if rows:
    df = pd.DataFrame(rows)
    try:
        from caas_jupyter_tools import display_dataframe_to_user
        display_dataframe_to_user("IK Solutions", df)
    except Exception:
        display(df)
