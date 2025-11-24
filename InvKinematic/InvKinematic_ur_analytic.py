import numpy as np
from ur_analytic_ik import ur5e

np.set_printoptions(precision=4, suppress=True, floatmode='fixed')

# ---- Pos ----
p = np.array([0.3, 0.0, 0.2])

# ---- Rot ----
R = np.array([
    [0, 1, 0],
    [1, 0, 0],
    [0, 0, -1]
])

T = np.eye(4)
T[:3, 3] = p
T[:3,:3] = R

# add 0.1m extension along tool z-axis
T_tool = np.eye(4)
T_tool[2, 3] = 0.1

T_new = T @ T_tool

solutions = ur5e.inverse_kinematics(T_new)

elbow_up_solutions = [q for q in solutions if q[1] < 0]

print("Solutions found:", len(solutions))
for q in solutions:
    print(q)
    
print("Elbow-up only:", len(elbow_up_solutions))
for q in elbow_up_solutions:
    print(q)


final_solution = elbow_up_solutions[0]
print("Final solution:", final_solution)

T_check = ur5e.forward_kinematics(*final_solution)
p_check = T_check[:3, 3]
R_check = T_check[:3, :3]

print("\n--- FK Check ---")
print("FK position:", p_check)
print("FK rotation:\n", R_check)