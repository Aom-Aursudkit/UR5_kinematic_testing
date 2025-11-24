import numpy as np
from ur_analytic_ik import ur5e


class UR5eAnalyticIK:
    def __init__(self, tool_extension=0.0):
        self.tool_extension = tool_extension
        
        np.set_printoptions(precision=4, suppress=True, floatmode='fixed')

    def build_transform(self, p, R):
        T = np.eye(4)
        T[:3, 3] = p
        T[:3, :3] = R
        return T

    def apply_tool_extension(self, T):
        if self.tool_extension == 0.0:
            return T

        T_tool = np.eye(4)
        T_tool[2, 3] = self.tool_extension
        return T @ T_tool

    def solve_ik(self, p, R, elbow_up_only=True):
        T = self.build_transform(p, R)
        T_ext = self.apply_tool_extension(T)

        # Solve IK
        solutions = ur5e.inverse_kinematics(T_ext)

        # Filter elbow-up (q[1] or q[2] idk pls check)
        if elbow_up_only:
            solutions = [q for q in solutions if q[1] < 0]

        return solutions, T_ext

    def forward_kinematics(self, q):
        return ur5e.forward_kinematics(*q)


if __name__ == "__main__":
    # Input pos+ rot
    p = np.array([0.3, 0.0, 0.2])
    R = np.array([
        [0, 1, 0],
        [1, 0, 0],
        [0, 0,-1]
    ])

    # Create IK solver with 0.1 m tool extension
    ik = UR5eAnalyticIK(tool_extension=0.1)

    # Solve IK
    solutions, T_target = ik.solve_ik(p, R, elbow_up_only=True)

    print("Number of solutions:", len(solutions))
    for i, q in enumerate(solutions):
        print(f"Solution {i+1}:", q)

    final_q = solutions[0]
    print("\nFinal solution:", final_q)

    # Verify with FK
    T_fk = ik.forward_kinematics(final_q)
    p_fk = T_fk[:3, 3]
    R_fk = T_fk[:3, :3]

    print("\n--- FK Check ---")
    print("FK position:", p_fk)
    print("FK rotation:\n", R_fk)
    print("Position error:", np.linalg.norm(p_fk - T_target[:3, 3]))
