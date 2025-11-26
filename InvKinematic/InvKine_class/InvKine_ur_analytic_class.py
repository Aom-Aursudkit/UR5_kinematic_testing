import numpy as np
from ur_analytic_ik import ur5e
import pinocchio as pin

class UR5eAnalyticIK:
    def __init__(self, urdf_path, tool_extension=0.0):
        self.tool_extension = tool_extension
                
        self.robot = pin.buildModelFromUrdf(urdf_path)
        self.data = self.robot.createData()

        # End-effector frame ID
        self.ee_name = "tool0" 
        self.ee_id = self.robot.getFrameId(self.ee_name)
        
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
    
    def jacobian(self, q):
        pin.computeJointJacobians(self.robot, self.data, q)
        pin.updateFramePlacements(self.robot, self.data)
        J = pin.getFrameJacobian(self.robot, self.data, self.ee_id, pin.ReferenceFrame.LOCAL)
        return J 

    def solve_joint_velocities(self, q, ee_velocity):
        J = self.jacobian(q)
        Jp = J[:3, :] 
        qdot = np.linalg.pinv(Jp) @ ee_velocity
        return qdot

    def solve_joint_accelerations(self, q, qdot, ee_acc, dt=1e-4):
        J = self.jacobian(q)

        q_next = q.copy()
        q_next += qdot * dt
        J_next = self.jacobian(q_next)

        Jdot = (J_next - J) / dt

        Jp = J[:3, :]
        Jpdot = Jdot[:3, :]

        qddot = np.linalg.pinv(Jp) @ (ee_acc - Jpdot @ qdot)
        return qddot

    def forward_kinematics(self, q):
        return ur5e.forward_kinematics(*q)


if __name__ == "__main__":
    
    urdf_path = "/home/aom/Documents/GitHub/UR5_kinematic_testing/InvKinematic/urdf/ur5e.urdf"
    
    # Input pos+ rot
    # p = np.array([0.3, 0.0, 0.2])
    p = np.array([0.0670, 0.0620, .0100])
    R = np.array([
        [0, 1, 0],
        [1, 0, 0],
        [0, 0,-1]
    ])

    # Create IK solver with 0.1 m tool extension
    ik = UR5eAnalyticIK(urdf_path, tool_extension=0.1)

    # Solve IK
    solutions, T_target = ik.solve_ik(p, R, elbow_up_only=True)

    print("Number of solutions:", len(solutions))
    for i, q in enumerate(solutions):
        print(f"Solution {i+1}:", q)

    final_q = solutions[0]
    print("\nFinal solution:", final_q)
    
    # Velocities
    ee_vel = np.array([0.1, 0.0, 0.0])
    qdot = ik.solve_joint_velocities(q, ee_vel)
    print("qdot =", qdot)

    # Accelerations
    ee_acc = np.array([0.2, 0.0, 0.0])
    qddot = ik.solve_joint_accelerations(q, qdot, ee_acc)
    print("qddot =", qddot)

    # Verify with FK
    T_fk = ik.forward_kinematics(final_q)
    p_fk = T_fk[:3, 3]
    R_fk = T_fk[:3, :3]

    print("\n--- FK Check ---")
    print("FK position:", p_fk)
    print("FK rotation:\n", R_fk)
    print("Position error:", np.linalg.norm(p_fk - T_target[:3, 3]))
