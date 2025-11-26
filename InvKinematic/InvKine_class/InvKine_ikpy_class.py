import numpy as np
from ikpy.chain import Chain
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper

class UR5eIK:
    def __init__(self, urdf_path, tool_offset=np.array([0.0, 0.0, 0.059])):
        self.urdf_path = urdf_path
        self.tool_offset = tool_offset
        
        # ============ IKPY for IK ============
        self.robot_chain = Chain.from_urdf_file(urdf_path)
        # -----------------------------------------
        # Mark only revolute joints active
        # -----------------------------------------
        for i, link in enumerate(self.robot_chain.links):
            if link.bounds is not None:  # revolute joints have bounds
                self.robot_chain.active_links_mask[i] = True
            else:
                self.robot_chain.active_links_mask[i] = False
                
        # ============ PINOCCHIO for JACOBIAN/FK ============
        self.model = pin.buildModelFromUrdf(urdf_path)
        # self.model = self.robot.model
        # self.data = self.robot.data
        self.data = self.model.createData()

        # End-effector frame name
        self.ee_frame_name = "tool0"
        self.ee_id = self.model.getFrameId(self.ee_frame_name)

    def solve_ik(self, position, rotation=np.eye(3), initial_guess=None):
        # Apply tool offset
        position = np.array(position) + self.tool_offset

        # Build target frame
        target_frame = np.eye(4)
        target_frame[:3, :3] = rotation
        target_frame[:3, 3] = position

        # Default initial guess
        if initial_guess is None:
            initial_guess = [
                0.0, 0.0,
                0.0, -np.pi/4, np.pi/2,
                -np.pi/2, np.pi/2, 0.0,
                0.0
            ]

        # Solve IK
        joint_angles = self.robot_chain.inverse_kinematics_frame(
            target=target_frame,
            initial_position=initial_guess,
            orientation_mode="all"
        )

        return joint_angles[2:8]
    
    def jacobian(self, q):
        pin.forwardKinematics(self.model, self.data, q)
        pin.computeJointJacobians(self.model, self.data)
        pin.updateFramePlacements(self.model, self.data)

        J = pin.getFrameJacobian(self.model, self.data, self.ee_id, pin.ReferenceFrame.LOCAL)
        return J

    def solve_joint_velocities(self, joint_angles, ee_velocity):
        J = self.jacobian(joint_angles)
        J_pos = J[:3, :]              # position part only
        qdot = np.linalg.pinv(J_pos) @ ee_velocity
        return qdot

    def solve_joint_accelerations(self, joint_angles, joint_velocities, ee_acc, dt=1e-3):
        J = self.jacobian(joint_angles)

        # numerical Jdot
        q_next = joint_angles + joint_velocities * dt
        J_next = self.jacobian(q_next)
        Jdot = (J_next - J) / dt

        J_pos = J[:3, :]
        Jdot_pos = Jdot[:3, :]

        qddot = np.linalg.pinv(J_pos) @ (ee_acc - Jdot_pos @ joint_velocities)
        return qddot

    def solve_fk(self, full_joint_vector):
        fk = self.robot_chain.forward_kinematics(full_joint_vector)
        return fk[:3, 3], fk[:3, :3]
    
    # def solve_fk(self, q):
    #     pin.forwardKinematics(self.model, self.data, q)
    #     pin.updateFramePlacements(self.model, self.data)

    #     T = self.data.oMf[self.ee_id]
    #     pos = T.translation
    #     rot = T.rotation
    #     return pos, rot

    def debug_links(self):
        print("\n=== LINK LIST ===")
        for i, link in enumerate(self.robot_chain.links):
            print(i, link.name,
                  "(fixed =", getattr(link, "is_fixed", None), ")")

        print("\nActive links mask:", self.robot_chain.active_links_mask)


if __name__ == "__main__":
    urdf_path = "/home/aom/Documents/GitHub/UR5_kinematic_testing/InvKinematic/urdf/ur5e.urdf"
    ik = UR5eIK(urdf_path)

    target_pos = [0.4, 0.4, 0.05]
    target_rot = np.eye(3)

    joint_angles = ik.solve_ik(target_pos, target_rot)
    print("\nIK joint angles:", joint_angles)

    ee_vel = np.array([0.1, 0.0, 0.0])
    qdot = ik.solve_joint_velocities(joint_angles, ee_vel)
    print("Joint velocities:", qdot)

    ee_acc = np.array([0.2, 0.0, 0.0])
    qddot = ik.solve_joint_accelerations(joint_angles, qdot, ee_acc)
    print("Joint accelerations:", qddot)

    full_joint = np.zeros(9)
    full_joint[2:8] = joint_angles

    pos, rot = ik.solve_fk(full_joint)
    print("\nFK Position:", pos)
    print("FK Rotation:\n", rot)
