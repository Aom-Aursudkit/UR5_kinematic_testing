import numpy as np
import pandas as pd
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from ikpy.chain import Chain
import pinocchio as pin


class UR5eIK:
    def __init__(self, urdf_path, tool_offset=np.array([0.0, 0.0, 0.059])):
        self.urdf_path = urdf_path
        self.tool_offset = tool_offset

        # ===== IKPY for IK =====
        self.robot_chain = Chain.from_urdf_file(urdf_path)
        for i, link in enumerate(self.robot_chain.links):
            if link.bounds is not None:  # revolute joints
                self.robot_chain.active_links_mask[i] = True
            else:
                self.robot_chain.active_links_mask[i] = False

        # ===== Pinocchio for Jacobian/FK =====
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.ee_frame_name = "tool0"
        self.ee_id = self.model.getFrameId(self.ee_frame_name)

    def solve_ik(self, position, rotation=np.eye(3), initial_guess=None):
        position = np.array(position) + self.tool_offset
        target_frame = np.eye(4)
        target_frame[:3, :3] = rotation
        target_frame[:3, 3] = position

        if initial_guess is None:
            initial_guess = [
                0.0,
                0.0,
                0.0,
                -np.pi / 4,
                np.pi / 2,
                -np.pi / 2,
                np.pi / 2,
                0.0,
                0.0,
            ]

        joint_angles = self.robot_chain.inverse_kinematics_frame(
            target=target_frame, initial_position=initial_guess, orientation_mode="all"
        )

        return joint_angles[2:8]

    def jacobian(self, q):
        pin.forwardKinematics(self.model, self.data, q)
        pin.computeJointJacobians(self.model, self.data)
        pin.updateFramePlacements(self.model, self.data)
        J = pin.getFrameJacobian(
            self.model, self.data, self.ee_id, pin.ReferenceFrame.LOCAL
        )
        return J

    def solve_joint_velocities(self, joint_angles, ee_velocity):
        J = self.jacobian(joint_angles)
        J_pos = J[:3, :]
        qdot = np.linalg.pinv(J_pos) @ ee_velocity
        return qdot

    def solve_joint_accelerations(
        self, joint_angles, joint_velocities, ee_acc, dt=1e-3
    ):
        J = self.jacobian(joint_angles)
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


class JointSpacePublisher(Node):
    def __init__(self):
        super().__init__("joint_space_node")

        self.joint_pub = self.create_publisher(
            Float64MultiArray, "joint_desire_pos", 10
        )
        self.vel_pub = self.create_publisher(Float64MultiArray, "joint_desire_vel", 10)
        self.acc_pub = self.create_publisher(Float64MultiArray, "joint_desire_acc", 10)

        urdf_path = (
            "/home/aom/Documents/GitHub/UR5_kinematic_testing/InvKinematic/ur5e.urdf"
        )
        self.ik_solver = UR5eIK(urdf_path)

        df = pd.read_csv(
            "/home/aom/Documents/GitHub/UR5_kinematic_testing/InvKinematic/trajectory_fixed_dt_xyz.csv"
        )
        self.positions = df[["x", "y", "z"]].to_numpy()
        self.velocities = df[["vx", "vy", "vz"]].to_numpy()
        self.accelerations = df[["ax", "ay", "az"]].to_numpy()

        self.R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        self.idx = 0
        self.hold_steps = int(15.0 / 0.02)
        self.last_q = None
        self.P = np.array([0.0, 0.0, 0.0])  # current desired position

        self.timer = self.create_timer(0.02, self.timer_callback)

    def timer_callback(self):
        if self.idx < self.hold_steps:
            row = 0
            self.get_logger().info("hold")
        elif self.idx - self.hold_steps >= len(self.positions):
            row = len(self.positions) - 1
            self.get_logger().info("done")
        else:
            row = self.idx - self.hold_steps
            self.get_logger().info("run")

        self.P = self.positions[row]
        V = self.velocities[row]
        A = self.accelerations[row]

        # Use last_q as initial guess if available
        initial_guess = None
        if self.last_q is not None:
            initial_guess = np.zeros(9)
            initial_guess[2:8] = self.last_q

        chosen_q = self.ik_solver.solve_ik(self.P, self.R, initial_guess=initial_guess)
        self.last_q = chosen_q

        qdot = self.ik_solver.solve_joint_velocities(chosen_q, V)
        qddot = self.ik_solver.solve_joint_accelerations(chosen_q, qdot, A)

        msg = Float64MultiArray()
        msg.data = chosen_q.tolist()
        self.joint_pub.publish(msg)

        vel_msg = Float64MultiArray()
        vel_msg.data = qdot.tolist()
        self.vel_pub.publish(vel_msg)

        acc_msg = Float64MultiArray()
        acc_msg.data = qddot.tolist()
        self.acc_pub.publish(acc_msg)

        self.idx += 1


def main(args=None):
    rclpy.init(args=args)
    node = JointSpacePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
