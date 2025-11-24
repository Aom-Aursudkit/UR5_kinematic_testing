import numpy as np
from ikpy.chain import Chain

class UR5eIK:
    def __init__(self, urdf_path, tool_offset=np.array([0.0, 0.0, 0.05])):
        self.urdf_path = urdf_path
        self.tool_offset = tool_offset
        self.robot_chain = Chain.from_urdf_file(urdf_path)

        # -----------------------------------------
        # Mark only revolute joints active
        # -----------------------------------------
        for i, link in enumerate(self.robot_chain.links):
            if link.bounds is not None:  # revolute joints have bounds
                self.robot_chain.active_links_mask[i] = True
            else:
                self.robot_chain.active_links_mask[i] = False

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

    def solve_fk(self, full_joint_vector):
        fk = self.robot_chain.forward_kinematics(full_joint_vector)
        return fk[:3, 3], fk[:3, :3]

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

    print("\nIK joint angles:")
    print(joint_angles)

    full_joint = np.zeros(9)
    full_joint[2:8] = joint_angles

    pos, rot = ik.solve_fk(full_joint)
    print("\nFK Position:", pos)
    print("FK Rotation:\n", rot)
