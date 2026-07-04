import numpy as np

def rotation_z(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,              0,             1]
    ])

def rotation_y(theta):
    return np.array([
        [ np.cos(theta), 0, np.sin(theta)],
        [ 0,             1, 0            ],
        [-np.sin(theta), 0, np.cos(theta)]
    ])

def forward_kinematics_6joint(angles, link_lengths):
    """
    angles: list of 6 joint angles (radians)
    link_lengths: list of 6 link lengths (meters) - simplified, real UR5 has offsets too
    Returns the XYZ position of the end-effector (gripper).
    """
    position = np.array([0.0, 0.0, 0.0])
    cumulative_rotation = np.eye(3)  # identity = no rotation yet

    for i in range(6):
        # Each joint rotates around alternating axes (simplified model)
        if i % 2 == 0:
            joint_rotation = rotation_z(angles[i])
        else:
            joint_rotation = rotation_y(angles[i])

        cumulative_rotation = cumulative_rotation @ joint_rotation
        segment_vector = cumulative_rotation @ np.array([link_lengths[i], 0, 0])
        position = position + segment_vector

    return position

if __name__ == "__main__":
    # All joints at 0 - arm fully extended
    angles = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    link_lengths = [0.1, 0.4, 0.35, 0.1, 0.1, 0.1]  # simplified approximate UR5-like lengths

    pos = forward_kinematics_6joint(angles, link_lengths)
    print(f"All joints at 0: X={pos[0]:.3f}, Y={pos[1]:.3f}, Z={pos[2]:.3f}")

    # Test: rotate the shoulder joint (joint 1) by 90 degrees
    angles2 = [1.5708, 0.0, 0.0, 0.0, 0.0, 0.0]
    pos2 = forward_kinematics_6joint(angles2, link_lengths)
    print(f"Shoulder rotated 90deg: X={pos2[0]:.3f}, Y={pos2[1]:.3f}, Z={pos2[2]:.3f}")s