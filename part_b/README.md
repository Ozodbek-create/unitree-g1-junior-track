# Part B — Walking Policy

## What I changed

I changed the velocity command in `g1.yaml` from:

`[0.5, 0.0, 0.0]`

to:

`[0.4, 0.0, 0.7]`

The first value represents forward velocity, the second lateral velocity, and the third yaw angular velocity. By keeping a positive forward velocity and adding a nonzero yaw rate, the robot continuously moves forward while turning, which produces a circular trajectory.

## What inputs does the policy receive?

The policy receives a 47-dimensional observation vector containing:

- angular velocity
- gravity orientation
- commanded velocity
- joint positions relative to the default pose
- joint velocities
- previous actions
- gait phase represented by sine and cosine values

## What does the policy output?

The policy outputs 12 action values, corresponding to the 12 controlled leg joints.

The actions are converted to target joint positions using:

`target_dof_pos = action * action_scale + default_angles`

A PD controller then compares the target joint positions and velocities with the actual robot state and converts the difference into joint torques:

`tau = (target_q - q) * kp + (target_dq - dq) * kd`

These torques are then applied to the robot in MuJoCo.
