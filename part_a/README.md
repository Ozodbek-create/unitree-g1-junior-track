# Part A — Human Motion to Robot

## Source motion

The motion was generated in UltimateBots Studio from a text prompt and exported as a LAFAN CSV file.

## How I determined what the CSV columns represent

The exported CSV contains 36 columns per frame.

The Unitree G1 29-DOF MuJoCo model also has 36 generalized position values (`qpos`):

- 3 values for the floating-base position
- 4 values for the floating-base quaternion
- 29 values for the robot joint positions

This matched the structure of the CSV exactly.

I also checked the quaternion values and found that the exported file uses the order:

`[qx, qy, qz, qw]`

while MuJoCo expects:

`[qw, qx, qy, qz]`

So I reordered the quaternion components before assigning each row of the CSV to the MuJoCo `qpos` vector.

The remaining 29 values correspond to the 29 actuated joints of the G1 model in the same model order.

## Playback

For each frame of the CSV:

1. The root position, reordered root quaternion, and 29 joint values are assigned to the G1 MuJoCo state.
2. `mujoco.mj_forward()` recomputes the robot pose.
3. The frame is rendered using MuJoCo.
4. The rendered frames are saved as an MP4 video.
