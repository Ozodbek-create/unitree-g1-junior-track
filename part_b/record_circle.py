import mujoco
import numpy as np
import torch
import yaml
import imageio

from legged_gym import LEGGED_GYM_ROOT_DIR


def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd


# Load config
config_path = f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_mujoco/configs/g1.yaml"

with open(config_path, "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

policy_path = config["policy_path"].replace(
    "{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR
)
xml_path = config["xml_path"].replace(
    "{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR
)

simulation_dt = config["simulation_dt"]
control_decimation = config["control_decimation"]

kps = np.array(config["kps"], dtype=np.float32)
kds = np.array(config["kds"], dtype=np.float32)

default_angles = np.array(config["default_angles"], dtype=np.float32)

ang_vel_scale = config["ang_vel_scale"]
dof_pos_scale = config["dof_pos_scale"]
dof_vel_scale = config["dof_vel_scale"]
action_scale = config["action_scale"]
cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

num_actions = config["num_actions"]
num_obs = config["num_obs"]

cmd = np.array(config["cmd_init"], dtype=np.float32)

# Simulation length for submission video
video_duration = 20.0
fps = 30

# State
action = np.zeros(num_actions, dtype=np.float32)
target_dof_pos = default_angles.copy()
obs = np.zeros(num_obs, dtype=np.float32)

counter = 0

# Load MuJoCo model
m = mujoco.MjModel.from_xml_path(xml_path)
d = mujoco.MjData(m)
m.opt.timestep = simulation_dt

# Load pretrained policy
policy = torch.jit.load(policy_path)

# Offscreen renderer
renderer = mujoco.Renderer(
    m,
    height=480,
    width=640,
)

# Fixed camera so the circular path is visible
camera = mujoco.MjvCamera()
mujoco.mjv_defaultFreeCamera(m, camera)

camera.lookat[:] = [0.0, 0.0, 0.6]
camera.distance = 4.0
camera.azimuth = 90
camera.elevation = -45

frames = []

total_steps = int(video_duration / simulation_dt)
# At 0.002 s timestep, this records approximately 30 fps
frame_interval = max(1, int(1.0 / (fps * simulation_dt)))

for step in range(total_steps):

    # PD controller converts desired joint positions into torques
    tau = pd_control(
        target_dof_pos,
        d.qpos[7:],
        kps,
        np.zeros_like(kds),
        d.qvel[6:],
        kds,
    )

    d.ctrl[:] = tau

    mujoco.mj_step(m, d)

    counter += 1

    # Run policy at control frequency
    if counter % control_decimation == 0:

        qj = d.qpos[7:]
        dqj = d.qvel[6:]
        quat = d.qpos[3:7]
        omega = d.qvel[3:6]

        qj = (qj - default_angles) * dof_pos_scale
        dqj = dqj * dof_vel_scale

        gravity_orientation = get_gravity_orientation(quat)
        omega = omega * ang_vel_scale

        period = 0.8
        count = counter * simulation_dt
        phase = count % period / period

        sin_phase = np.sin(2 * np.pi * phase)
        cos_phase = np.cos(2 * np.pi * phase)

        obs[:3] = omega
        obs[3:6] = gravity_orientation
        obs[6:9] = cmd * cmd_scale

        obs[9:9 + num_actions] = qj

        obs[
            9 + num_actions:
            9 + 2 * num_actions
        ] = dqj

        obs[
            9 + 2 * num_actions:
            9 + 3 * num_actions
        ] = action

        obs[
            9 + 3 * num_actions:
            9 + 3 * num_actions + 2
        ] = np.array([sin_phase, cos_phase])

        obs_tensor = torch.from_numpy(obs).unsqueeze(0)

        with torch.no_grad():
            action = policy(obs_tensor).detach().numpy().squeeze()

        # Convert policy action to joint-position targets
        target_dof_pos = action * action_scale + default_angles

    # Capture video frame
    if step % frame_interval == 0:
        renderer.update_scene(d, camera=camera)
        frame = renderer.render()
        frames.append(frame.copy())


output_path = "g1_circle.mp4"

imageio.mimsave(
    output_path,
    frames,
    fps=fps,
)

renderer.close()

print(f"Saved video to: {output_path}")
