import os
import numpy as np
import mujoco
import imageio

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CSV_PATH = os.path.join(ROOT, "submission", "part_a", "motion.csv")
XML_PATH = os.path.join(
    ROOT,
    "resources",
    "robots",
    "g1_description",
    "g1_29dof.xml"
)

OUTPUT_PATH = os.path.join(ROOT, "submission", "part_a", "motion.mp4")

# Load motion
motion = np.loadtxt(CSV_PATH, delimiter=",")

print("Motion shape:", motion.shape)

# Load G1 model
model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

print("MuJoCo nq:", model.nq)

if motion.shape[1] != model.nq:
    raise ValueError(
        f"CSV has {motion.shape[1]} columns, but model expects {model.nq} qpos values."
    )

# Renderer
renderer = mujoco.Renderer(
    model,
    height=480,
    width=640,
)

camera = mujoco.MjvCamera()
mujoco.mjv_defaultFreeCamera(model, camera)

camera.lookat[:] = [0.0, 0.0, 0.8]
camera.distance = 3.0
camera.azimuth = 135
camera.elevation = -20

frames = []

# UltimateBots export has 300 frames.
# Assume 30 FPS initially unless the source/export specifies otherwise.
fps = 30

for i, row in enumerate(motion):
    qpos = row.copy()

    # UltimateBots/LAFAN CSV quaternion order:
    # [qx, qy, qz, qw]
    #
    # MuJoCo free-joint quaternion order:
    # [qw, qx, qy, qz]
    qpos[3:7] = row[[6, 3, 4, 5]]

    data.qpos[:] = qpos

    # Recompute all body positions/orientations after setting qpos
    mujoco.mj_forward(model, data)

    renderer.update_scene(data, camera=camera)
    frame = renderer.render()
    frames.append(frame.copy())

renderer.close()

imageio.mimsave(
    OUTPUT_PATH,
    frames,
    fps=fps
)

print(f"Saved video to: {OUTPUT_PATH}")
