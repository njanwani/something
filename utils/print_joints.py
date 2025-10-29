import mujoco
from pathlib import Path

# Load your model
xml = Path("xmls/scene.xml")

model = mujoco.MjModel.from_xml_string(xml.read_text())

print("Total joints:", model.njnt)
print("=" * 40)

# Loop through joints
for j in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
    print(f"Index {j + 7:2d} : {name}")
