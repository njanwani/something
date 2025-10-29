import mujoco

# Load your model
xml_path = "something/xmls/humanoid/humanoid.xml"
model = mujoco.MjModel.from_xml_path(xml_path)

print("Total joints:", model.njnt)
print("=" * 40)

# Loop through joints
for j in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
    print(f"Index {j + 7:2d} : {name}")
