import mujoco


def create_name2idx(model):
    name2idx = {}
    qpos_idx = 0  # running index into qpos

    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        jtype = model.jnt_type[j]

        if jtype == mujoco.mjtJoint.mjJNT_FREE:
            # free joint: 7 qpos values (x, y, z, qw, qx, qy, qz)
            name2idx[f"{name}_xyz"] = list(range(qpos_idx, qpos_idx + 3))
            name2idx[f"{name}_xyzw"] = list(range(qpos_idx + 3, qpos_idx + 7))
            qpos_idx += 7
        elif jtype == mujoco.mjtJoint.mjJNT_BALL:
            # ball joint: 4 qpos values (quaternion)
            name2idx[name] = list(range(qpos_idx, qpos_idx + 4))
            qpos_idx += 4
        elif jtype in [mujoco.mjtJoint.mjJNT_SLIDE, mujoco.mjtJoint.mjJNT_HINGE]:
            # hinge/slide joint: 1 qpos value
            name2idx[name] = qpos_idx
            qpos_idx += 1
        else:
            raise ValueError(f"Unknown joint type {jtype} for joint {name}")

    return name2idx


if __name__ == '__main__':
    # Load your model
    xml_path = "something/xmls/humanoid/humanoid.xml"
    model = mujoco.MjModel.from_xml_path(xml_path)

    print("Total joints:", model.njnt)
    print("=" * 40)

    # Loop through joints
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        print(f"Index {j + 7:2d} : {name}")