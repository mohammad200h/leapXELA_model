import mujoco as mj


FINGER_TIP_SITES = ["rf_pointcloud_base_frame", "mf_pointcloud_base_frame",
 "if_pointcloud_base_frame", "th_pointcloud_base_frame"]

_4_6_SITES = ["4_6_1_base_frame", "4_6_2_base_frame", "4_6_3_base_frame"]

def _delete_free_joints_in_subtree(spec: mj.MjSpec, root_body) -> None:
    stack = [root_body]
    while stack:
        body = stack.pop()
        for j in list(body.joints):
            if j.type == mj.mjtJoint.mjJNT_FREE:
                spec.delete(j)
        stack.extend(list(body.bodies))

def get_fingertip_spec():
    spec = mj.MjSpec.from_file("fingertip.xml")
    return spec

def get_4_6_spec():
    spec = mj.MjSpec.from_file("4_6.xml")
    return spec

def get_robot_spec():
    spec = mj.MjSpec.from_file("robot_with_sites.xml")
    return spec

def attach_fingertip_to_sites(fingertip_spec, robot_spec):
    for site_name in FINGER_TIP_SITES:
        site = robot_spec.site(site_name)
        prefix = f"{site_name}_"
        # Attach requires the attached elements to come from the child spec.
        robot_spec.attach(fingertip_spec.copy(), prefix=prefix, suffix="", site=site)
        attached_root = robot_spec.body(f"{prefix}root")
        _delete_free_joints_in_subtree(robot_spec, attached_root)

    return robot_spec


def attach_4_6_sensors_to_sites(_4_6_spec,robot_spec):
    for site_name in _4_6_SITES:
        site = robot_spec.site(site_name)
        prefix = f"{site_name}_"
        robot_spec.attach(_4_6_spec.copy(), prefix=prefix, suffix="", site=site)
        # If the attached model has a free joint at its root, remove it.
        # Some sensor models may not have an explicit "root" body; handle both cases.
        try:
            attached_root = robot_spec.body(f"{prefix}root")
        except KeyError:
            attached_root = None
        if attached_root is not None:
            _delete_free_joints_in_subtree(robot_spec, attached_root)

    return robot_spec

if __name__ == "__main__":
    fingertip_spec = get_fingertip_spec()
    _4_6_spec = get_4_6_spec()
    robot_spec = get_robot_spec()
    attach_fingertip_to_sites(fingertip_spec, robot_spec)
    attach_4_6_sensors_to_sites(_4_6_spec,robot_spec)
    # Compile once before exporting XML.
    robot_spec.compile()
    with open("robot_with_sensors.xml", "w", encoding="utf-8") as f:
        f.write(robot_spec.to_xml())