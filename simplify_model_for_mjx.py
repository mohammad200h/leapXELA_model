import mujoco as mj

black_list = [
    "palm_collision_9",
    "palm_collision_12",
    "if_bs_collision_1",
    "if_bs_collision_2",
    "if_bs_collision_3",
    "if_bs_collision_4",
    "if_bs_uspa44",
    "if_ds_collision_1",
    "if_ds_collision_2",
    "if_ds_collision_3",
    "if_ds_collision_4",
    "if_ds_tip",
    "if_md_collision_1",
    "if_md_collision_2",
    "if_md_collision_3",
    "if_md_collision_4",
    "if_md_uspa44",
    "if_px_uspa44",
    "mf_px_collision_2",
    "mf_bs_collision_1",
    "mf_bs_collision_2",
    "mf_bs_collision_3",
    "mf_bs_collision_4",
    "mf_bs_uspa44",
    "mf_ds_collision_1",
    "mf_ds_collision_2",
    "mf_ds_collision_3",
    "mf_ds_collision_4",
    "mf_ds_tip",
    "mf_md_collision_1",
    "mf_md_collision_2",
    "mf_md_collision_3",
    "mf_md_collision_4",
    "mf_md_uspa44",
    "mf_px_uspa44",
    "palm_collision_11",
    "palm_collision_13",
    "palm_collision_17",
    "palm_collision_18",
    "palm_collision_19",
    "palm_collision_20",
    "palm_collision_21",
    "rf_bs_collision_1",
    "rf_bs_collision_2",
    "rf_bs_collision_3",
    "rf_bs_collision_4",
    "rf_bs_uspa44",
    "rf_ds_collision_1",
    "rf_ds_collision_2",
    "rf_ds_collision_3",
    "rf_ds_collision_4",
    "rf_ds_tip",
    "rf_md_collision_1",
    "rf_md_collision_2",
    "rf_md_collision_3",
    "rf_md_collision_4",
    "rf_md_uspa44",
    "rf_px_uspa44",
    "th_bs_collision_1",
    "th_bs_collision_2",
    "th_bs_collision_3",
    "th_bs_collision_4",
    "th_bs_uspa44",
    "th_ds_collision_3",
    "th_ds_tip",
    "th_px_collision_2",
    "th_px_collision_3",
    "th_px_collision_4",
    "th_px_collision_5",
    "th_px_collision_6",
    "th_ds_collision_2",
    "th_px_uspa44",
]

finger_tip_types = ["box","decomposd","low_poly","high_poly"]

def remove_collision_geom_from_model(spec=None):
    if spec == None:
        spec = mj.MjSpec.from_file("robot_touch_sensor_array_base_model.xml")

    geoms = spec.worldbody.find_all(mj.mjtObj.mjOBJ_GEOM)
    for geom in geoms:
        if geom.name in black_list:
            spec.delete(geom)
    return spec

def add_grasp_site(spec):
    """
    <site name="grasp_site" pos="0.11 0 0.03" group="5"/>

    """
    spec.worldbody.add_site(
        name="grasp_site",
        pos=[0.11, 0, 0.03],
        group=5
    )
    return spec

def add_finger_tips_collision_geom_to_model(spec):
    # https://github.com/google-deepmind/mujoco/blob/a26f09accfdb52c8967474d7c35350fc9651f1be/python/mujoco/specs_test.py#L573
    """
    <material name="col" rgba="0.6 1 0.6 0.2"/>
    <default class="collision">
      <geom type="box" group="3" material="col"/>
    </default>
    <geom name="rf_ds_collision_1" class="collision" pos="-0.004 -0.04 0.0145" size="0.019 0.02 0.016"/>
    """
    # get collison default
    geom_col_palm = spec.geom("palm_collision_1")

    finger =  ["if","mf","rf","th"]
    for name in finger:
        body_name = f"{name}_ds"
        body = spec.body(body_name)
        geom_ref=body.add_geom(
            name= f"{name}_tip",
            type=mj.mjtGeom.mjGEOM_BOX,
         size=[0.019, 0.02, 0.016],
         pos=[-0.004, -0.04, 0.0145], 
        default=geom_col_palm.classname)

    return spec
       
def add_marker_to_model(spec):
     th_site = spec.site("th_tip")
     marker_default = th_site.classname

     fingers = ["if","mf","rf"]
     for name in fingers:
        body_name = f"{name}_ds"
        body = spec.body(body_name)
        site = body.add_site(
            name= f"{name}_tip",
            default=marker_default)
    
def write_xml_model(spec):
  xml = spec.to_xml()
  
  with open("robot_touch_sensor_array_mjx_generated_model.xml", "w") as f:
    f.write(xml)

if __name__ == "__main__":
    spec = remove_collision_geom_from_model()
    spec = add_grasp_site(spec)
    spec = add_finger_tips_collision_geom_to_model(spec)
    add_marker_to_model(spec)

    write_xml_model(spec)