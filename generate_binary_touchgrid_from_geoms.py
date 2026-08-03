#### starting positions of sensors on plam represented by sites####
# name="uspa46_1" 
# name="uspa46_2" 
# name="uspa46_3" 

#### starting positions of sensors on finger represented by sites####
# <finger> = "if", "mf", "rf"
# name="<finger>_bs_uspa44"
# name="<finger>_px_uspa44"
# name="<finger>_md_uspa44"
# name="<finger>_uSCuALHA"


import mujoco as mj


def remove_collsions_geom_convering_sensors_except_for_finger(spec):
    for idx in range(3):
        name = f"uspa46_{idx+1}"
        geom = spec.geom(name)
        spec.delete(geom)
    
    for finger in ["if", "mf", "rf"]:
        name = f"{finger}_bs_uspa44"
        geom = spec.geom(name)
        spec.delete(geom)
        name = f"{finger}_px_uspa44"
        geom = spec.geom(name)
        spec.delete(geom)
        name = f"{finger}_md_uspa44"
        geom = spec.geom(name)
        spec.delete(geom)
    


def add_sensor_patch_defaults_for_if_mf_rf(spec,sensor_patch_default):

    sensor_patch_default_dict = {}
    sensor_patch_params = {}  # Store pos and euler for each default
    
    # top surface
    locations = [([0.01, -0.035, 0.0145],[0,0,0]),
                ([0.0100, -0.045, 0.0145],[0,0,0]),
                ([0.004, -0.057, 0.0145],[0,0,0.7])]
    for idx,loc in enumerate(locations):
        pos,euler = loc
        ref = spec.add_default(f"fingertip_sensor_top_surface_{idx+1}",sensor_patch_default)
        ref.geom.pos = pos
        # Note: euler cannot be set on geom defaults, will be set when adding geom

        sensor_patch_default_dict[f"fingertip_sensor_top_surface_{idx+1}"] = ref
        sensor_patch_params[f"fingertip_sensor_top_surface_{idx+1}"] = {"pos": pos, "euler": euler}
    
    # left surface
    locations = [([0.007, -0.035, 0.023],[0,-0.9,0]),
                ([0.007, -0.045, 0.023],[0,-0.9,0])]
    for idx,loc in enumerate(locations):
        pos,euler = loc
        ref = spec.add_default(f"fingertip_sensor_left_surface_{idx+1}",sensor_patch_default)
        ref.geom.pos = pos
        # Note: euler cannot be set on geom defaults, will be set when adding geom

        sensor_patch_default_dict[f"fingertip_sensor_left_surface_{idx+1}"] = ref
        sensor_patch_params[f"fingertip_sensor_left_surface_{idx+1}"] = {"pos": pos, "euler": euler}
    
    # right surface
    locations = [([0.007, -0.035, 0.006],[0,0.9,0]),
                ([0.007, -0.045, 0.006],[0,0.9,0])]
    for idx,loc in enumerate(locations):
        pos,euler = loc
        ref = spec.add_default(f"fingertip_sensor_right_surface_{idx+1}",sensor_patch_default)
        ref.geom.pos = pos
        # Note: euler cannot be set on geom defaults, will be set when adding geom

        sensor_patch_default_dict[f"fingertip_sensor_right_surface_{idx+1}"] = ref
        sensor_patch_params[f"fingertip_sensor_right_surface_{idx+1}"] = {"pos": pos, "euler": euler}

    return sensor_patch_default_dict, sensor_patch_params

def add_sensor_path_to_if_mf_rf(spec, finger_name = "if", sensor_patch_default_dict = None, sensor_patch_params = None):
    body = spec.body(f"{finger_name}_ds")
    
    # top surface
    for idx in range(3):
        default_key = f"fingertip_sensor_top_surface_{idx+1}"
        params = sensor_patch_params[default_key]
        body.add_geom(
            name=f"{finger_name}_sensor_top_surface_{idx+1}",
            default=sensor_patch_default_dict[default_key],
            pos=params["pos"],
            euler=params["euler"],
        )

    # left surface
    for idx in range(2):
        default_key = f"fingertip_sensor_left_surface_{idx+1}"
        params = sensor_patch_params[default_key]
        body.add_geom(
            name=f"{finger_name}_sensor_left_surface_{idx+1}",
            default=sensor_patch_default_dict[default_key],
            pos=params["pos"],
            euler=params["euler"],
        )
    # right surface
    for idx in range(2):
        default_key = f"fingertip_sensor_right_surface_{idx+1}"
        params = sensor_patch_params[default_key]
        body.add_geom(
            name=f"{finger_name}_sensor_right_surface_{idx+1}",
            default=sensor_patch_default_dict[default_key],
            pos=params["pos"],
            euler=params["euler"],
        )

def add_sensor_path_to_th(spec, sensor_default = None):
    # (pos,euler)
    locations =[
        ([0.01, -0.040, 0.0145],[0,0,0]),
        ([0.0100, -0.050 ,0.0145],[0,0,0]),
        ([0.004 ,-0.062  ,0.0145],[0,0,0.7]),
        ([0.007 ,-0.040  ,0.023],[0,-0.9,0]),
        ([0.007 ,-0.050  ,0.023],[0,-0.9,0]),
        ([0.007 ,-0.040  ,0.006],[0,0.9,0]),
        ([0.007 ,-0.050  ,0.006],[0,0.9,0]),
    ]
    for idx,loc in enumerate(locations):
        pos,euler = loc
        spec.body("th_ds").add_geom(
            name=f"th_sensor_{idx+1}",
            pos=pos,
            euler=euler,
            default=sensor_default
        )


def add_uspa44(spec, site_name,link_name,sensor_default):

    site_x0_y0 = spec.site(site_name)
    sites_parent_body = site_x0_y0.parent
    x,y,z = site_x0_y0.pos
    if link_name == "bs":
        y += 0.0025
    elif link_name == "px":
        z -= 0.0025
    elif link_name == "md":
        y -= 0.0025
    elif link_name == "th_bs":
        y -= 0.0025
    elif link_name == "th_px":
        x += 0.0025

    
    offset = 0.0025*2
    for j in range(4):
        for i in range(4):
            pos = None
            if link_name in ["bs", "md","th_bs"]:
                pos = [x - offset * i, y, z - offset * j]
            elif link_name =="px":
                pos = [x - offset * j , y - offset * i, z ]
            elif link_name == "th_px":
                pos = [x  , y - offset * i, z + offset * j]

            sites_parent_body.add_geom(
                name=f"{site_name}_sensor_patch_{i}_{j}",
                type=mj.mjtGeom.mjGEOM_BOX,
                size=[0.0025, 0.0025, 0.0025],
                pos=pos,
                default=sensor_default

            )
def add_uspa46(spec, site_name,sensor_default):
    site_x0_y0 = spec.site(site_name)
    sites_parent_body = site_x0_y0.parent
    x,y,z = site_x0_y0.pos

    x_offset = 0.0025*2
    z_offset = 0.0035*2
    y += 0.0025

    size=[0.0025, 0.0025, 0.0035]
    for j in range(4):
        for i in range(8):
            sites_parent_body.add_geom(
                name=f"{site_name}_sensor_patch_{i}_{j}",
                type=mj.mjtGeom.mjGEOM_BOX,
                size = size,
                pos=[x - x_offset * i, y , z - z_offset * j],
                default=sensor_default
            )


def write_xml_model(spec):
  xml = spec.to_xml()
  
  with open("robot_touch_sensor_array_binary_touchgrid_generated.xml", "w") as f:
    f.write(xml)

if __name__ == "__main__":
    spec = mj.MjSpec.from_file("leapXela_base_model.xml")
    site_names = ["if_bs_uspa44", "mf_bs_uspa44" ,"rf_bs_uspa44",
                  "if_md_uspa44", "mf_md_uspa44" ,"rf_md_uspa44",
                  "if_px_uspa44", "mf_px_uspa44" ,"rf_px_uspa44",
                  
     ]
    th_site_names = ["th_bs_uspa44","th_px_uspa44"]
    palm_site_names = ["uspa46_1", "uspa46_2" ,"uspa46_3"]

    collision_default = spec.geom("if_ds_collision_1").classname

    # add sensor_patch to collision default
    sensor_patch_default = spec.add_default("sensor_patch",collision_default)
    sensor_patch_default.geom.size=[0.005, 0.005, 0.005]
    sensor_patch_default.geom.rgba=[1, 0, 0, 1]
    sensor_patch_default.geom.group=4

    remove_collsions_geom_convering_sensors_except_for_finger(spec)
 
    # add sensor_patch to finger links
    for site_name in site_names:
        if "bs" in site_name:
            add_uspa44(spec, site_name,"bs",sensor_patch_default)
        elif "md" in site_name:
            add_uspa44(spec, site_name,"md",sensor_patch_default)
        elif "px" in site_name:
            add_uspa44(spec, site_name,"px",sensor_patch_default)
    sensor_patch_default_dict, sensor_patch_params = add_sensor_patch_defaults_for_if_mf_rf(spec,sensor_patch_default)
    for finger in ["if", "mf", "rf"]:
        add_sensor_path_to_if_mf_rf(spec, finger, sensor_patch_default_dict, sensor_patch_params)
       
    # add sensor_patch to thumb links
    for site_name in th_site_names:
        if "th_bs" in site_name:
            add_uspa44(spec, site_name,"th_bs",sensor_patch_default)
        elif "th_px" in site_name:
            add_uspa44(spec, site_name,"th_px",sensor_patch_default)
    add_sensor_path_to_th(spec,sensor_patch_default)
    
    # add sensor_patch to palm
    for site_name in palm_site_names:
        add_uspa46(spec, site_name,sensor_patch_default)

    write_xml_model(spec)