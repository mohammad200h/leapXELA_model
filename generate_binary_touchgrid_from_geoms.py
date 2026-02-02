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
    spec = mj.MjSpec.from_file("robot_touch_sensor_array_base_model.xml")
    site_names = ["if_bs_uspa44", "mf_bs_uspa44" ,"rf_bs_uspa44",
                  "if_md_uspa44", "mf_md_uspa44" ,"rf_md_uspa44",
                  "if_px_uspa44", "mf_px_uspa44" ,"rf_px_uspa44",
                  
     ]
    th_site_names = ["th_bs_uspa44","th_px_uspa44"]
    palm_site_names = ["uspa46_1", "uspa46_2" ,"uspa46_3"]

    geom_default = spec.geom("th_sensor_1").classname
 
    for site_name in site_names:
        if "bs" in site_name:
            add_uspa44(spec, site_name,"bs",geom_default)
        elif "md" in site_name:
            add_uspa44(spec, site_name,"md",geom_default)
        elif "px" in site_name:
            add_uspa44(spec, site_name,"px",geom_default)
       

    for site_name in th_site_names:
        if "th_bs" in site_name:
            add_uspa44(spec, site_name,"th_bs",geom_default)
        elif "th_px" in site_name:
            add_uspa44(spec, site_name,"th_px",geom_default)
    
    for site_name in palm_site_names:
        add_uspa46(spec, site_name,geom_default)

    write_xml_model(spec)