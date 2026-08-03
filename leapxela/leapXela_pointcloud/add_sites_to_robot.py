import mujoco as mj
import json

map = {
    "finger":"rf",
    "finger_2":"mf",
    "finger_3":"if",
    "thumb":"th",

}

reverse_map = {

    "palm": "leap_hand_xela_back_cover",

    "rf_bs": "p4_unified",
    "mf_bs": "p4_unified_2",
    "if_bs": "p4_unified_3",

    "rf_px": "p3_unified",
    "mf_px": "p3_unified_2",
    "if_px": "p3_unified_3",

    "rf_md": "p2_unified",
    "mf_md": "p2_unified_2",
    "if_md": "p2_unified_3",

    "if_ds": "finger",
    "mf_ds": "finger_2",
    "rf_ds": "finger_3",

    "th_mp": "thp2_unified",
    "th_bs": "clipper",
    "th_px": "thp1_unified",
    "th_ds": "thfingertip_unified",

}

def add_figer_tip_sites(spec,map):
    delete_list = []
    for b_name,finger_name in map.items():
        body = spec.body(b_name)
        for geom in body.geoms:
            # add site
            if geom.material == "pointcloud_material" and geom.classname.name == "collision":
                body.add_site(
                    name=f"{finger_name}_pointcloud_base_frame",
                    pos=geom.pos,
                    quat=geom.quat
                )
            # add collision and visual geom to delete list
            if geom.material == "pointcloud_material":
                delete_list.append(geom)

    for geom in delete_list:
        spec.delete(geom)

def add_4_6_sites(spec):
    idx = 1
    delete_list = []
    body = spec.body("leap_hand_xela_back_cover")
    for geom in body.geoms:
        if geom.material == "4_6_origin_material" and geom.classname.name == "collision":
            body.add_site(
                name=f"4_6_{idx}_base_frame",
                pos=geom.pos,
                quat=geom.quat
            )
            idx +=1
        if geom.material == "4_6_origin_material":
            delete_list.append(geom)

    for geom in delete_list:
        spec.delete(geom)


# def add_4_4_sites(spec):
#     with open("4_4_sites.json", "r") as f:
#         sites = json.load(f)
#     for site_name, site_data in sites.items():
#         parent = reverse_map[site_data["parent"]]
#         print(f"add_4_4_sites::parent: {parent}")

#         spec.body(parent).add_site(
#             name=site_name,
#             pos=site_data["pos"],
#             quat=site_data["quat"]
#         )
    

def write_xml(spec):
    
    with open("robot_with_sites.xml", "w") as f:
        f.write(spec.to_xml())

if __name__ == "__main__":
    spec = mj.MjSpec.from_file("robot.xml")
    add_figer_tip_sites(spec,map)
    add_4_6_sites(spec)
    # add_4_4_sites(spec)
    write_xml(spec)