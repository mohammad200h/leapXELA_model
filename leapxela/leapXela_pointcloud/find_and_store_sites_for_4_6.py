import mujoco as mj
import json

map = {
    "leap_hand_xela_back_cover":"palm"
}

def find_and_store_sites_for_4_6_on_if(spec):
    sites = {}
    idx = 1
    for geom in spec.geoms:
        if geom.material == "4_6_origin_material" and geom.classname.name == "collision":
         
            sites[f"4_6_{idx}_base_frame"] = {
                "pos": geom.pos.tolist(),
                "quat": geom.quat.tolist(),
                "parent": map[geom.parent.name],
                
            }
            idx += 1
    return sites



def write_sites(sites):
    with open("4_6_sites.json", "w") as f:
        json.dump(sites, f)
    



if __name__ == "__main__":
       spec = mj.MjSpec.from_file("robot.xml")
       sites = find_and_store_sites_for_4_6_on_if(spec)
       write_sites(sites)
    
    
