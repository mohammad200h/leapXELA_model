import mujoco as mj
import json


def produce_dict_of_pointcloud_pose(spec):
    pointcloud_dict = {}
    # Navigate to finger
   
    for i in range(1, 31):
        name = f"mf_pointcloud_base_frame_1_aftc_{i}_base2x"
        body = spec.body(name)
     
        # get joints limits first bosy is x then y then z
        x_body = body
        j_x = x_body.joints[0]
        y_body = x_body.bodies[0]
        j_y = y_body.joints[0]
        z_body = y_body.bodies[0]
        j_z = z_body.joints[0]
        pointcloud_dict[i] = {
            "pos": body.pos.tolist(),
            "quat": body.quat.tolist(),
            "j_x_range": j_x.range.tolist(),
            "j_y_range": j_y.range.tolist(),
            "j_z_range": j_z.range.tolist(),
        }

     

        
    print(f"length of pointcloud_dict: {len(pointcloud_dict.keys())}")
    return pointcloud_dict

def get_sites_info(spec):
    sites_dict = {}
    for finger in ["if", "mf", "rf", "th"]:
        
        name = f"{finger}_pointcloud_base_frame_root"
        site_name = f"{finger}_pointcloud_base_frame"
        body = spec.body(name)
        sites_dict[site_name] = {
            "pos": body.pos.tolist(),
            "quat": body.quat.tolist(),
        }
    return sites_dict

def combine_dicts(dict1, dict2):
    combined_dict = {}
    for key in dict1.keys():
        combined_dict[key] = dict1[key]
    for key in dict2.keys():
        combined_dict[key] = dict2[key]
    return combined_dict

def write_json(dict, filename):
    with open(filename, "w") as f:
        json.dump(dict, f)



if __name__ == "__main__":
    spec = mj.MjSpec.from_file("robot_with_sensors.xml")
    produce_dict_of_pointcloud_pose(spec)
    pointcloud_dict = produce_dict_of_pointcloud_pose(spec)
    sites_dict = get_sites_info(spec)
    combined_dict = combine_dicts(pointcloud_dict, sites_dict)
    write_json(combined_dict, "fingertip_magnet_pose.json")