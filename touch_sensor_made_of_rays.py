from numpy.dtypes import BoolDType
import mujoco as mj
import numpy as np

import mujoco.viewer


def overwrite_pose_of_the_hand(spec):
  

    spec.body("palm").pos = [0.1, -0.1, -0.1]
    spec.body("palm").quat =  [0.707107, -0.707107, 0, 0]
    
    return spec

def place_cameras_on_palm(spec):
   
    plam_cameras = {}
    palm_body = spec.body("palm")
    geoms_names = ["uspa46_1", "uspa46_2", "uspa46_3"]

    for geom_name in geoms_names:
        geom = spec.geom(geom_name)
        geom_size = geom.size
        print(f"geom_size: {geom_size}")
        geom_pos = geom.pos
        camera_name = f"{geom_name}_cam"
        camera = palm_body.add_camera(name= camera_name, 
         pos=geom_pos, euler=[-1.57, 0, 0], 
         resolution=[4, 4], fovy=0.04,
         proj=mj.mjtProjection.mjPROJ_ORTHOGRAPHIC
        )
        plam_cameras[camera_name] = camera
    
    geoms_names = ["if_bs_uspa44", "mf_bs_uspa44", "rf_bs_uspa44"]
    for geom_name in geoms_names:
        geom = spec.geom(geom_name)
        geom_size = geom.size
        print(f"geom_size: {geom_size}")
        geom_pos = geom.pos
        geom_pos[1] -= geom_size[1]/2
        camera_name = f"{geom_name}_cam"
        camera = palm_body.add_camera(name= camera_name, 
         pos=geom_pos, euler=[-1.57, 0, 0], 
         resolution=[4, 4], fovy=0.02,
         proj=mj.mjtProjection.mjPROJ_ORTHOGRAPHIC
        )
        plam_cameras[camera_name] = camera

    return plam_cameras
    
def place_camera_on_fingers(spec):
    cameras = {}

    for finger in ["if", "mf", "rf"]:
        ##### px ######
        b_name = f"{finger}_px"
        g_name = f"{finger}_px_uspa44"
        body = spec.body(b_name)
        geom = spec.geom(g_name)
        geom_size = geom.size
        print(f"geom_size: {geom_size}")
        geom_pos = geom.pos
        geom_pos[2] += geom_size[2]/2
        camera_name = f"{g_name}_cam"
        camera = body.add_camera(name= camera_name, 
         pos=geom_pos, euler=[1.57*2, 0, 0], 
         resolution=[4, 4], fovy=0.02,
         proj=mj.mjtProjection.mjPROJ_ORTHOGRAPHIC
        )
        cameras[camera_name] = camera
    
        ##### md ######
        b_name = f"{finger}_md"
        g_name = f"{finger}_md_uspa44"
        body = spec.body(b_name)
        geom = spec.geom(g_name)
        geom_pos = geom.pos
        geom_size = geom.size
        geom_pos[1] += geom_size[1]/2
        camera_name = f"{g_name}_cam"
        camera = body.add_camera(name= camera_name, 
         pos=geom.pos, euler=[1.57, 0, 0], 
         resolution=[4, 4], fovy=0.02,
         proj=mj.mjtProjection.mjPROJ_ORTHOGRAPHIC
        )
        cameras[camera_name] = camera
    
    return cameras

def place_sites_on_fingertips_if_mf_rf(spec, finger="if"):
    """
     <default class="construction_line_2">
        <site pos="0.015 -0.035 0.0145" />
    </default>
     <default class="construction_line_4">
        <site pos="0.0142 -0.045 0.0145"/>
    </default>
     <default class="construction_line_7">
        <site pos="0.008 -0.060 0.0145"  />
    </default>
    
    <default class="h1_construction_line_3">
        <site pos="0.005 -0.035 0.029" />
    </default>

    <default class="h2_construction_line_3">
        <site pos="0.002 -0.045 0.029" />
    </default>

     <default class="h1_construction_line_6">
        <site pos="0.011 -0.035 0.004" />
    </default>
    <default class="h1_construction_line_7">
        <site pos="0.005 -0.035 0.000" />
        </default>
    <default class="h1_construction_line_8">
        <site pos="0.00 -0.035 -0.001" />
    </default>
     <default class="h2_construction_line_6">
          <site pos="0.002 -0.045 -0.0001" />
    </default>

    """
    body_name = f"{finger}_ds"
    body = spec.body(body_name)
    sites = {}
    sites_data = {
        "ray_1":{
            "pos": [0.015, -0.035, 0.0145],
            "euler": [0, 1.57, 0]
        },
        "ray_2":{
            "pos": [0.0142, -0.045, 0.0145],
            "euler": [0, 1.57, 0]
        },
        "ray_3":{
            "pos": [0.008, -0.060, 0.0145],
            "euler": [1.57, 0, 1.57]
        },
        "ray_4":{
            "pos": [0.005, -0.035, 0.029],
            "euler": [0, 0, 0]
        },
        "ray_5":{
            "pos": [0.002, -0.045, 0.029],
            "euler": [0, 0, 0]
        },
   
        "ray_7":{
            "pos": [0.005, -0.035, 0.000],
            "euler": [1.57*2, 0, 0]
        },
        "ray_9":{
            "pos": [0.002, -0.045, -0.0001],
            "euler": [1.57*2, 0, 0]
        },
    }

    for site_name, site_data in sites_data.items():
        finger_site_name = site_name+f"_{finger}"
        site = body.add_site(name=finger_site_name, pos=site_data["pos"], euler=site_data["euler"],
        group=2)
        sites[finger_site_name] = site

    return sites
    
 

def place_cameras_on_where_touch_sensor_is_placed(spec):
    palm_cameras = place_cameras_on_palm(spec)
    finger_cameras = place_camera_on_fingers(spec)
    return {**palm_cameras, **finger_cameras}
   
def attach_range_finder_to_cameras(spec,cameras):
    """
    This is intersesting, can a contact snesor be made of camera?
    https://github.com/google-deepmind/mujoco/blob/6ec808e2ce3af289ab3ddea6f6628eb11243e245/python/mujoco/specs_test.py#L1567
    """
    #https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html#mjtsensor
    #https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html#mjtobj
    # https://github.com/google-deepmind/mujoco/blob/6ec808e2ce3af289ab3ddea6f6628eb11243e245/python/mujoco/specs_test.py#L1767
    
    # Raydata field enum values for dataspec bitfield
    rd = mj.mjtRayDataField
    dist_val = int(rd.mjRAYDATA_DIST)
    dir_val = int(rd.mjRAYDATA_DIR)
    origin_val = int(rd.mjRAYDATA_ORIGIN)
    point_val = int(rd.mjRAYDATA_POINT)
    normal_val = int(rd.mjRAYDATA_NORMAL)
    depth_val = int(rd.mjRAYDATA_DEPTH)

    all_fields = (
        (1 << dist_val) | (1 << dir_val) | (1 << origin_val) |
        (1 << point_val) | (1 << normal_val) | (1 << depth_val)
    )
    
    for camera_name, _ in cameras.items():
        rf_sensor = spec.add_sensor(name=camera_name+"_rf",
            type=mj.mjtSensor.mjSENS_RANGEFINDER,
            objtype=mj.mjtObj.mjOBJ_CAMERA,
            objname=camera_name,
        )
        rf_sensor.intprm[0] = all_fields

def attach_range_finder_to_sites(spec,sites):
  # Raydata field enum values for dataspec bitfield
  rd = mj.mjtRayDataField
  dist_val = int(rd.mjRAYDATA_DIST)
  dir_val = int(rd.mjRAYDATA_DIR)
  origin_val = int(rd.mjRAYDATA_ORIGIN)
  point_val = int(rd.mjRAYDATA_POINT)
  normal_val = int(rd.mjRAYDATA_NORMAL)
  depth_val = int(rd.mjRAYDATA_DEPTH)
  all_fields = (
      (1 << dist_val) | (1 << dir_val) | (1 << origin_val) |
      (1 << point_val) | (1 << normal_val) | (1 << depth_val)
  )
  for site_name, site in sites.items():
    rf_sensor = spec.add_sensor(
      name=site_name+"_rf",
      type=mj.mjtSensor.mjSENS_RANGEFINDER,
      objtype=mj.mjtObj.mjOBJ_SITE,
      objname=site_name,
    )

    rf_sensor.intprm[0] = all_fields
  return spec

def write_xml(xml):
    with open("leapXela_touch_sensor_made_of_rays.xml", "w") as f:
        f.write(xml)

def main():
    spec = mj.MjSpec.from_file("leapXela_base_model.xml")
    ##### Camera Based Sensor #####
    overwrite_pose_of_the_hand(spec)
    cameras = place_cameras_on_where_touch_sensor_is_placed(spec)
    attach_range_finder_to_cameras  (spec,cameras)
    ##### Site Based Sensor #####
    sites_if = place_sites_on_fingertips_if_mf_rf(spec, finger="if")
    sites_mf = place_sites_on_fingertips_if_mf_rf(spec, finger="mf")
    sites_rf = place_sites_on_fingertips_if_mf_rf(spec, finger="rf")
    sites = {**sites_if, **sites_mf, **sites_rf}
    spec = attach_range_finder_to_sites(spec, sites)
    xml = spec.to_xml()
    write_xml(xml)


if __name__ == "__main__":
    main()