import mujoco as mj
import json
import argparse
import re
from sensor_array import *
from sensor_array import write_xml_model as write_xml_model_cpu

black_list = json.load(open("gpu_collision_black_list.json"))["black_list"]

finger_tip_types = ["box","decomposd","low_poly","high_poly"]

def remove_collision_geom_from_model(spec=None):
    if spec == None:
        spec = mj.MjSpec.from_file("robot_touch_sensor_array_generated.xml")

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
    return spec
    




def add_options_to_model(spec):
    """
    <option timestep="0.001" integrator="implicitfast" >
      <flag eulerdamp="disable"/>
    </option>
    """
  
    spec.option.timestep = 0.001
    spec.option.iterations = 100
    spec.option.ls_iterations = 50
    spec.option.integrator = "implicitfast"
    return spec

def add_options_to_model_xml(spec,timestep=0.001,iterations=100,ls_iterations=50,integrator="implicitfast"):
    xml = spec.to_xml()
    # Remove any existing option tags
    xml = re.sub(r'<option[^>]*>.*?</option>', '', xml, flags=re.DOTALL)
    
    # Create the new option XML
    replace_option_xml = f"""  <option timestep="{timestep}" 
        integrator="{integrator}" 
        iterations="{iterations}" 
        ls_iterations="{ls_iterations}"> 
        <flag eulerdamp="disable"/>  
    </option>
    """
    
    # Insert after compiler tag (or after mujoco tag if no compiler)
    if re.search(r'<compiler[^>]*/>', xml):
        xml = re.sub(r'(<compiler[^>]*/>)', r'\1\n' + replace_option_xml, xml)
    else:
        xml = re.sub(r'(<mujoco[^>]*>)', r'\1\n' + replace_option_xml, xml)
  
    return xml

def add_custom_to_model_xml(xml,max_contact_points=30,max_geom_pairs=12):
    """
     <custom>
        <numeric data="30" name="max_contact_points"/>
        <numeric data="12" name="max_geom_pairs"/>
      </custom>
    """
    # Remove any existing custom tags
    xml = re.sub(r'<custom[^>]*>.*?</custom>', '', xml, flags=re.DOTALL)
    
    # Create the new custom XML
    replace_custom_xml = f'  <custom>\n    <numeric data="{max_contact_points}" name="max_contact_points"/>\n    <numeric data="{max_geom_pairs}" name="max_geom_pairs"/>\n  </custom>'
    
    # Insert after option tag (or after compiler if no option, or after mujoco as last resort)
    if re.search(r'</option>', xml):
        xml = re.sub(r'(</option>)', r'\1\n' + replace_custom_xml, xml, flags=re.DOTALL)
    elif re.search(r'<compiler[^>]*/>', xml):
        xml = re.sub(r'(<compiler[^>]*/>)', r'\1\n' + replace_custom_xml, xml)
    else:
        xml = re.sub(r'(<mujoco[^>]*>)', r'\1\n' + replace_custom_xml, xml)
    
    return xml
    
    

   
def write_xml(xml:str):
    with open("robot_touch_sensor_array_mjx_generated_model.xml", "w") as f:
        f.write(xml)

def write_xml_model(spec):
  xml = spec.to_xml()
  
  with open("robot_touch_sensor_array_mjx_generated_model.xml", "w") as f:
    f.write(xml)

if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description="Simplify MuJoCo model for MJX")
    # CPU
    # Gneerate sensor array model
    parser.add_argument("--sensor-array", default=True, help="Flat to indicate if sensor array model should be generated")
    args = parser.parse_args()
    if args.sensor_array:
        spec = place_grid_on_model(site="uspa46_1",grid=[3, 3, 3], spacing=[-0.02,0.0,-0.015])
        spec = place_grid_on_model(site="uspa46_2",grid=[3, 3, 3], spacing=[-0.02,0.0,-0.015],spec=spec)
        spec = place_grid_on_model(site="uspa46_3",grid=[3, 3, 3], spacing=[-0.02,0.0,-0.015],spec=spec)
    
        #figers
        for site in [f"{b}_bs_uspa44" for b in ["if", "mf", "rf"] ]:
          spec = place_grid_on_model(site=site,grid=[3, 3, 3], spacing=[-0.01,0.0,-0.01],spec=spec)

        for site in [f"{b}_px_uspa44" for b in ["if", "mf", "rf"] ]:
          spec = place_grid_on_model(site=site,grid=[3, 3, 3], spacing=[-0.01,-0.01,0],spec=spec)

        for site in [f"{b}_md_uspa44" for b in ["if", "mf", "rf"] ]:
          spec = place_grid_on_model(site=site,grid=[3, 3, 3], spacing=[-0.01,0.0,-0.01],spec=spec)

        #thumb
        spec = place_grid_on_model(site="th_bs_uspa44",grid=[3, 3, 3], spacing=[-0.01,0.0,-0.01],spec=spec)
        spec = place_grid_on_model(site="th_px_uspa44",grid=[3, 3, 3], spacing=[0,-0.01,0.01],spec=spec)

        # figertips
        for site in [f"{b}_uSCuALHA" for b in ["if", "mf", "rf","th"] ]:
          place_griod_on_figertip(site,spec)


        spec = add_touch_sensor_to_sites(spec)
  
        write_xml_model_cpu(spec)
 

    # GPU

    print(f"spec.option.timestep::{spec.option.timestep}")
    print(f"spec.option.integrator::{spec.option.integrator}")
    print(f"spec.option.iterations::{spec.option.iterations}")
    print(f"spec.option.ls_iterations::{spec.option.ls_iterations}")
    spec = remove_collision_geom_from_model()
    spec = add_grasp_site(spec)
    spec = add_finger_tips_collision_geom_to_model(spec)
    spec = add_marker_to_model(spec)
   
    xml = add_options_to_model_xml(spec)
    xml = add_custom_to_model_xml(xml)
    write_xml(xml)

    # write_xml_model(spec)