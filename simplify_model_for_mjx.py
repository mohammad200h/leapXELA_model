from email.policy import default
from pickletools import read_unicodestring1
import argparse
from pathlib import Path
import mujoco as mj
import re
import numpy as np

finger_tips_black_list = ["if_ds_tip", "mf_ds_tip", "rf_ds_tip", "th_ds_tip"]
touch_grid_black_list = ["if_bs_uspa44", "mf_bs_uspa44", "rf_bs_uspa44",
 "if_md_uspa44", "mf_md_uspa44", "rf_md_uspa44",
 "if_px_uspa44", "mf_px_uspa44", "rf_px_uspa44",
 "th_bs_uspa44", "th_px_uspa44"]

black_list = [
    "palm_collision_9",
    "palm_collision_12",
    "if_bs_collision_1",
    "if_bs_collision_2",
    "if_bs_collision_3",
    "if_bs_collision_4",
    "if_ds_collision_1",
    "if_ds_collision_2",
    "if_ds_collision_3",
    "if_ds_collision_4",
    "if_md_collision_1",
    "if_md_collision_2",
    "if_md_collision_3",
    "if_md_collision_4",
    "mf_px_collision_2",
    "mf_bs_collision_1",
    "mf_bs_collision_2",
    "mf_bs_collision_3",
    "mf_bs_collision_4",
    "mf_ds_collision_1",
    "mf_ds_collision_2",
    "mf_ds_collision_3",
    "mf_ds_collision_4",
    "mf_md_collision_1",
    "mf_md_collision_2",
    "mf_md_collision_3",
    "mf_md_collision_4",
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
    "rf_ds_collision_1",
    "rf_ds_collision_2",
    "rf_ds_collision_3",
    "rf_ds_collision_4",
    "rf_md_collision_1",
    "rf_md_collision_2",
    "rf_md_collision_3",
    "rf_md_collision_4",
    "th_bs_collision_1",
    "th_bs_collision_2",
    "th_bs_collision_3",
    "th_bs_collision_4",
    "th_ds_collision_3",
    "th_px_collision_2",
    "th_px_collision_3",
    "th_px_collision_4",
    "th_px_collision_5",
    "th_px_collision_6",
    "th_ds_collision_2",
]

finger_tip_types = ["CoACD","Box"]


LEAPXELA_PALM_ELUR = [0, 1.57 + 0.31, -1.57]
CUBE_SCALE_FACTOR = 1.1
CUBE_POS = [0.11, 0.0, 0.1]
CUBE_EULER = [0, 0, 0]




LEAPXELA_ENV_CUBE_FRICTION = 0.3

def rename_finger_tips(spec):
    fingers = ["if","mf","rf","th"]
    for idx, name in enumerate(finger_tips_black_list):
        finger = fingers[idx]
        spec.geom(name).name = f"{finger}_tip"
    return spec

def overwrite_pose_of_the_hand(spec , euler):
    """
    pos="0 0.011 -0.01" quat="0.411476 -0.574943 0.575401 -0.411148"
    """

    # Compute quaternion from Euler angles using MuJoCo helper
    quat = np.zeros((4, 1), dtype=np.float64)
    euler = np.array(euler, dtype=np.float64).reshape(3, 1)
    mj.mju_euler2Quat(quat, euler, "xyz")

    spec.body("palm").pos = [0, 0.011, -0.01]
    # `mju_euler2Quat` writes into `quat` in-place; flatten for MuJoCo spec assignment
    spec.body("palm").quat = quat.flatten().tolist()
    
    return spec

def overwrite_cube(friction, scale_factor, pos, euler):
    quat = np.zeros((4, 1), dtype=np.float64)
    euler = np.array(euler, dtype=np.float64).reshape(3, 1)
    mj.mju_euler2Quat(quat, euler, "xyz")
    quat = quat.flatten().tolist()
    size = scale = [0.035 * scale_factor] * 3
    xml = f"""
        <mujoco>
          <default>
            <default class="cube">
              <geom friction="{friction} 0.05" conaffinity="2" condim="3"/>
            </default>
          </default>

          <asset>
            <texture name="cube" type="cube" fileup="reorientation_cube_textures/fileup.png"
              fileback="reorientation_cube_textures/fileback.png" filedown="reorientation_cube_textures/filedown.png"
              filefront="reorientation_cube_textures/filefront.png" fileleft="reorientation_cube_textures/fileleft.png"
              fileright="reorientation_cube_textures/fileright.png"/>
            <material name="cube" texture="cube"/>
            <texture name="graycube" type="cube" fileup="reorientation_cube_textures/grayup.png"
              fileback="reorientation_cube_textures/grayback.png" filedown="reorientation_cube_textures/graydown.png"
              filefront="reorientation_cube_textures/grayfront.png" fileleft="reorientation_cube_textures/grayleft.png"
              fileright="reorientation_cube_textures/grayright.png"/>
            <material name="graycube" texture="graycube"/>
            <texture name="dexcube" type="2d" file="reorientation_cube_textures/dex_cube.png"/>
            <material name="dexcube" texture="dexcube"/>
            <mesh name="cube_mesh" file="./meshes/dex_cube.obj" scale="{scale[0]} {scale[1]} {scale[2]}"/>
          </asset>

          <worldbody>
            <body name="cube" pos="{pos[0]} {pos[1]} {pos[2]}" 
            quat="{quat[0]} {quat[1]} {quat[2]} {quat[3]}" childclass="cube">
              <freejoint name="cube_freejoint"/>
              <geom type="mesh" mesh="cube_mesh" material="dexcube" contype="0" conaffinity="0" density="0" group="2"/>
              <geom name="cube" type="box" size="{size[0]} {size[1]} {size[2]}" mass=".108" group="3"/>
              <site name="cube_center" pos="0 0 0" group="4"/>
            </body>
          </worldbody>
        </mujoco>
    """

    return xml


def remove_collision_geom_from_model(spec=None):

    geoms = spec.worldbody.find_all(mj.mjtObj.mjOBJ_GEOM)
    for geom in geoms:
        if geom.name in black_list:
            spec.delete(geom)
    return spec

def add_grasp_site(spec):
    """
    <site name="grasp_site" pos="0.11 0.0 0.03" group="4"/>
    """
    spec.worldbody.add_site(
        name="grasp_site",
        pos=[0.11, 0, 0.03],
        group=4
    )
    return spec

def add_simplified_finger_tips_collision_geom_to_model(spec):
    # https://github.com/google-deepmind/mujoco/blob/a26f09accfdb52c8967474d7c35350fc9651f1be/python/mujoco/specs_test.py#L573
    """
    <material name="col" rgba="0.6 1 0.6 0.2"/>
    <default class="collision">
      <geom type="box" group="3" material="col"/>
    </default>
    <geom name="rf_ds_collision_1" class="collision" pos="-0.004 -0.04 0.0145" size="0.019 0.02 0.016"/>
    """

    # remove fingertip meshes
    tips = []
    for finger in ["if","mf","rf","th"]:
        tips += [f"{finger}_ds_tip"]
        tips += [f"{finger}_ds_tip_{idx}" for idx in range(2, 7)]
    for tip in tips:
        spec.delete(spec.geom(tip))

    # print(f"\n\ntips:: {tips}\n\n")
    # create default for fingertps 
    """
    <default class="uSCuALHA_simplified">
        <geom  group="3" type="box" size="0.015 0.02 0.014" pos="-0.0009 -0.04 0.0145"
                 euler="0 0 0.05" friction="0.5" material="orange" />
    </default>
    """
    # uSCuALHA =spec.find_default('uSCuALHA')
    # print(f"uSCuALHA::geom::friction:: {uSCuALHA.geom.friction}")
    
    main_def = spec.default
    uSCuALHA_simplified = spec.add_default("uSCuALHA_simplified",main_def)
    uSCuALHA_simplified.geom.type = mj.mjtGeom.mjGEOM_BOX
    uSCuALHA_simplified.geom.size = [0.015, 0.02, 0.014]
    uSCuALHA_simplified.geom.pos = [-0.0009, -0.04, 0.0145]
    uSCuALHA_simplified.geom.quat = [0.9996875, 0, 0, 0.024997]
    uSCuALHA_simplified.geom.friction[0] = 0.5
    uSCuALHA_simplified.geom.material = "orange"
    uSCuALHA_simplified.geom.group = 3

    for finger in ["if","mf","rf","th"]:
        body_name = f"{finger}_ds"
        body = spec.body(body_name)
        if finger == "th":
            body.add_geom(
                pos = [-0.001, -0.045, 0.0145],
                name= f"{finger}_ds_tip",
                default=uSCuALHA_simplified)
        else:
            body.add_geom(
                name= f"{finger}_ds_tip",
                default=uSCuALHA_simplified)
        

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
    
def write_xml_given_spec_model(spec):
  xml = spec.to_xml()
  
  with open("leapXela_generated_mjx.xml", "w") as f:
    f.write(xml)

def write_xml(xml, filename):
  model_dir = Path(__file__).resolve().parent
  path = model_dir / filename
  with open(path.as_posix(), "w") as f:
    f.write(xml)


def replace_solver_options(xml):
    # Remove existing option tags and replace with option_str
    option_str = """  <option timestep="0.01" integrator="Euler" iterations="5" ls_iterations="8">
    <flag eulerdamp="disable"/>
  </option>"""

    # Pattern to match <option> tag - handles both self-closing and tags with content
    # First pattern: self-closing tags like <option .../> (with optional whitespace)
    pattern1 = r'<option[^>]*\s*/>'
    # Second pattern: tags with content like <option ...>...</option>
    pattern2 = r'<option[^>]*>.*?</option>'
    
    # Replace option tags with content first (more specific)
    xml = re.sub(pattern2, option_str, xml, flags=re.DOTALL)
    # Then replace self-closing option tags
    xml = re.sub(pattern1, option_str, xml, flags=re.DOTALL)
    
    return xml

def replace_dynamics_options(xml):
    # Replace joint tag inside default tags only (not in body elements)
    # Only replace the joint tag that's a direct child of default, not those in nested default classes
    joint_str = """      <joint damping="0.2" armature="0.00149376" 
      actuatorfrcrange="-0.2196 0.2196" 
      frictionloss="0.02"/>"""

    # Strategy: Find joint tags within <default> sections but not within <body> sections
    # Only replace joints that are direct children of default tags (not in nested default classes)
    
    # Split XML into parts: before <worldbody> and after
    # Joints in default sections appear before <worldbody>
    if '<worldbody>' in xml:
        parts = xml.split('<worldbody>', 1)
        default_section = parts[0]
        body_section = '<worldbody>' + parts[1]
        
        # Replace joint tags that are direct children of default tags
        # Pattern: match joint tags that appear after <default> or </default> and before the next <default class=...>
        # We want to match joints like: <default class="leapXELA">\n      <joint .../>
        # But NOT joints in nested defaults like: <default class="mcp">\n      <joint .../>
        
        # Match joint tags that come right after a default opening (possibly nested)
        # and have the specific attributes we want to replace (damping="0.03")
        pattern = r'(<default[^>]*class="leapXELA"[^>]*>\s*\n\s*)(<joint[^>]*damping="0\.03"[^>]*/>)'
        
        def replace_joint(match):
            default_open = match.group(1)
            joint_tag = match.group(2)
            return default_open + joint_str
        
        default_section = re.sub(pattern, replace_joint, default_section)
        
        # Replace position and general tags inside <default class="leapXELA">
        actuator_str = """      <position kp="3.0" inheritrange="1"/>"""
        
        # Match position and general tags that are direct children of <default class="leapXELA">
        # Strategy: Find the leapXELA default block and replace tags within it, but stop at nested defaults
        # We'll match tags that appear after <default class="leapXELA"> and before the next <default or </default>
        
        # Pattern to match the leapXELA default block content (up to first nested default or closing)
        # Then replace position and general tags within that content
        leapxela_pattern = r'(<default[^>]*class="leapXELA"[^>]*>)(.*?)(?=<default class=|</default>)'
        
        def replace_tags_in_leapxela(match):
            default_open = match.group(1)
            content = match.group(2)
            
            # Replace position tags in this content
            content = re.sub(r'<position[^>]*/>', actuator_str, content)
            # Replace general tags in this content
            content = re.sub(r'<general[^>]*/>', actuator_str, content)
            
            return default_open + content
        
        default_section = re.sub(leapxela_pattern, replace_tags_in_leapxela, default_section, flags=re.DOTALL)
        
        xml = default_section + body_section
    else:
        # If no worldbody, replace joints in the entire XML
        pattern = r'(<default[^>]*class="leapXELA"[^>]*>\s*\n\s*)(<joint[^>]*damping="0\.03"[^>]*/>)'
        def replace_joint(match):
            default_open = match.group(1)
            return default_open + joint_str
        xml = re.sub(pattern, replace_joint, xml)
        
        # Replace position and general tags
        actuator_str = """      <position kp="3.0" inheritrange="1"/>"""
        pattern_position = r'(<default[^>]*class="leapXELA"[^>]*>.*?)(<position[^>]*/>)'
        pattern_general = r'(<default[^>]*class="leapXELA"[^>]*>.*?)(<general[^>]*/>)'
        xml = re.sub(pattern_position, lambda m: m.group(1) + actuator_str, xml, flags=re.DOTALL)
        xml = re.sub(pattern_general, lambda m: m.group(1) + actuator_str, xml, flags=re.DOTALL)
    
    return xml

def add_custome_settings(xml):
    # Add custom settings after the option tag
    custom_setting_str = """  <custom>
    <numeric data="30" name="max_contact_points"/>
    <numeric data="12" name="max_geom_pairs"/>
  </custom>"""

    # Pattern to match the closing </option> tag
    # Handle both self-closing and multi-line option tags
    pattern = r'(</option>)'
    
    # Check if custom settings already exist to avoid duplicates
    if '<custom>' in xml and 'max_contact_points' in xml:
        # Custom settings already exist, don't add again
        return xml
    
    # Replace the closing option tag with option tag + custom settings
    xml = re.sub(pattern, r'\1\n\n' + custom_setting_str, xml, count=1)
    
    return xml


def write_scene_xml(filename):
    xml = f"""
    <mujoco model="leap_scene">
      <include file="{filename}"/>
      <include file="reorientation_cube_generated_mjx.xml"/>

      <statistic center="0.15 0 0" extent="0.4" meansize="0.01"/>

      <visual>
        <headlight diffuse=".8 .8 .8" ambient=".2 .2 .2" specular="1 1 1"/>
        <rgba force="1 0 0 1"/>
        <global azimuth="120" elevation="-20"/>
        <map force="0.01" stiffness="500"/>
        <scale forcewidth="0.1" contactwidth="0.5" contactheight="0.2"/>
        <quality shadowsize="8192"/>
      </visual>

      <asset>
        <texture type="skybox" builtin="gradient" rgb1="1 1 1" rgb2="1 1 1" width="800" height="800"/>
        <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="1 1 1" rgb2="1 1 1" markrgb="0 0 0"
          width="300" height="300"/>
        <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0"/>
      </asset>

      <worldbody>
        <camera name="side" pos="-0.183 0.396 0.296" xyaxes="-0.783 -0.622 -0.000 0.332 -0.419 0.845"/>
        <geom name="floor" pos="0 0 -0.25" size="0 0 0.01" type="plane" material="groundplane" contype="2" conaffinity="2"/>
        <body name="goal" mocap="true" pos="0.325 0.17 0.0475">
          <!-- <geom type="mesh" mesh="cube_mesh" material="dexcube" contype="0" conaffinity="0" density="0" group="2"/> -->
          <geom type="mesh" mesh="cube_mesh" material="dexcube" contype="0" conaffinity="0" density="0" group="2"/>
          <geom type="box" size=".035 .035 .035" mass=".108" group="3"/>
        </body>
      </worldbody>

      <sensor>
        <!-- cube. -->
        <framepos name="cube_position" objtype="body" objname="cube"/>
        <framequat name="cube_orientation" objtype="body" objname="cube"/>
        <framelinvel name="cube_linvel" objtype="body" objname="cube"/>
        <frameangvel name="cube_angvel" objtype="body" objname="cube"/>
        <frameangacc name="cube_angacc" objtype="body" objname="cube"/>
        <framezaxis name="cube_upvector" objtype="body" objname="cube"/>

        <!-- hand. -->
        <framepos name="palm_position" objtype="site" objname="grasp_site"/>
        <framepos name="th_tip_position" objtype="site" objname="th_tip" reftype="site" refname="grasp_site"/>
        <framepos name="if_tip_position" objtype="site" objname="if_tip" reftype="site" refname="grasp_site"/>
        <framepos name="mf_tip_position" objtype="site" objname="mf_tip" reftype="site" refname="grasp_site"/>
        <framepos name="rf_tip_position" objtype="site" objname="rf_tip" reftype="site" refname="grasp_site"/>

        <!-- goal. -->
        <framequat name="cube_goal_orientation" objtype="body" objname="goal"/>
        <framezaxis name="cube_goal_upvector" objtype="body" objname="goal"/>
      </sensor>

      <keyframe>
        <key name="home"
          qpos="
          0.8 0 0.8 0.8
          0.8 0 0.8 0.8
          0.8 0 0.8 0.8
          0.8 0.8 0.8 0
          0.1 0.0 0.05 0.810967 -0.00262895 -0.585086 -0.000254303"
          ctrl="
          0.8 0 0.8 0.8
          0.8 0 0.8 0.8
          0.8 0 0.8 0.8
          0.8 0.8 0.8 0" mpos="0.25 0.16 0"
          mquat="1 0 0 0"/>
      </keyframe>
    </mujoco>
    """
    return xml


def load_base_model(mode):
  spec = None
  model_dir = Path(__file__).resolve().parent
  path = {
    "base_model": (model_dir / "leapXela_base_model.xml").as_posix(),
    "touchgrid": (model_dir / "robot_touch_sensor_array_binary_touchgrid_generated.xml").as_posix(),
  }
  if mode in finger_tip_types:
        spec = mj.MjSpec.from_file(path["base_model"])
        print(f"Loaded base model from {path['base_model']}")
  elif mode == "touchgrid":
      spec = mj.MjSpec.from_file(path["touchgrid"])
      print(f"Loaded base model from {path['touchgrid']}")

  else:
      raise ValueError(f"Invalid mode: {mode}")
  return spec

def generate_model_with_box_finger_tips(spec,mode, palm_euler, cube_friction, cube_scale_factor, cube_pos, cube_euler):
    print("Generating model with box finger tips")
    spec = remove_collision_geom_from_model(spec)
    spec = add_grasp_site(spec)
 
    spec = add_simplified_finger_tips_collision_geom_to_model(spec)
    
    spec = add_marker_to_model(spec)
    spec = overwrite_pose_of_the_hand(spec, euler=palm_euler)
    spec= rename_finger_tips(spec)
    xml = spec.to_xml()
    xml = replace_solver_options(xml)
    xml = replace_dynamics_options(xml)
    xml = add_custome_settings(xml)
    filename = f"leapXela_generated_mjx_{mode}.xml"
    write_xml(xml, filename)

    #########Cube#########
    xml = overwrite_cube(friction=cube_friction, 
                         scale_factor=cube_scale_factor,
                         pos=cube_pos,
                         euler=cube_euler)
    write_xml(xml, "reorientation_cube_generated_mjx.xml")
    ##### Write Scene XML #####
    xml = write_scene_xml(filename)
    write_xml(xml, f"scene_mjx_cube_{mode}_mjx.xml")

def generate_model_with_coacd_finger_tips(spec,mode, palm_euler, cube_friction, cube_scale_factor, cube_pos, cube_euler):
    print("Generating model with coacd finger tips")
    spec = remove_collision_geom_from_model(spec)
    spec = add_grasp_site(spec)
    spec = add_marker_to_model(spec)
    spec = overwrite_pose_of_the_hand(spec, euler=LEAPXELA_PALM_ELUR)
    spec= rename_finger_tips(spec)
    xml = spec.to_xml()
    xml = replace_solver_options(xml)
    xml = replace_dynamics_options(xml)
    xml = add_custome_settings(xml)
    filename = f"leapXela_generated_mjx_{mode}.xml"
    write_xml(xml, filename)

    #########Cube#########
    xml = overwrite_cube(friction=LEAPXELA_ENV_CUBE_FRICTION, 
                         scale_factor=CUBE_SCALE_FACTOR,
                         pos=CUBE_POS,
                         euler=CUBE_EULER)
    write_xml(xml, "reorientation_cube_generated_mjx.xml")
    ##### Write Scene XML #####
    xml = write_scene_xml(filename)
    write_xml(xml, f"scene_mjx_cube_{mode}_mjx.xml")

if __name__ == "__main__":
    spec= None

    parser = argparse.ArgumentParser(description="Simplify LeapXELA model for MuJoCo.")
    parser.add_argument(
        "--mode",
        choices=finger_tip_types,
        default="CoACD",
        help="Type of fingertip representation to use.",
    )
    args = parser.parse_args()
    mode = args.mode
    

    spec = load_base_model(mode)
    if mode == "Box":
        generate_model_with_box_finger_tips(spec, mode, LEAPXELA_PALM_ELUR, LEAPXELA_ENV_CUBE_FRICTION, CUBE_SCALE_FACTOR, CUBE_POS, CUBE_EULER)
    elif mode == "CoACD":
        generate_model_with_coacd_finger_tips(spec, mode, LEAPXELA_PALM_ELUR, LEAPXELA_ENV_CUBE_FRICTION, CUBE_SCALE_FACTOR, CUBE_POS, CUBE_EULER)
    else:
        raise ValueError(f"Invalid mode: {mode}")
    
    print("Model generation completed")