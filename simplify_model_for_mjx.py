from pickletools import read_unicodestring1
import mujoco as mj
import re

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

finger_tip_types = ["decomposd","touchgrid"]

def rename_finger_tips(spec):
    fingers = ["if","mf","rf","th"]
    for idx, name in enumerate(finger_tips_black_list):
        finger = fingers[idx]
        spec.geom(name).name = f"{finger}_tip"
    return spec

def overwrite_pose_of_the_hand(spec):
    """
    pos="0 0.011 -0.01" quat="0.411476 -0.574943 0.575401 -0.411148"
    """


    spec.body("palm").pos = [0, 0.011, -0.01]
    spec.body("palm").quat = [0.411476, -0.574943, 0.575401, -0.411148]
    
    return spec

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
    
def write_xml_given_spec_model(spec):
  xml = spec.to_xml()
  
  with open("leapXela_generated_mjx.xml", "w") as f:
    f.write(xml)

def write_xml(xml):
    with open("leapXela_generated_mjx.xml", "w") as f:
        f.write(xml)


def replace_solver_options(xml):
    # Remove existing option tags and replace with option_str
    option_str = """  <option timestep="0.001" integrator="Euler" iterations="5" ls_iterations="8">
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

if __name__ == "__main__":
    spec= None
    mode = "decomposd" # Argparser
    if mode == "decomposd":
        spec = mj.MjSpec.from_file("leapXela_base_model.xml")
    elif mode == "touchgrid":
        spec = mj.MjSpec.from_file("robot_touch_sensor_array_binary_touchgrid_generated.xml")
    else:
        raise ValueError(f"Invalid mode: {mode}")

    spec = remove_collision_geom_from_model(spec)
    spec = add_grasp_site(spec)
    # spec = add_finger_tips_collision_geom_to_model(spec)
    spec = add_marker_to_model(spec)
    spec = overwrite_pose_of_the_hand(spec)
    spec= rename_finger_tips(spec)
    xml = spec.to_xml()
    xml = replace_solver_options(xml)
    xml = replace_dynamics_options(xml)
    xml = add_custome_settings(xml)
    write_xml(xml)