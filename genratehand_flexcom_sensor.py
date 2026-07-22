

import mujoco as mj
from pathlib import Path


finger_tip_types = ["CoACD","Box"]


uspa_46 = {
  "width": 0.05,
  "height":0.03
}

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


def flex_mounting_points(spec,site_name):
  site = spec.site(site_name)
  body = site.parent
  print(body.name) 

  sites = {}
  for i in range(2):
    for j in range(2):
      if i == 0 and j == 0:
        continue
      x = site.pos[0] + i * uspa_46["width"]
      y = site.pos[1] + j * uspa_46["height"]
      z = site.pos[2]
      site = body.add_site(
        name=f"flex_mounting_point_{i}_{j}",
        pos=[x,y,z],
        quat=[1,0,0,0]
      )
      sites[site.name] = site

  return sites

def add_flex_sensor(spec, sites):
  b_palm = spec.body("palm")
  geom = spec.geom("uspa46_1")
  # Box half-sizes: [hx, hy, hz]. Sensor face is -Y in palm frame.
  hx, hy, hz = (float(s) for s in geom.size[:3])
  count = [3, 3, 1]
  # Centered on the outer face of uspa46_1.
  flex_pos = [
      float(geom.pos[0]),
      float(geom.pos[1]) - hy,
      float(geom.pos[2]),
  ]
  # +90° about X: local XY sheet -> palm XZ (flat on the sensor face).
  _s2 = 0.5 ** 0.5
  flex2 = b_palm.make_flex(
        name='shell_flex',
        type='grid',
        dim=2,
        count=count,
        spacing=[2 * hx / (count[0] - 1), 2 * hz / (count[1] - 1), 0.01],
        pos=flex_pos,
        quat=[_s2, _s2, 0, 0],
        mass=0.01,
        equality=1,
        elastic2d=2,  # bend
    )

  flex2.young = 1e3
  flex2.thickness = 0.01
  flex2.selfcollide = mj.mjtFlexSelf.mjFLEXSELF_NONE


def write_xml_given_spec_model(spec):
  xml = spec.to_xml()
  
  with open("leapXela_generated_flex_sensor.xml", "w") as f:
    f.write(xml)



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


def write_xml(xml, filename):
  model_dir = Path(__file__).resolve().parent
  path = model_dir / filename
  with open(path.as_posix(), "w") as f:
    f.write(xml)

def main():
    mode = "Box"
    spec = load_base_model(mode)
    sites = flex_mounting_points(spec,"uspa46_1")
    print(sites)
    add_flex_sensor(spec,sites)

    write_xml_given_spec_model(spec)
    xml = write_scene_xml("leapXela_generated_flex_sensor.xml")
    write_xml(xml, f"scene_flex_sensor_{mode}.xml")


if __name__ == "__main__":
    main()