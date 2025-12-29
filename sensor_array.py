import mujoco as mj
import numpy as np

def place_grid_on_model(grid=[3, 5],spacing=[0.3,0.25]):
  spec = mj.MjSpec.from_file("Flat_array_base_model.xml")

  root = spec.worldbody
  box = spec.body("box")
  site_x0_y0 = spec.site("x0_y0")
  pos_x0_y0 = site_x0_y0.pos
  print(pos_x0_y0)

  for x in np.arange(0, grid[0],1):
    for y in np.arange(0, grid[1],1):  
      print(x,y)
      box.add_site(
        name=f"{x}_{y}",
        pos=[
        pos_x0_y0[0] + spacing[0] * x,
        pos_x0_y0[1] + spacing[1] * y,
        pos_x0_y0[2]
      ],
      size=site_x0_y0.size)

  return spec

def add_touch_sensor_to_sites(spec):
  for site in spec.sites:
    print(f"site::name::{site.name}")
    spec.add_sensor(
      name= site.name+"_touch",
      objname= site.name,
      type= mj.mjtSensor.mjSENS_TOUCH,
      objtype= mj.mjtObj.mjOBJ_SITE
    )

  return spec

def add_force_sensor_to_sites(spec):
  for site in spec.sites:
    print(f"site::name::{site.name}")
    spec.add_sensor(
      name= site.name+"_force",
      objname= site.name,
      type= mj.mjtSensor.mjSENS_FORCE,
      objtype= mj.mjtObj.mjOBJ_SITE
    )
  return spec

def write_xml_model(spec):
  xml = spec.to_xml()
  
  with open("Flat_array_model.xml", "w") as f:
    f.write(xml)
   
if __name__=="__main__":
  spec = place_grid_on_model()
  spec = add_touch_sensor_to_sites(spec)
  spec = add_force_sensor_to_sites(spec)
  write_xml_model(spec)
