import mujoco as mj
import numpy as np

def place_grid_on_model( site:str, grid=[3, 3, 3], spacing=[-0.02,0.0,-0.015],spec=None):
  
  if spec == None:
    spec = mj.MjSpec.from_file("robot_touch_sensor_array_base_model.xml")

 
  site_x0_y0 = spec.site(site)
  pos_x0_y0 = site_x0_y0.pos
  body = site_x0_y0.parent
  print(pos_x0_y0)

  for x in np.arange(0, grid[0],1):
    for y in np.arange(0, grid[1],1):
      for z in np.arange(0, grid[1],1): 
        print(x,y,z)
        body.add_site(
          name=f"{site}_{x}_{y}_{z}",
          pos=[
          pos_x0_y0[0] + spacing[0] * x,
          pos_x0_y0[1] + spacing[1] * y,
          pos_x0_y0[2] + spacing[2] * z
        ],
        group=3 ,
        size=site_x0_y0.size)

  return spec

def place_griod_on_figertip(site:str,spec=None):
  if spec == None:
    spec = mj.MjSpec.from_file("robot_touch_sensor_array_base_model.xml")
  
  site_x0_y0 = spec.site(site)
  pos_x0_y0 = site_x0_y0.pos
  body = site_x0_y0.parent

  # top
  body.add_site(
        name=f"{site}_top_ass",
        pos=[
          pos_x0_y0[0] -0.005,
          pos_x0_y0[1] - 0.014,
          pos_x0_y0[2] 
        ],
        group=3 ,
        size=site_x0_y0.size
      )

  # sides
  for spacig in [0,0.01]:
    for idx,sign in enumerate([-1,1]):
      body.add_site(
        name=f"{site}_{spacig}_{idx}_ass",
        pos=[
          pos_x0_y0[0] - 0.005 ,
          pos_x0_y0[1] + spacig,
          pos_x0_y0[2] + sign * 0.011
        ],
        group=3 ,
        size=site_x0_y0.size
      )
  # buttom
  body.add_site(
        name=f"{site}_buttom_ass",
        pos=[
          pos_x0_y0[0] ,
          pos_x0_y0[1] + 0.01,
          pos_x0_y0[2] 
        ],
        group=3 ,
        size=site_x0_y0.size
      )
  
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
  
  with open("robot_touch_sensor_array_gem.xml", "w") as f:
    f.write(xml)
   
if __name__=="__main__":
  ######################### addig sites #################
  #palm
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
  spec = add_force_sensor_to_sites(spec)
  write_xml_model(spec)
 