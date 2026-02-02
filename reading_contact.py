import mujoco as mj
import numpy as np

import mujoco.viewer



if __name__ == "__main__":
  spec = mj.MjSpec.from_file("scene_mjx_cube.xml")

  model = spec.compile()
  data = mj.MjData(model)

  # visualization
  with mj.viewer.launch_passive(
          model=model, data=data, show_left_ui=False, show_right_ui=False
      ) as viewer:
    mj.mjv_defaultFreeCamera(model, viewer.cam)
    mj.mj_forward(model, data)

    while viewer.is_running():
      mj.mj_step(model, data)
          
      for i in range(data.ncon):
        con = data.contact[i]
        g1 = con.geom1
        g2 = con.geom2

        name1 = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, g1)
        name2 = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, g2)
        print(f"Contact {i}: {name1} and {name2}")
      
      viewer.sync()