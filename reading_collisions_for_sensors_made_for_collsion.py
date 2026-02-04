import mujoco as mj
import numpy as np
import sys
import mujoco.viewer


uspa46_1_grid_locations = {f"uspa46_1_sensor_patch_{i}_{j}":(i,j) for i in range(8) for j in range(4)}
uspa46_2_grid_locations = {f"uspa46_2_sensor_patch_{i}_{j}":(i,j) for i in range(8) for j in range(4)}
uspa46_3_grid_locations = {f"uspa46_3_sensor_patch_{i}_{j}":(i,j) for i in range(8) for j in range(4)}

uspa44_if_bs_grid_locations = {f"if_bs_uspa44_sensor_patch_{i}_{j}":(i,j) for i in range(4) for j in range(4)}
uspa44_if_px_grid_locations = {f"if_px_uspa44_sensor_patch_{i}_{j}":(i,j) for i in range(4) for j in range(4)}
uspa44_if_md_grid_locations = {f"if_md_uspa44_sensor_patch_{i}_{j}":(i,j) for i in range(4) for j in range(4)}
uSCuALHA_if = []
uSCuALHA_mf = []
uSCuALHA_rf = []
uSCuALHA_th = []
for i in range(3):
    for side in ["left", "right","top"]:
        if side != "top" and i == 2:
            continue
        uSCuALHA_if.append(f"if_fingertip_sensor_{side}_surface_{i+1}")

for i in range(3):
    for side in ["left", "right","top"]:
        if side != "top" and i == 2:
            continue
        uSCuALHA_mf.append(f"mf_fingertip_sensor_{side}_surface_{i+1}")

for i in range(3):
    for side in ["left", "right","top"]:
        if side != "top" and i == 2:
            continue
        uSCuALHA_rf.append(f"rf_fingertip_sensor_{side}_surface_{i+1}")

for i in range(7):
        uSCuALHA_th.append(f"th_sensor_{i+1}")

uspa44_mf_bs_grid_locations = {f"mf_bs_uspa44_sensor_patch_{i}_{j}":(i,j) for i in range(4) for j in range(4)}
uspa44_mf_px_grid_locations = {f"mf_px_uspa44_sensor_patch_{i}_{j}":(i,j) for i in range(4) for j in range(4)}
uspa44_mf_md_grid_locations = {f"mf_md_uspa44_sensor_patch_{i}_{j}":(i,j) for i in range(4) for j in range(4)}

uspa44_rf_bs_grid_locations = {f"rf_bs_uspa44_sensor_patch_{i}_{j}":(i,j) for i in range(4) for j in range(4)}
uspa44_rf_px_grid_locations = {f"rf_px_uspa44_sensor_patch_{i}_{j}":(i,j) for i in range(4) for j in range(4)}
uspa44_rf_md_grid_locations = {f"rf_md_uspa44_sensor_patch_{i}_{j}":(i,j) for i in range(4) for j in range(4)}

th_bs_grid_locations = {f"th_bs_uspa44_sensor_patch_{i}_{j}":(i,j) for i in range(4) for j in range(4)}
th_px_grid_locations = {f"th_px_uspa44_sensor_patch_{i}_{j}":(i,j) for i in range(4) for j in range(4)}

cube_collsion_name = "cube_collision"





if __name__ == "__main__":
  spec = mj.MjSpec.from_file("scene_collsion.xml")

  model = spec.compile()
  data = mj.MjData(model)

  cube_collision_id = data.geom("cube_collision").id
  

  uspa46_1_grid_geoms_id = {data.geom(key).id:key for key in uspa46_1_grid_locations.keys()} 
  uspa46_2_grid_geoms_id = {data.geom(key).id:key for key in uspa46_2_grid_locations.keys()}
  uspa46_3_grid_geoms_id = {data.geom(key).id:key for key in uspa46_3_grid_locations.keys()}
  
  uspa44_if_bs_grid_geoms_id = {data.geom(key).id:key for key in uspa44_if_bs_grid_locations.keys()}
  uspa44_if_px_grid_geoms_id = {data.geom(key).id:key for key in uspa44_if_px_grid_locations.keys()}
  uspa44_if_md_grid_geoms_id = {data.geom(key).id:key for key in uspa44_if_md_grid_locations.keys()}
  uSCuALHA_if_grid_geoms_id = {data.geom(key).id:key for key in uSCuALHA_if}

  uspa44_mf_bs_grid_geoms_id = {data.geom(key).id:key for key in uspa44_mf_bs_grid_locations.keys()}
  uspa44_mf_px_grid_geoms_id = {data.geom(key).id:key for key in uspa44_mf_px_grid_locations.keys()}
  uspa44_mf_md_grid_geoms_id = {data.geom(key).id:key for key in uspa44_mf_md_grid_locations.keys()}
  uSCuALHA_mf_grid_geoms_id = {data.geom(key).id:key for key in uSCuALHA_mf}

  uspa44_rf_bs_grid_geoms_id = {data.geom(key).id:key for key in uspa44_rf_bs_grid_locations.keys()}
  uspa44_rf_px_grid_geoms_id = {data.geom(key).id:key for key in uspa44_rf_px_grid_locations.keys()}
  uspa44_rf_md_grid_geoms_id = {data.geom(key).id:key for key in uspa44_rf_md_grid_locations.keys()}
  uSCuALHA_rf_grid_geoms_id = {data.geom(key).id:key for key in uSCuALHA_rf}

  th_bs_grid_geoms_id = {data.geom(key).id:key for key in th_bs_grid_locations.keys()}
  th_px_grid_geoms_id = {data.geom(key).id:key for key in th_px_grid_locations.keys()}
  uSCuALHA_th_grid_geoms_id = {data.geom(key).id:key for key in uSCuALHA_th}

  activated_sensors = []
  
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
        if g1 == cube_collision_id:
            if g2 in uspa46_1_grid_geoms_id.keys():
                activated_sensors.append(uspa46_1_grid_geoms_id[g2])
            elif g2 in uspa46_2_grid_geoms_id.keys():
                activated_sensors.append(uspa46_2_grid_geoms_id[g2])
            elif g2 in uspa46_3_grid_geoms_id.keys():
                activated_sensors.append(uspa46_3_grid_geoms_id[g2])
            elif g2 in uspa44_if_bs_grid_geoms_id.keys():
                activated_sensors.append(uspa44_if_bs_grid_geoms_id[g2])
            elif g2 in uspa44_if_px_grid_geoms_id.keys():
                activated_sensors.append(uspa44_if_px_grid_geoms_id[g2])
            elif g2 in uspa44_if_md_grid_geoms_id.keys():
                activated_sensors.append(uspa44_if_md_grid_geoms_id[g2])
            elif g2 in uspa44_mf_bs_grid_geoms_id.keys():
                activated_sensors.append(uspa44_mf_bs_grid_geoms_id[g2])
            elif g2 in uspa44_mf_px_grid_geoms_id.keys():
                activated_sensors.append(uspa44_mf_px_grid_geoms_id[g2])
            elif g2 in uspa44_mf_md_grid_geoms_id.keys():
                activated_sensors.append(uspa44_mf_md_grid_geoms_id[g2])
            elif g2 in uspa44_rf_bs_grid_geoms_id.keys():
                activated_sensors.append(uspa44_rf_bs_grid_geoms_id[g2])
            elif g2 in uspa44_rf_px_grid_geoms_id.keys():
                activated_sensors.append(uspa44_rf_px_grid_geoms_id[g2])
            elif g2 in uspa44_rf_md_grid_geoms_id.keys():
                activated_sensors.append(uspa44_rf_md_grid_geoms_id[g2])
            elif g2 in th_bs_grid_geoms_id.keys():
                activated_sensors.append(th_bs_grid_geoms_id[g2])
            elif g2 in th_px_grid_geoms_id.keys():
                activated_sensors.append(th_px_grid_geoms_id[g2])
            elif g2 in uSCuALHA_if_grid_geoms_id.keys():
                activated_sensors.append(uSCuALHA_if_grid_geoms_id[g2])
            elif g2 in uSCuALHA_mf_grid_geoms_id.keys():
                activated_sensors.append(uSCuALHA_mf_grid_geoms_id[g2])
            elif g2 in uSCuALHA_rf_grid_geoms_id.keys():
                activated_sensors.append(uSCuALHA_rf_grid_geoms_id[g2])
            elif g2 in uSCuALHA_th_grid_geoms_id.keys():
                activated_sensors.append(uSCuALHA_th_grid_geoms_id[g2])
        elif g2 == cube_collision_id:
            if g1 in uspa46_1_grid_geoms_id.keys():
                activated_sensors.append(uspa46_1_grid_geoms_id[g1])
            elif g1 in uspa46_2_grid_geoms_id.keys():
                activated_sensors.append(uspa46_2_grid_geoms_id[g1])
            elif g1 in uspa46_3_grid_geoms_id.keys():
                activated_sensors.append(uspa46_3_grid_geoms_id[g1])
            elif g1 in uspa44_if_bs_grid_geoms_id.keys():
                activated_sensors.append(uspa44_if_bs_grid_geoms_id[g1])
            elif g1 in uspa44_if_px_grid_geoms_id.keys():
                activated_sensors.append(uspa44_if_px_grid_geoms_id[g1])
            elif g1 in uspa44_if_md_grid_geoms_id.keys():
                activated_sensors.append(uspa44_if_md_grid_geoms_id[g1])
            elif g1 in uspa44_mf_bs_grid_geoms_id.keys():
                activated_sensors.append(uspa44_mf_bs_grid_geoms_id[g1])
            elif g1 in uspa44_mf_px_grid_geoms_id.keys():
                activated_sensors.append(uspa44_mf_px_grid_geoms_id[g1])
            elif g1 in uspa44_mf_md_grid_geoms_id.keys():
                activated_sensors.append(uspa44_mf_md_grid_geoms_id[g1])
            elif g1 in uspa44_rf_bs_grid_geoms_id.keys():
                activated_sensors.append(uspa44_rf_bs_grid_geoms_id[g1])
            elif g1 in uspa44_rf_px_grid_geoms_id.keys():
                activated_sensors.append(uspa44_rf_px_grid_geoms_id[g1])
            elif g1 in uspa44_rf_md_grid_geoms_id.keys():
                activated_sensors.append(uspa44_rf_md_grid_geoms_id[g1])
            elif g1 in th_bs_grid_geoms_id.keys():
                activated_sensors.append(th_bs_grid_geoms_id[g1])
            elif g1 in th_px_grid_geoms_id.keys():
                activated_sensors.append(th_px_grid_geoms_id[g1])
            elif g1 in uSCuALHA_if_grid_geoms_id.keys():
                activated_sensors.append(uSCuALHA_if_grid_geoms_id[g1])
            elif g1 in uSCuALHA_mf_grid_geoms_id.keys():
                activated_sensors.append(uSCuALHA_mf_grid_geoms_id[g1])
            elif g1 in uSCuALHA_rf_grid_geoms_id.keys():
                activated_sensors.append(uSCuALHA_rf_grid_geoms_id[g1])
            elif g1 in uSCuALHA_th_grid_geoms_id.keys():
                activated_sensors.append(uSCuALHA_th_grid_geoms_id[g1])
        


        # name1 = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, g1)
        # name2 = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, g2)
        # print(f"Contact {i}: {name1} and {name2}")
      
      print(f"activated_sensors::{set(activated_sensors)}")
      activated_sensors = []
      viewer.sync()