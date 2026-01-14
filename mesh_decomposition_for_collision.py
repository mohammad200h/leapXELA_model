import coacd
import trimesh
import torch

# from pamo import PaMO
# https://github.com/SarahWeiii/pamo




def decompose(mesh, name):

    mesh = coacd.Mesh(mesh.vertices, mesh.faces)
    parts = coacd.run_coacd(mesh,decimate=True, max_ch_vertex=50,threshold=0.06) # a list of convex hulls.
    print(f"parts::type:: {type(parts)}")
    print(f"parts::len:: {len(parts)}")


    for i, part in enumerate(parts):
        verts, faces = part
        output_mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        output_mesh.export(f"./assets/{name}_part_{i}.obj")


if __name__ == "__main__":
    name = "outer_skin_v1_3_1_low_poly"
    mesh = trimesh.load(f"./assets/{name}.obj", force="mesh")
    
    decompose(mesh, name)