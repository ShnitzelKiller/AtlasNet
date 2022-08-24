import torch

def sample_point_cloud(n, v, f, use_normals=False):
    #print(f.shape)
    v0 = v[f[0,:]]
    v1 = v[f[1,:]]
    v2 = v[f[2,:]]

    normals = (v1 - v0).cross(v2 - v0)
    area = normals.norm(dim=1)
    normals /= area.unsqueeze(-1)

    sum_area = area.sum()#dim=1)
    p_area = area / sum_area
    tris = p_area.multinomial(n, replacement=True)

    v0 = v0[tris]
    v1 = v1[tris]
    v2 = v2[tris]

    u = torch.rand(n, 1)
    v = torch.rand(n, 1)
    bounds_check = u + v > 1
    u[bounds_check] = 1 - u[bounds_check]
    v[bounds_check] = 1 - v[bounds_check]
    w = 1 - (u + v)

    pc = u * v0 + v * v1 + w * v2

    if use_normals:
        return pc.float(), normals[tris].float(), tris
    else:
        return pc.float(), tris
    

