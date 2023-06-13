import torch
import pytorch3d 
# from pytorch3d.ops import cot_laplacian
# from pytorch3d.structures import Meshes

from .utils import one_hot_sparse

def faces_angle(meshs):
    Face_coord = meshs.verts_packed()[meshs.faces_packed()]
    A = Face_coord[:,1,:] - Face_coord[:,0,:]
    B = Face_coord[:,2,:] - Face_coord[:,1,:]
    C = Face_coord[:,0,:] - Face_coord[:,2,:]
    angle_0 = torch.arccos(-torch.sum(A*C,dim=1)/torch.norm(A,dim=1)/torch.norm(C,dim=1))
    angle_1 = torch.arccos(-torch.sum(A*B,dim=1)/torch.norm(A,dim=1)/torch.norm(B,dim=1))
    angle_2 = torch.arccos(-torch.sum(B*C,dim=1)/torch.norm(B,dim=1)/torch.norm(C,dim=1))
    angles = torch.stack([angle_0,angle_1,angle_2],dim=1)
    return angles

def dual_area_weights_on_faces(Surfaces):
    angles = faces_angle(Surfaces)
    sin2angle = torch.sin(2*angles)
    dual_area_weight = torch.ones_like(Surfaces.faces_packed())*(torch.sum(sin2angle,dim=1).view(-1,1).repeat(1,3))
    for i in range(3):
        j,k = (i+1)%3, (i+2)%3
        dual_area_weight[:,i] = 0.5*(sin2angle[:,j]+sin2angle[:,k])/dual_area_weight[:,i]
    return dual_area_weight


def Dual_area_for_vertices(Surfaces):
    dual_area_weight = dual_area_weights_on_faces(Surfaces)
    dual_area_faces = Surfaces.faces_areas_packed().view(-1,1).repeat(1,3)*dual_area_weight
    face_vertices_to_idx = one_hot_sparse(Surfaces.faces_packed().view(-1),num_classes=Surfaces.num_verts_per_mesh().sum())
    dual_area_vertex = torch.sparse.mm(face_vertices_to_idx.float().T,dual_area_faces.view(-1,1)).T
    return dual_area_vertex


def Gaussian_curvature(Surfaces,return_topology=False):
    face_vertices_to_idx = one_hot_sparse(Surfaces.faces_packed().view(-1),num_classes=Surfaces.num_verts_per_mesh().sum())
    vertices_to_meshid = one_hot_sparse(Surfaces.verts_packed_to_mesh_idx(),num_classes=Surfaces.num_verts_per_mesh().shape[0])
    sum_angle_for_vertices = torch.sparse.mm(face_vertices_to_idx.float().T,faces_angle(Surfaces).view(-1,1)).T
    # Euler_chara = torch.sparse.mm(vertices_to_meshid.float().T,(2*torch.pi - sum_angle_for_vertices).T).T/torch.pi/2
    # Euler_chara = torch.round(Euler_chara)
    # print('Euler_characteristic:',Euler_chara)
    # Genus = (2-Euler_chara)/2
    #print('Genus:',Genus)
    gaussian_curvature = (2*torch.pi - sum_angle_for_vertices)/Dual_area_for_vertices(Surfaces)
    if return_topology:
        Euler_chara = torch.sparse.mm(vertices_to_meshid.float().T,(2*torch.pi - sum_angle_for_vertices).T).T/torch.pi/2
        Euler_chara = torch.round(Euler_chara)
        return gaussian_curvature, Euler_chara, Genus
    return gaussian_curvature

def Average_from_verts_to_face(Surfaces, vect_verts):
    assert vect_verts.shape[0] == Surfaces.verts_packed().shape[0]
    dual_weight = dual_area_weights_on_faces(Surfaces).view(-1)
    wg = one_hot_sparse(Surfaces.faces_packed().view(-1),num_classes=Surfaces.num_verts_per_mesh().sum(),value=dual_weight).float()
    return torch.sparse.mm(wg,vect_verts).view(-1,3).sum(dim=1)
