import os
from os import getcwd
from os.path import join
import numpy as np
from stl import mesh

def load_vx_station(delimiter=','):
    """
    load voxelized station from /models/remeshed_parts folder using OBS
    """
    meshes = []
    normals = []
    offset = -1 * np.array([0.09087864,  0.75695956, -0.10063456])
    mesh_dir = os.path.abspath(join(getcwd(),'..', 'data', 'model_remeshed'))
    meshfilename = join(mesh_dir, 'vxc_faces_normals.csv')
    faces, normals = load_single_mesh(meshfilename, offset=offset, delimiter=delimiter)
    meshes.append(faces)
    normals.append(normals)
    return meshes, normals

def load_single_mesh(mesh_file, new=False, offset=[0,0,0], scale = 4, delimiter=' '):
    """
    """
    faces_normals = np.loadtxt(mesh_file, delimiter=delimiter)
    f = []
    n = []
    offset = np.array(offset)
    for face in faces_normals:
        f.append([tuple((np.array([face[0], face[1], face[2]])+offset) * scale), 
                  tuple((np.array([face[3], face[4], face[5]])+offset) * scale), 
                  tuple((np.array([face[6], face[7], face[8]])+offset) * scale)])
        n.append([face[9], face[10], face[11]])

    return f,n

def load_meshes(convex=False):
    """
    load meshes from /models/remeshed_parts folder using OBS
    """
    meshes = []
    normals = []
    if convex:
        num_meshes = 15
        offset = [2.529, 4.821, 2.591]
        offset += np.array([1.205/4, 0, 0.775/4])
        offset -= np.array([2.92199515, 5.14701097, 2.63653781])
    else:
        num_meshes = 10
        offset = np.array([-2.92199515, -5.14701097, -2.63653781])
        # offset = [0,0,0]
    if convex:
        mesh_dir = os.path.abspath(join(getcwd(),'..', 'data', 'model_convex'))
    else:
        mesh_dir = os.path.abspath(join(getcwd(),'..', 'data', 'model_remeshed'))
    for i in range(num_meshes):
        meshfilename = join(mesh_dir, str(i) + '_faces_normals.csv')
        if convex: faces, normals = load_single_mesh(meshfilename, offset=offset, delimiter=',')
        else: faces, normals = load_single_mesh(meshfilename, offset=offset, delimiter=' ')
        meshes.append(faces)
        normals.append(normals)

    minx = np.min( [np.min(np.array(mesh).reshape(-1,3)[:,0]) for mesh in meshes])
    miny = np.min( [np.min(np.array(mesh).reshape(-1,3)[:,1]) for mesh in meshes])
    minz = np.min( [np.min(np.array(mesh).reshape(-1,3)[:,2]) for mesh in meshes])

    maxx = np.max( [np.max(np.array(mesh).reshape(-1,3)[:,0]) for mesh in meshes])
    maxy = np.max( [np.max(np.array(mesh).reshape(-1,3)[:,1]) for mesh in meshes])
    maxz = np.max( [np.max(np.array(mesh).reshape(-1,3)[:,2]) for mesh in meshes])


    return meshes

def load_original_stl():
    m = mesh.Mesh.from_file( join(getcwd(), '..', 'data', 'model_original', 'station.stl'))
    normals = m.normals

    # offset = np.array([-0.09174518423,	-0.3260109881,	0.1482121607])
    offset = np.array([-0.3669807369,	-1.304043952,	0.5928486428])
    scale = 4

    faces = []
    for i in range(len(m.v0)):
        faces.append([
            tuple(m.v0[i] * scale + offset),
            tuple(m.v1[i] * scale + offset),
            tuple(m.v2[i] * scale + offset)
       ])

    minx = np.min(np.min(np.array(faces).reshape(-1,3)[:,0]))
    miny = np.min(np.min(np.array(faces).reshape(-1,3)[:,1]))
    minz = np.min(np.min(np.array(faces).reshape(-1,3)[:,2]))

    maxx = np.max(np.max(np.array(faces).reshape(-1,3)[:,0]))
    maxy = np.max(np.max(np.array(faces).reshape(-1,3)[:,1]))
    maxz = np.max(np.max(np.array(faces).reshape(-1,3)[:,2]))

    print('Original')
    print( 'min:', minx, miny, minz)
    print( 'max:', maxx, maxy, maxz)


    return [faces]