import numpy as np
from tqdm import tqdm
from scipy import sparse

def uniform_laplace(mesh):
    """
    Calculates the uniform laplacian of a mesh with num_vertices vertices and vertex_neighbours as its one rings.
    """
    vertices = mesh.vertices
    vertex_neighbours = mesh.vertex_neighbors
    num_vertices = len(vertices)
    L = sparse.lil_matrix((num_vertices,num_vertices))
    for i in range(num_vertices):
        L[i,vertex_neighbours[i]] = 1/len(vertex_neighbours[i])
        L[i,i] = -1
    L = sparse.csr_matrix(L)
    return L

def mean_curvature(mesh,laplaceFunc="uniform"):
    """
    Calculates the mean curvature H of the mesh — ∆x/2 
    """   
    vertices = mesh.vertices
    normals = mesh.vertex_normals

    if laplaceFunc == "uniform":
        laplace_on_coords = uniform_laplace(mesh)@vertices
        signs = np.sign(np.sum(normals*laplace_on_coords, axis = 1))*(-1)
        H = 0.5*np.sum(laplace_on_coords **2,axis=1)
        H = H*signs

    else:
        M_inv, C = laplace_beltrami_operator(mesh)
        laplace_on_coords = M_inv@C@vertices
        H = 0.5*np.sum(laplace_on_coords **2,axis=1)

    return H
    
def gauss_curvature(mesh):
    """
    Gaussian curvature K of the mesh
    """    
    vertices = mesh.vertices
    vertex_neighbours = mesh.vertex_neighbors

    gaussCurvature = np.zeros(vertices.shape[0])
    for i in range(vertices.shape[0]):
        neigh = vertex_neighbours[i]
        numNeigh = len(neigh)
        
        # find the a*b*cos(theta) for each vertex with its neighbours
        dot_products = np.array([np.sum((vertices[i,:] - vertices[neigh[i],:])*(vertices[i,:] - vertices[neigh[(i+1)%numNeigh],:]))
                            for i in range(numNeigh)])
        # find ||a*b|| for each vertex with its neighbours i.e., its magnitude
        magnitudes = np.array([
            (np.linalg.norm(vertices[i,:] - vertices[neigh[i],:])*np.linalg.norm(vertices[i,:] - vertices[neigh[(i+1)%numNeigh],:])) 
                            for i in range(numNeigh)])
        # find cos(thetas) for each vertex with its neighbours
        cosines = np.clip(dot_products/magnitudes, -1,1)
        # find sin(thetas) for each vertex with its neighbours
        sines = np.clip((1 - cosines**2)**0.5, -1, 1)
        # angle deficit = 2pi - sum of angles around a vertex
        angle_deficit = 2*np.pi - np.sum(np.arccos(cosines))
        # total area of the triangle fan around a vertex (one ring neighbourhood) = sum of (1/2)absin(theta) for each triangle
        total_area = np.sum(0.5*magnitudes*sines)
        # normalize the local neighbourhood area by dividing by 3
        gaussCurvature[i] = angle_deficit/(total_area/3.0)
    return gaussCurvature

def sort_onerings(mesh):
    face_neighbours = mesh.vertex_faces
    faces = mesh.faces
    sorted_nbrs = []
    edge_vertices = np.zeros(face_neighbours.shape[0], dtype=bool)
    
    for i in range(face_neighbours.shape[0]):
        my_face_neighbours = [face_neighbours[i,j] for j in range(face_neighbours.shape[1]) ]
        
        if -1 in my_face_neighbours:
            my_face_neighbours.remove(-1)
        
        curF = faces[my_face_neighbours[0]]
        curV, nxtV = sorted([v for v in curF if v != i])[:2]
        V0, V1 = curV.copy(), nxtV.copy()
        
        my_sorted_nbrs = [curV]
        options = [face for face in my_face_neighbours if
                   (nxtV in faces[face] and not curV in faces[face]) and i in faces[face]]        
        
        while len(options)>0 and nxtV!=V0:
            curF = faces[options[0]]
            curV = nxtV.copy()
            nxtV = min([v for v in curF if (v != i and v != curV)])
            my_sorted_nbrs.append(curV)

            options = [fn for fn in my_face_neighbours if nxtV in faces[fn] and curV not in faces[fn]]
        
        if nxtV != V0:
            my_sorted_nbrs.append(nxtV)
            edge_vertices[i] = True
            nxtV = V0

            options = [face for face in my_face_neighbours if
                   (V0 in faces[face] and not V1 in faces[face]) and i in faces[face]] 
            
            nxtV = V0.copy()

            while len(options) > 0:
                curF = faces[options[0]]
                curV, nxtV = nxtV, min(v for v in curF if v != i and v != curV)
                
                if curV != V0:
                    my_sorted_nbrs.insert(0, curV)

                options = [face for face in my_face_neighbours if
                   (nxtV in faces[face] and not curV in faces[face])] 

            my_sorted_nbrs.insert(0, nxtV)
        sorted_nbrs.append(my_sorted_nbrs)
    print('Done.')
    return sorted_nbrs, edge_vertices

def laplace_beltrami_operator(mesh, onerings, edge_vertices, ignore_boundary=True):
    print('Calculating Laplace-Beltrami operator...')
    v = mesh.vertices
    M = sparse.lil_matrix((v.shape[0], v.shape[0]))
    Minv = sparse.lil_matrix((v.shape[0], v.shape[0]))
    C = sparse.lil_matrix((v.shape[0], v.shape[0]))
    
    for j in tqdm(range(v.shape[0]), desc="Processing LB operator"):
        my_nbrs = onerings[j]
        if not my_nbrs:
            continue
        
        if edge_vertices[j] and ignore_boundary:
            continue
        
        if edge_vertices[j]:
            magnitudes = np.linalg.norm(v[j] - v[my_nbrs[[0, -1]]], axis=1)
            C[j, my_nbrs[0]] = 1 / magnitudes[0]
            C[j, my_nbrs[-1]] = 1 / magnitudes[1]
            C[j, j] = -np.sum(1 / magnitudes)
            M[j, j] = np.sum(magnitudes) / 2
            Minv[j, j] = 2 / np.sum(magnitudes)
            continue
        
        cotans = np.zeros(len(my_nbrs))
        total_area = 0
        
        for i in range(len(my_nbrs)):
            vi, vj = v[j], v[my_nbrs[i]]
            vk = v[my_nbrs[(i + 1) % len(my_nbrs)]]
            vl = v[my_nbrs[(i - 1) % len(my_nbrs)]]
            
            cross_vk = np.cross(vi - vk, vj - vk)
            cross_vl = np.cross(vi - vl, vj - vl)
            norm_vk = np.linalg.norm(cross_vk)
            norm_vl = np.linalg.norm(cross_vl)
            
            if norm_vk > 1e-8:
                cotan1 = np.dot(vi - vk, vj - vk) / norm_vk
            else:
                cotan1 = 0.0
                
            if norm_vl > 1e-8:
                cotan2 = np.dot(vi - vl, vj - vl) / norm_vl
            else:
                cotan2 = 0.0
            
            cotans[i] = cotan1 + cotan2
            
            area = 0.5 * np.linalg.norm(np.cross(vi - vj, vi - vk))
            total_area += area
        
        for i, ni in enumerate(my_nbrs):
            C[j, ni] = cotans[i]
        C[j, j] = -np.sum(cotans)
        M[j, j] = 2 * total_area / 3
        Minv[j, j] = 3 / (2 * total_area)
    
    M = sparse.csr_matrix(M)
    Minv = sparse.csr_matrix(Minv)
    C = sparse.csr_matrix(C)
    L = Minv @ C
    
    return L, M, Minv, C
    
    return L, M, Minv, C