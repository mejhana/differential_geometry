import os
import trimesh 
import open3d as o3d
import numpy as np
from scipy import sparse
from PoissonMesh.laplace_beltrami_operator import laplace_beltrami_operator, sort_onerings
from PoissonMesh.render_utils import viz_3d, trimeshToO3D, visualize
from PoissonMesh.math_utils import *

def load_eigen_and_basis(mesh_name, n=1000, load=True):
    # create eigen/{mesh_name} directory
    if not os.path.exists(f"eigen/{mesh_name}"):
        os.makedirs(f"eigen/{mesh_name}")

    eigenval_path = f"eigen/{mesh_name}/eigenvalues.npy"
    eigenvec_path = f"eigen/{mesh_name}/eigenvectors.npy"
    basis_path = f"eigen/{mesh_name}/basis.npy"

    if os.path.exists(eigenvec_path) and os.path.exists(basis_path) and load:
        # simply load the eigenvalues, eigenvectors and basis
        print('Loading eigen values, eigen vectors and basis...')
        eigvals = np.load(eigenval_path, allow_pickle=True, fix_imports=True)
        eigvecs = np.load(eigenvec_path, allow_pickle=True, fix_imports=True)
        basis = np.load(basis_path, allow_pickle=True, fix_imports=True)

    else: 
        # either the eigenvalues or the eigenvectors or the basis is missing or load is False
        # load_mesh 
        mesh = trimesh.load(f"meshes/{mesh_name}.obj")
        onerings, edge_vertices = sort_onerings(mesh)
        L, M, Minv, C = laplace_beltrami_operator(mesh, onerings, edge_vertices)

        if (not os.path.exists(eigenvec_path) or not load):
            # to ensure that the matrix is symmetric, and the eigenvalues are orthogonal 
            D = np.sqrt(Minv)@C@np.sqrt(Minv)
    
            print('Calculating eigen values and eigen vectors...')
            eigvals,eigvecs = sparse.linalg.eigsh(D,k=n,which='SM')
            indices = np.argsort(np.abs(eigvals))
            eigvecs = eigvecs[:,indices]
            eigvals = eigvals[indices]

            eigvecs = eigvecs/(np.sum(eigvecs**2, axis = 0)**0.5)

            np.save(eigenval_path, eigvals, allow_pickle=True, fix_imports=True)
            np.save(eigenvec_path, eigvecs, allow_pickle=True, fix_imports=True)

            print('Calculating basis...')
            # normalize the eigenvectors
            eigvecs= np.sqrt(Minv)@eigvecs
            # 1 = ∫_M <ϕ_i,ϕ_j> dA (M = 1/A)
            basis = gram_schmidt(eigvecs)
            np.save(basis_path, basis, allow_pickle=True, fix_imports=True)

        elif not os.path.exists(basis_path) or not load:
            # load the eigenvalues and eigenvectors
            print('Loading eigen values and eigen vectors...')
            eigvals = np.load(eigenval_path, allow_pickle=True, fix_imports=True)
            eigvecs = np.load(eigenvec_path, allow_pickle=True, fix_imports=True)

            print('Calculating basis...')
            # normalize the eigenvectors
            eigvecs= np.sqrt(Minv)@eigvecs
            # 1 = ∫_M <ϕ_i,ϕ_j> dA (M = 1/A)
            basis = gram_schmidt(eigvecs)
            np.save(basis_path, basis, allow_pickle=True, fix_imports=True) 

    return eigvals, eigvecs, basis

load = True
n = 1000

operation = input("Enter the operation you want to perform: \n\t1. Implicit Smoothing \n\t2. Mesh Cloning\n\t3. Mesh Deforming\n\t4. Pose Transfer\n\t5. Mesh Segmentation\n\t6. Reconstruction\n: ")

# # lets transform all the meshes to have mean 0
# mesh_names = ["bunny", "armadillo", "homer"]
# for mesh_name in mesh_names:
#     mesh = trimesh.load(f"meshes/{mesh_name}.obj")
#     mesh.vertices -= np.mean(mesh.vertices, axis=0)
#     if mesh_name == "armadillo":
#         # rotate by 180 degrees around y axis
#         R = np.array([[np.cos(np.pi), 0, np.sin(np.pi)], [0, 1, 0], [-np.sin(np.pi), 0, np.cos(np.pi)]])
#         mesh.vertices = mesh.vertices @ R
#     mesh.export(f"meshes/{mesh_name}.obj")
    
if operation == '1':
    print("You chose implicit smoothing")
    niter = int(input("Enter the number of iterations: "))
    mesh_name = input("Enter mesh name: ")
    mesh = trimesh.load(f"meshes/{mesh_name}.obj")

    # calculate laplacian matrix
    onerings, edge_vertices = sort_onerings(mesh)
    L, M, Minv, C = laplace_beltrami_operator(mesh, onerings, edge_vertices)
    lam = 0.0001
    smooth_v = mesh.vertices
    vertices = []
    for i in range(niter):
        vertices.append(smooth_v)
        smooth_v = implicit_smooth(lam, M, C, smooth_v)
    
    # visualize the mesh
    viz_3d(mesh, vertices, "results")

elif operation == '2':
    print("You chose mesh cloning")
    mesh_1_name = input("Enter source (mesh to attach) mesh name: ")
    mesh_2_name = input("Enter target (destination) mesh name: ")

    # load the meshes using o3d
    mesh_1 = trimesh.load(f"meshes/{mesh_1_name}.obj")
    mesh_2 = trimesh.load(f"meshes/{mesh_2_name}.obj")

    print(mesh_1.vertices.shape, mesh_2.vertices.shape)

    # find boundary vertices 
    onerings, edge_vertices1 = sort_onerings(mesh_1)
    _, edge_vertices2 = sort_onerings(mesh_2)

     # find correspondences 
    source_indices = np.where(edge_vertices1)[0]
    target_indices = np.where(edge_vertices2)[0]
    new_target_indices = find_correspondences(mesh_1.vertices[source_indices], mesh_2.vertices[target_indices])
    
    assert len(source_indices) == len(new_target_indices)
    # find laplace beltrami operator for source
    L1, M1, Minv1, C1 = laplace_beltrami_operator(mesh_1, onerings, edge_vertices1)

    delta = L1 @ mesh_1.vertices

    # replace the L and delta to incorporate the boundary conditions
    L1[:, edge_vertices1] = 0
    L1[edge_vertices1, edge_vertices1] = 1
    # print(delta.shape, mesh_2.vertices[new_target_indices].shape)
    delta[edge_vertices1] = mesh_2.vertices[new_target_indices].reshape(-1, 3)

    # solve the linear system
    new_vertices = sparse.linalg.spsolve(L1, delta)

    mesh_1.vertices = new_vertices

    # convert to open3d format
    mesh_1 = trimeshToO3D(mesh_1)
    visualize(mesh_1)

elif operation == '3':
    print("You chose mesh deforming")
    pass

elif operation == '4':
    print("You chose pose transfer (please ensure that the meshes are oriented correctly)")
    mesh_1_name = input("Enter source (of pose) mesh name: ")
    mesh_2_name = input("Enter target (of pose) mesh name: ")
    print(f"Transfering the pose of {mesh_1_name} to {mesh_2_name}...")
    mesh_1 = trimesh.load(f"meshes/{mesh_1_name}.obj")
    mesh_2 = trimesh.load(f"meshes/{mesh_2_name}.obj")

    _, _, basis1 = load_eigen_and_basis(mesh_1_name,  n, load)
    _, _, basis2 = load_eigen_and_basis(mesh_2_name,  n, load)

    # ensure the vertices have std of 1 
    vertices1 = mesh_1.vertices/np.std(mesh_1.vertices)
    vertices2 = mesh_2.vertices/np.std(mesh_2.vertices)
    alpha = project_onto_basis(vertices1, basis1) # source
    beta = project_onto_basis(vertices2, basis2) # target

    k = 3
    # reconstruct the mesh (after pose transfer from mesh_1 to mesh_2) using mesh_1's low frequency and mesh_2's high frequency basis functions
    reconstruction = recon(alpha[:k], basis2[:, :k]) + recon(beta[k:], basis2[:, k:])
    
    recon_mesh = mesh_2.copy()
    recon_mesh.vertices = reconstruction

    # convert to open3d format
    recon_mesh = trimeshToO3D(recon_mesh)
    visualize(recon_mesh)

elif operation == '5':
    print("You chose mesh segmentation")
    mesh_name = input("Enter mesh name: ")
    mesh = trimesh.load(f"meshes/{mesh_name}.obj")

    _, _, basis = load_eigen_and_basis(mesh_name, n, load)

    embedding = basis[:, :5]
    colours = segmentation(embedding, method='meanshift', num_clusters=6)

    # convert mesh to Open3D format
    mesh = trimeshToO3D(mesh, colours)
    visualize(mesh)

elif operation == '6':
    print("You chose mesh reconstruction")
    mesh_name = input("Enter mesh name: ")
    mesh = trimesh.load(f"meshes/{mesh_name}.obj")

    _, _, basis = load_eigen_and_basis(mesh_name, n, load)

    reconstruction = approx(1000, mesh.vertices, basis)
    print(reconstruction.shape)
    
    recon_mesh = mesh.copy()
    recon_mesh.vertices = reconstruction

    # convert to open3d format
    recon_mesh = trimeshToO3D(recon_mesh)
    visualize(recon_mesh)

    
