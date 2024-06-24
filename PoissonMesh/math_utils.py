import numpy as np
from tqdm import tqdm
from scipy import sparse
from sklearn.neighbors import KDTree
from sklearn.cluster import KMeans,MeanShift

#Orthogonalisation Process
def gram_schmidt(vectors):
    basis = np.zeros_like(vectors)
    for i in tqdm(range(vectors.shape[1]), desc="Calculating basis"):
        temp = vectors[:, i]
        temp -= np.sum(np.array([np.sum(temp*basis[:,k])*basis[:,k] for k in range(i)]),axis=0)
        norm = np.linalg.norm(temp)
        if norm != 0:
            basis[:, i] = temp/norm
    return basis

def implicit_smooth(lam, M, C, p):
    p_new = p.copy()
    B = (M@p).copy()
    p_new[:,0],_ = sparse.linalg.cg(M - lam*C, B[:,0])
    p_new[:,1],_ = sparse.linalg.cg(M - lam*C, B[:,1])
    p_new[:,2],_ = sparse.linalg.cg(M - lam*C, B[:,2])
    return p_new

def segmentation(embedding, method='kmeans', num_clusters=5):
    if method == 'kmeans':
        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(embedding)
        labels = kmeans.labels_
    elif method == 'meanshift':
        meanshift = MeanShift()
        meanshift.fit(embedding)
        labels = meanshift.labels_

    n_labels = len(np.unique(labels))
    print(f"Number of clusters: {n_labels}")
    # assign colours to labels
    colours = np.random.rand(n_labels,3)
    vertex_colours = np.zeros((embedding.shape[0],3))
    for i in range(num_clusters):
        indices = np.where(labels == i)[0]
        vertex_colours[indices] = colours[i]
    return vertex_colours

def approx(k,X,basis):
    X_twiddle = basis.T @ X#k x 25000 times 25000 x 3 is k x 3
    return np.sum(np.array([np.reshape(basis[:,i], (-1,1)) @ np.reshape(X_twiddle[i,:],(1,-1)) for i in range(k)]),axis=0)

def project_onto_basis(vertices, basis):
    # vertices is n x 3, basis is n x k
    # Project vertices onto the basis to get the coefficients
    coefficients = basis.T @ vertices  # k x n X n x 3
    return coefficients # k x 3

def recon(coefficients, basis):
    k = basis.shape[1]
    assert coefficients.shape[0] == k
    return np.sum(np.array([np.reshape(basis[:,i], (-1,1)) @ np.reshape(coefficients[i,:],(1,-1)) for i in range(k)]),axis=0)

def find_correspondences(source_boundary, target_boundary):
    # source_boundary and target_boundary are a list of edge vertices from source_mesh and target_mesh
    # find the closest points of target_boundary in source_boundary
    # return the target_boundary indices that are the closest to each source_boundary point
    # create a KDTree from source_boundary
    target_tree = KDTree(target_boundary)
    _, indices = target_tree.query(source_boundary)
    # return the source_boundary points that are the closest to each target_boundary point
    return indices
