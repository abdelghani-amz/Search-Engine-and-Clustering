
import numpy as np

euclideanDist = lambda x,y : np.linalg.norm(x-y)
def getNeighbors(dataset, point, minDist):
    indices = []
    for i in range(dataset.shape[0]):
        if euclideanDist(dataset[point], dataset[i]) <= minDist and point!= i  :
            indices.append(i)
    return indices


def dbscan(dataset, minDist, minNeighbors):
    clusters = np.full(dataset.shape[0], -1, dtype=np.int8)
    visited = np.zeros(dataset.shape[0], dtype = np.int8)
    c = 0
    
    for i in range(dataset.shape[0]):
        
        if visited[i] == 0:
            visited[i] = 1 
            neighbors = getNeighbors(dataset, i, minDist)
                
            if len(neighbors) >= minNeighbors:
                clusters[i] = c
                j = 0
                while j < len(neighbors):
                    if visited[neighbors[j]] == 0:
                        visited[neighbors[j]] = 1
                        neighbors2 = getNeighbors(dataset, neighbors[j], minDist)
                            
                        if len(neighbors2) >= minNeighbors:
                            for k in neighbors2:
                                if visited[k] == 0 :
                                    neighbors.append(k)
                    
                    if clusters[neighbors[j]] == -1:
                        clusters[neighbors[j]] = c

                    j = j + 1
                
                c = c + 1

    np.savetxt("clusters.csv",clusters, fmt='%d', delimiter=",")

    extractProbabilities(dataset, clusters)
    return clusters


def extractProbabilities(dataset, clusters):

    """
        c : Number of clusters
        clusters : array containing the cluster of each element in dataset
    """

    c = np.unique(clusters).max() + 1
    probas = np.ones((c,dataset.shape[1]))
    for i in range(c):
        for j in range(dataset.shape[1]):
            probas[i,j] += np.intersect1d(np.where(clusters == i)[0], np.where(dataset[:, j] > 0)[0]).size 

    cluster_counts = np.reshape(np.bincount(np.delete(clusters, np.where(clusters < 0))), (c, 1))
    probas = probas / (cluster_counts + 1)
    probas = np.column_stack((probas, (cluster_counts + 1) / (dataset.shape[0] + c)) )

    np.savetxt("probas.csv",probas, fmt='%.8f', delimiter=",")



def bayesianInference(samples):
    probas = np.genfromtxt("probas.csv", delimiter=",", dtype=np.float128)
    if len(probas.shape) == 1:
        probas = np.reshape(probas, (1, probas.shape[0]))
    labels = np.zeros(samples.shape[0])
    activation = np.zeros(probas.shape[0], np.float128)

    for i in range(samples.shape[0]):
        for c in range(activation.shape[0]):
            activation[c] = probas[c, -1] * np.prod(probas[c, np.where(samples[i] > 0)[0]]) * np.prod(1 - probas[c, np.where(samples[i] == 0)[0]])
        
        labels[i] = np.argmax(activation)

    return labels
