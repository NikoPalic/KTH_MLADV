import numpy as np
from numpy import genfromtxt
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
from numpy.linalg import eig, norm
from scipy.linalg import sqrtm
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.neighbors import kneighbors_graph
import networkx as nx

#Load data
names=np.array(pd.read_csv('zoo.data', header=None).iloc[:,0])
my_data = genfromtxt('zoo.data', delimiter=',')
my_data = np.array(my_data)[:,1:]
labels = my_data[:,16]
#print(labels)
my_data = my_data[:,:16] #as logical
#print(my_data)
my_trans = np.transpose(my_data) #as in lectures
#print(my_trans)
N=len(my_data); M=len(my_data[0]) #N is no. of smaples, M is no. of features

OPTIONS = {"PCA":False,
           "MDS":False,
           "MDS_distance": False,
           "MDS_distance_scale":False,
           "ISOMap":True}


#######################################################################################################
def plot(X, label):
    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1],c=labels)
    ax.set_ylabel(label+"2"); ax.set_xlabel(label+"1")

    #annotate
    for i, name in enumerate(names):
       ax.annotate(name,(X[i, 0], X[i, 1]), fontsize=9)

    plt.show()

#######################################################################################################
#PCA1
if OPTIONS["PCA"]:
    pca = PCA(n_components=2, random_state=0)
    pca.fit(my_data)
    print("PCA: variance:", pca.explained_variance_)
    print("PCA: variance ratio:", pca.explained_variance_ratio_, "Total: ", sum(pca.explained_variance_ratio_))
    reduced = pca.transform(my_data)
    #print(reduced)
    plot(reduced, "PCA")

#######################################################################################################
#MDS
def classicMDS(S, label="MDS"):
    w, v = eig(S)  # w=eigenvalues v=eigenvectors
    w = np.diag(w)
    w = sqrtm(w)

    X = np.dot(w, np.transpose(v))
    X = np.real(X)  # remove complex part
    reduced = np.transpose(X[:2, :])  # revert back to logical

    plot(reduced, label)

#######################################################################################################
#distance approach
def build_distance(Y):
    D= []
    for i in range(N):
        D.append([])
        for j in range(N):
            D[i].append(norm(Y[:,i]-Y[:,j])**2) #have to do like this because Y is transposed in lectures
    return np.array(D)

#weighted distance approach
def feature_scale(Y, weighting=[]):
    if len(weighting)==0: #custom scaling
        for feat_index in [1]:
            Y[feat_index, :]*=20
    else:
        k=2
        weighting = [(val, i) for i, val in enumerate(weighting)]
        sorted_weighting = list(reversed(sorted(weighting)))
        sorted_weighting = sorted_weighting[:k] #take k first best features
        #print(sorted_weighting)
        for i in range(k):
            feat_index = sorted_weighting[i][1]
            Y[feat_index,:]*=sorted_weighting[i][0]
    return Y

def feature_select(X, labels):
    X = SelectKBest(chi2, k=10).fit(X,labels)
    #print(X.scores_)
    return X.scores_

#######################################################################################################
#MDS driver code

if OPTIONS["MDS"] or OPTIONS["MDS_distance"] or OPTIONS["MDS_distance_scale"]:
    Y=np.transpose(my_data) #as in lectures

    #center the data (sum of columns = 0)
    center_mat = np.identity(N)-1./N*np.ones(N)
    Y=np.dot(Y,center_mat)

    if OPTIONS["MDS"]:
        #standard approach with known Y
        S=np.dot(np.transpose(Y), Y)
        classicMDS(S)

    if OPTIONS["MDS_distance"] or OPTIONS["MDS_distance_scale"]:
        if OPTIONS["MDS_distance_scale"]:
            feat_scores = feature_select(my_data, labels)
            Y = feature_scale(Y, feat_scores)

        D = build_distance(Y)
        mean_row = np.vstack(np.mean(D, axis=1))
        mean_col = np.mean(D, axis=0)

        S = -0.5 * (D - mean_row - mean_col + np.mean(D))
        classicMDS(S)

#######################################################################################################
#ISOMap

def print_csr(neighbor_distances): #for printing purposes
    d = neighbor_distances.todok()  # convert to dictionary of keys format
    Z=dict(d.items())
    print(Z)
    for key in sorted(Z.keys(), key=lambda x:x[0]):
        print(key, Z[key])

def build_iso_distance(vec_distances): #takes sparse dictionary of datapoint distances as input
    D=[]
    for data_point in vec_distances.keys(): #data_point should be integer id of a sample
        D.append([])
        for neighbour in range(N):
            if neighbour in vec_distances[data_point]:
                D[data_point].append(vec_distances[data_point][neighbour])
            else:
                D[data_point].append(20) #unreachable in the ISOMap graph
    return np.array(D)

if OPTIONS["ISOMap"]:
    p=10

    neighbor_distances = kneighbors_graph(my_data, p, mode='distance')
    network = nx.Graph(neighbor_distances)
    vec_distances=nx.all_pairs_dijkstra_path_length(network)
    vec_distances = dict(vec_distances)

    #build distance matrix from vector distances
    D=build_iso_distance(vec_distances)

    mean_row = np.vstack(np.mean(D, axis=1))
    mean_col = np.mean(D,axis=0)

    S = -0.5*(D-mean_row - mean_col + np.mean(D))
    classicMDS(S, "ISO")