""" This file is created as a template for question 2.4 in DD2434 - Assignment 2.
    We encourage you to keep the function templates as is.
    However this is not a "must" and you can code however you like.
    You can write helper functions however you want.
    You do not have to implement the code for finding a maximum spanning tree from scratch. We provided two different
    implementations of Kruskal's algorithm and modified them to return maximum spanning trees as well as the minimum
    spanning trees. However, it will be beneficial for you to try and implement it. You can also use another
    implementation of maximum spanning tree algorithm, just do not forget to reference the source (both in your code
    and in your report)! Previously, other students used NetworkX package to work with trees and graphs, keep in mind.
    We also provided an example regarding the Robinson-Foulds metric (see Phylogeny.py).
    If you want, you can use the class structures provided to you (Node, Tree and TreeMixture classes in Tree.py file),
    and modify them as needed. In addition to the sample files given to you, it is very important for you to test your
    algorithm with your own simulated data for various cases and analyse the results.
    For those who do not want to use the provided structures, we also saved the properties of trees in .txt and .npy
    format.
    Note that the sample files are tab delimited with binary values (0 or 1) in it.
    Each row corresponds to a different sample, ranging from 0, ..., N-1
    and each column corresponds to a vertex from 0, ..., V-1 where vertex 0 is the root.
    Example file format (with 5 samples and 4 nodes):
    1   0   1   0
    1   0   1   0
    1   0   0   0
    0   0   1   1
    0   0   1   1
    Also, I am aware that the file names and their extensions are not well-formed, especially in Tree.py file
    (i.e example_tree_mixture.pkl_samples.txt). I wanted to keep the template codes as simple as possible.
    You can change the file names however you want (i.e tmm_1_samples.txt).
    For this assignment, we gave you a single tree mixture (q2_4_tree_mixture).
    The mixture has 3 clusters, 5 nodes and 100 samples.
    We want you to run your EM algorithm and compare the real and inferred results
    in terms of Robinson-Foulds metric and the likelihoods.
    """
import dendropy
import numpy as np
import matplotlib.pyplot as plt
from Tree import Tree, TreeMixture
from sklearn.preprocessing import normalize
from math import log
from Kruskal import Graph

def tm_likelihood(tm, samples, N, K): #calulate log likelihood; l=sum_n{log(sum_k{pi_k*p(x_vec|tree_k)})}
    log_hood=0
    for n in range(N):
        suma = 0
        for k in range(K):
            nth_sample = samples[n]
            prob = tree_sample_likelihood(tm.clusters[k], nth_sample)
            pi_k = tm.pi[k]
            suma += pi_k * prob
        log_hood += log(suma)
    return log_hood

def tree_sample_likelihood(tree, sample):
    def adjust_tree(curr):
        curr.cat = np.array(curr.cat)
        curr.name = int(curr.name)

        for descendant in curr.descendants:
            adjust_tree(descendant)

    adjust_tree(tree.root)

    def recur(curr, sample):
        product = 1
        for descendant in curr.descendants: #gather cond probabilities from children
            product*=recur(descendant, sample)
        if curr==tree.root:
            return product*curr.cat[sample[curr.name]]
        ancestor_observation = sample[curr.ancestor.name]; curr_observation = sample[curr.name]
        curr_contribution = curr.cat[ancestor_observation,curr_observation]
        return product*curr_contribution

    likelihood = recur(tree.root, sample)
    return likelihood

def em_algorithm(seed_val, samples, num_clusters, max_num_iter=10, tm = None):
    """
    This function is for the EM algorithm.
    :param seed_val: Seed value for reproducibility. Type: int
    :param samples: Observed x values. Type: numpy array. Dimensions: (num_samples, num_nodes)
    :param num_clusters: Number of clusters. Type: int
    :param max_num_iter: Maximum number of EM iterations. Type: int
    :return: loglikelihood: Array of log-likelihood of each EM iteration. Type: numpy array.
                Dimensions: (num_iterations, ) Note: num_iterations does not have to be equal to max_num_iter.
    :return: topology_list: A list of tree topologies. Type: numpy array. Dimensions: (num_clusters, num_nodes)
    :return: theta_list: A list of tree CPDs. Type: numpy array. Dimensions: (num_clusters, num_nodes, 2)
    This is a suggested template. Feel free to code however you want.
    """
    # Set the seed
    np.random.seed(seed_val)

    # TODO: Implement EM algorithm here.
    N = len(samples); K = num_clusters; V = samples.shape[1]
    if tm == None:
        tm = TreeMixture(num_clusters=num_clusters, num_nodes=samples.shape[1])
        tm.simulate_pi(seed_val=seed_val)
        tm.simulate_trees(seed_val=seed_val)
    log_hoods=[]

    for iteration in range(max_num_iter):

        #STEP 1
        R=np.zeros(shape=(N,K))
        for n in range(N):
            for k in range(K):
                nth_sample = samples[n]
                kth_tree = tm.clusters[k]
                hood = tree_sample_likelihood(kth_tree, nth_sample)
                R[n,k]=tm.pi[k]*hood
        R = normalize(R, axis=1, norm='l1')

        #STEP 2
        new_pi = np.zeros(shape=(K))
        for k in range (K):
            suma=0
            for n in range(N):
                suma+=R[n,k]
            new_pi[k]=suma/N
        tm.pi=new_pi

        for k in range(K):
            #STEP 3
            Qstab=np.zeros(shape=(V,V,2,2)) #Xs x Xt x (0 or 1) x (0 or 1)
            Nstab = np.zeros(shape=(V,V,2,2)) #Xs x Xt x (0 or 1) x (0 or 1)
            #2 vertex relation
            for Xs in range(V): #foreach vertex pair
                for Xt in range(V):
                    if Xs == Xt:
                        continue
                    for n in range(N):
                        a = samples[n][Xs]
                        b = samples[n][Xt]
                        r_nk = R[n,k]
                        Nstab[Xs,Xt,a,b]+=r_nk
            for Xs in range(V): #foreach vertex pair
                for Xt in range(V):
                    if Xs == Xt:
                        continue
                    denom = sum(R[:,k])
                    for a in range(2): #for each observation (0 or 1)
                        for b in range(2):
                            num = Nstab[Xs,Xt,a,b]
                            Qstab[Xs,Xt,a,b]=num/denom
            #1 vertex relation
            Qsa = np.zeros(shape=(V,2))
            Nsa = np.zeros(shape=(V,2))
            for Xs in range(V):  # foreach vertex
                for n in range(N):
                    a = samples[n][Xs]
                    r_nk = R[n, k]
                    Nsa[Xs, a] += r_nk
            for Xs in range(V):
                for a in range(2):
                    num = Nsa[Xs,a]
                    denom = sum(Nsa[Xs,:])
                    Qsa[Xs,a]=num/denom
            #mutual information
            Info = np.zeros(shape=(V,V)) #information between vertices
            for Xs in range(V): #foreach vertex pair
                for Xt in range(V):
                    if Xs == Xt:
                        continue
                    for a in range(2):
                        for b in range(2):
                            qab = Qstab[Xs,Xt,a,b]
                            qa = Qsa[Xs,a]
                            qb = Qsa[Xt, b]
                            if qab/(qa*qb)!=0:
                                Info[Xs,Xt]+=qab*log(qab/(qa*qb))
                            else:
                                Info[Xs, Xt]+=0
            #conditional information (for step 5)
            Qcond_stab = np.zeros(shape=(V,V,2,2))
            for Xs in range(V): #foreach vertex pair
                for Xt in range(V):
                    if Xs == Xt:
                        continue
                    for a in range(2):
                        for b in range(2):
                            num = Nstab[Xs, Xt, a, b]
                            denom = sum(Nstab[Xs, Xt, a, :])
                            Qcond_stab[Xs, Xt, a, b] = num / denom #p(Xt=b|Xs=a)

            #STEP 4
            g = Graph(V)
            for Xs in range(V): #foreach vertex pair
                for Xt in range(V):
                    if Xs == Xt:
                        continue
                    g.addEdge(Xs, Xt, Info[Xs,Xt])
            mst = g.maximum_spanning_tree() #this is an array
            mst = sorted(mst, key = lambda x : x[0])

            #STEP 5
            topology_array = [np.nan for i in range(V)]
            theta_array=[None for i in range(V)] #placeholder
            topology_array = np.array(topology_array); theta_array=np.array(theta_array)
            #root
            root = 0
            theta_array[0] = Qsa[root, :]

            MST={}
            for u, v, w in mst:
                if u not in MST:
                    MST[u]=[]
                MST[u].append(v)
                if v not in MST:
                    MST[v]=[]
                MST[v].append(u)

            VISITED=[]
            def dfs(curr, prior):
                VISITED.append(curr)
                if prior!=-1:
                    cat = Qcond_stab[prior,curr]
                    theta_array[curr] = cat
                    topology_array[curr] = prior

                for child in MST[curr]:
                    if child in VISITED:
                        continue
                    dfs(child, curr)

            dfs(root, -1)

            new_tree = Tree()
            #print(topology_array)
            #print(theta_array)
            new_tree.load_tree_from_direct_arrays(topology_array, theta_array)

            tm.clusters[k]=new_tree

        #print("End iteration ", iteration)
        log_hood = tm_likelihood(tm, samples, N, num_clusters)
        #print(log_hood)
        log_hoods.append(log_hood)

    loglikelihood_list = np.array(log_hoods)

    return loglikelihood_list, tm

def sieving(seed_val, samples, num_clusters, N_start_points=50):
    results=[] #(final_likelihood, tm)
    for i in range(N_start_points):
        likelihood_list, tm = em_algorithm(seed_val+i,samples, num_clusters)
        results.append((likelihood_list[-1], tm))

    results=list(reversed(sorted(results, key= lambda x: x[0])))
    results = results[:10] #take 10 best candidates

    extended_results=[] #(final_likelihood, tm)

    for i in range(len(results)):
        likelihood_list, tm = em_algorithm(seed_val + i, samples, num_clusters, max_num_iter=50, tm=results[i][1])
        extended_results.append((likelihood_list[-1], tm, likelihood_list))

    extended_results=list(reversed(sorted(extended_results, key= lambda x: x[0])))

    best = extended_results[0]
    tm = best[1]
    likelihood_list = best[2]

    return likelihood_list, tm

def main():
    print("Hello World!")

    seed_val = 123

    #sample_filename = "q2_4/q2_4_tree_mixture.pkl_samples.txt"
    #real_values_filename = "q2_4/q2_4_tree_mixture.pkl"

    #sample_filename = "q2_4/case1.pkl_samples.txt"
    #real_values_filename = "q2_4/case1.pkl"

    #sample_filename = "q2_4/case2.pkl_samples.txt"
    #real_values_filename = "q2_4/case2.pkl"

    sample_filename = "q2_4/case3.pkl_samples.txt"
    real_values_filename = "q2_4/case3.pkl"

    num_clusters = 2 #need to change this fpr each case!

    samples = np.loadtxt(sample_filename, delimiter="\t", dtype=np.int32)

    loglikelihood, my_tm = sieving(seed_val, samples, num_clusters=num_clusters)

    plt.figure(figsize=(8, 3))
    plt.subplot(121)
    plt.plot(np.exp(loglikelihood), label='Estimated')
    plt.ylabel("Likelihood of Mixture")
    plt.xlabel("Iterations")
    plt.subplot(122)
    plt.plot(loglikelihood, label='Estimated')
    plt.ylabel("Log-Likelihood of Mixture")
    plt.xlabel("Iterations")
    plt.legend(loc=(1.04, 0))
    plt.show()

    if real_values_filename != "":
        real = TreeMixture(0,0)
        real.load_mixture(real_values_filename)

        print("\t4.1. Make the Robinson-Foulds distance analysis.\n")
        tns = dendropy.TaxonNamespace()

        real_trees = [i.newick for i in real.clusters]
        my_trees = [i.newick for i in my_tm.clusters]
        print(my_trees)

        i=0;
        for real_tree in real_trees:
            real_den = dendropy.Tree.get(data=real_tree, schema="newick", taxon_namespace=tns)
            j=0
            for my_tree in my_trees:
                my_den = dendropy.Tree.get(data=my_tree, schema="newick", taxon_namespace=tns)
                print("RF distance: $<",i,j,">$\t=", dendropy.calculate.treecompare.symmetric_difference(real_den, my_den),"\\\\")
                j+=1
            i+=1

        print("4.2. Make the likelihood comparison.\n")
        real_log_hood = tm_likelihood(real, samples, len(samples), num_clusters)
        print("Real: ", real_log_hood)
        print("Infered: ", loglikelihood)

def case1():
    real = TreeMixture(7, 4)
    real.simulate_pi()
    real.simulate_trees(seed_val=123)
    real.sample_mixtures(100, seed_val=123)
    real.save_mixture("q2_4/case1.pkl")

def case2():
    real = TreeMixture(2, 5)
    real.simulate_pi()
    real.simulate_trees(seed_val=123)
    real.sample_mixtures(100, seed_val=123)
    real.save_mixture("q2_4/case2.pkl")

def case3():
    real = TreeMixture(2, 5)
    real.simulate_pi()
    real.simulate_trees(seed_val=123)
    real.sample_mixtures(30, seed_val=123)
    real.save_mixture("q2_4/case3.pkl")

if __name__ == "__main__":
    #case1()
    #case2()
    case3()
    main()