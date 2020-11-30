""" This file is created as a suggested solution template for question 2.2 in DD2434 - Assignment 2.
    We encourage you to keep the function templates as is.
    However this is not a "must" and you can code however you like.
    You can write helper functions however you want.
    If you want, you can use the class structures provided to you (Node, Tree and TreeMixture classes in Tree.py
    file), and modify them as needed. In addition to the sample files given to you, it is very important for you to
    test your algorithm with your own simulated data for various cases and analyse the results.
    For those who do not want to use the provided structures, we also saved the properties of trees in .txt and .npy
    format. Let us know if you face any problems.
    Also, we are aware that the file names and their extensions are not well-formed, especially in Tree.py file
    (i.e example_tree_mixture.pkl_samples.txt). We wanted to keep the template codes as simple as possible.
    You can change the file names however you want (i.e tmm_1_samples.txt).
    For this assignment, we gave you three different trees (q_2_2_small_tree, q_2_2_medium_tree, q_2_2_large_tree).
    Each tree has 5 samples (whose inner nodes are masked with np.nan values).
    We want you to calculate the likelihoods of each given sample and report it.
"""

import numpy as np
from Tree import Tree
from Tree import Node


def calculate_likelihood(tree, beta):
    K = len(tree.root.cat)

    def recurse(curr):
        if len(curr.descendants) == 0:
            observation = beta[int(curr.name)]
            return curr.cat[:, observation]
        vectors = []
        for descendand in curr.descendants:
            vectors.append(recurse(descendand))

        '''
        K=2:
        e.g. sum over x_2 of (p(x_2|x_1)p(x_3|x_2)p(x_4|x_2) ==>
        left = [[. .][. .]] center=[. .] right=[. .]
        foreach symbol: (left=[. .] center=. right=.)

        suma=<. .>
        '''
        suma = np.zeros(shape=(K,))
        if curr == tree.root:
            suma = 0
        for symbol in range(K):  # e.g. sum over x_2 of (p(x_2|x_1)p(x_3|x_2)p(x_4|x_2) == (left=[. .] center=. right=.)
            if curr == tree.root:
                left = curr.cat[symbol]
            else:
                left = curr.cat[:, symbol]
            center = vectors[0][symbol]
            right = 1
            if len(vectors) == 2:
                right = vectors[1][symbol]

            suma += left * center * right
        return suma

    likelihood = recurse(tree.root)  # final_vec = <. .>
    return likelihood

def adjust_tree(curr):
    curr.cat = np.array(curr.cat)
    curr.name = int(curr.name)

    for descendant in curr.descendants:
        adjust_tree(descendant)

# def adjust_beta(beta):
#     from math import isnan
#     for i in range(len(beta)):
#         if not isnan(beta[i]):
#             beta[i]=int(beta[i])
#         else:
#             beta[i]=-1


def main():
    print("Hello World!")
    print("This file is the solution template for question 2.2.")

    print("\n1. Load tree data from file and print it\n")

    filename = "q2_2/q2_2_large_tree.pkl"  # "q2_2_medium_tree.pkl", "q2_2_large_tree.pkl"
    t = Tree()
    t.load_tree(filename)
    #t.create_random_tree(1,k=5,max_num_nodes=10, max_branch=2)
    #t.sample_tree(5,0)
    #t.print()

    adjust_tree(t.root)

    print("\n2. Calculate likelihood of each FILTERED sample\n")
    # These filtered samples already available in the tree object.
    # Alternatively, if you want, you can load them from corresponding .txt or .npy files

    for sample_idx in range(t.num_samples):
        beta = t.filtered_samples[sample_idx]; beta=beta.astype("int")
        #print("\n\tSample: ", sample_idx, "\tBeta: ", beta)
        sample_likelihood = calculate_likelihood(t, beta)
        print("\tLikelihood: ", sample_likelihood)

    if (False):
        #sample for DEBUG
        t.sample_tree(100000,0)
        print(t.samples[:,[3,7,9]])
        C={}
        for i in t.samples[:,[3,7,9]]:
            k=tuple(i)
            if k not in C:
                C[k]=0
            C[k]+=1
        for sample in [(4,4,1),(4,3,4),(3,4,4),(3,4,3),(0,1,3)]:
            print(sample, C[sample]/100000.)

main()
