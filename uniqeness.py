import numpy as np  # used to compute mean and standard deviation
import pymnet  # library for handling networks (graphs) and multilayer networks. [Ref: http://www.mkivela.com/pymnet/]
import networkx as nx
import os

"""This code computes the average degree (average number of edges per node) of a random graph/network corresponding 
to a given percentage of unique neighborhoods structures ("uniqueness"). It does that with the function called 
"binarySearchUnique". There is also a main with an example of call of the function. The other functions are 
auxiliary. To estimate the average degree of a graph corresponding to a percentage of unique neighborhoods, 
the code generates random graphs and computes the mean and the standard deviation of the obtained uniqueness. A 
neighborhood is the subgraph of a graph that contains the neighbors of a node and the edges between those (but does 
not contain the central node itself). A neighborhood structure is unique if there are no other neighborhoods in the 
original graph that have its same structure (i.e. that are not isomorphic). The considered graphs are undirected (the 
edges have no direction), unlabelled (the nodes have no label), without self loops and without multi-edges (multiple 
edges between a pair of nodes). To handle graphs, Pymnet, a multilayer network library, is used. Thanks to this, 
one can extend the code to handle graphs with multiple layers. To compute isomorphism, Pymnet uses PyBliss (
http://www.tcs.hut.fi/Software/bliss/). It is assumed that the given interval for binary search (continuos version, 
that just needs the extreme values) contains the value we are looking for (otherwise the method fails). Large-enough 
networks in their sparse-region (such as with relatively small average degree) are considered in this problem. 
Empirical evaluations have shown that higher the average degree, higher the number of unique neighborhoods (this is 
not necessarily true in the graph dense region, i.e. when the graph is almost complete). This last consideration is 
crucial in order to make a decision regarding which new interval to consider in the recursion of the binary search."""

directory_path = "data/SNAP_facebook_clean_data"


def readNetworkFromRLD(directory_path):
    edges = []
    for filename in os.listdir(directory_path):
        filepath = os.path.join(directory_path, filename)
        with open(filepath, 'r') as file:
            lines = file.readlines()[2:]  # Skip the first line (header)
            for line in file:
                try:
                    edge = tuple(map(int, line.strip().split()))
                    edges.append(edge)
                except ValueError:
                    print(f"Skipping line with non-integer values: {line.strip()}")

    RDatasetGraph = nx.Graph(edges)
    return RDatasetGraph