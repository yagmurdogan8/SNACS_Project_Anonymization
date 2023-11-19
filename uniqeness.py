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
                    edge = tuple(map(int, line.strip().split("\t")))
                    edges.append(edge)
                except ValueError:
                    print(f"Skipping line with non-integer values: {line.strip()}")

    RDatasetGraph = nx.Graph(edges)
    return RDatasetGraph


def generateNetwork(size, name_model, avg_degree, model_par_prob=None):
    if name_model == 'random':
        edges = int((avg_degree * size) / 2.000)
        return pymnet.models.er(size, edges=[edges])
    elif name_model in ['er', 'ws']:
        if not model_par_prob:
            raise ValueError("model_par_prob must be provided for 'er' or 'ws' models.")
        directory_path = model_par_prob[0]
        return readNetworkFromRLD(directory_path)
    else:
        raise ValueError("Invalid network model name. Supported names are 'random', 'er', and 'ws'.")


def compute_uniqueness(net):
    """ This function computes the percentage of unique structures in a network (with one layers).
    It extracts the neighborhood of every node and maps them to an isomorphism class,
    represented by complete graph invariant (equivalent to a canonical labeling).
    It then stores the number of neighborhoods for each isomorphism class
    (in a dictionary with the complete invariant as a key), and return the number of classes occurring just one time

    Parameters
    ----------
    net : Multilayer network (also single-layer networks are acceptable)
        the network

    Returns
    --------
    float
        the percentage of unique neighborhoods in the graph (e.g., 1.00 if all the neighborhoods
        have unique structures, 0.00 if there no unique structures)
    """
    dic_layer_1_neigh = {}
    for n in list(net):
        dic_layer_1_neigh[n] = []

    for e in list(net.edges):
        if e[0] != e[1]:
            dic_layer_1_neigh[e[0]].append(e[1])
            dic_layer_1_neigh[e[1]].append(e[0])

    dic_count_n = {}
    for k in dic_layer_1_neigh.keys():
        neigh_net = pymnet.MultilayerNetwork(aspects=0)
        for neigh in dic_layer_1_neigh[k]:
            for sec_neigh in dic_layer_1_neigh[neigh]:
                if sec_neigh in dic_layer_1_neigh[k]:
                    neigh_net[neigh, sec_neigh] = 1

        compl_inv_n = str(pymnet.get_complete_invariant(neigh_net))
        try:
            dic_count_n[compl_inv_n] += 1
        except KeyError as e:
            dic_count_n[compl_inv_n] = 1

    count_n = 0
    total_neighborhoods = float(len(list(net)))
    if total_neighborhoods > 0:
        for k in dic_count_n.keys():
            if dic_count_n[k] == 1:
                count_n += 1
        return float(count_n) / total_neighborhoods
    else:
        return 0.0


def computeUniquenessInNetwork(size, name_model, deg, par=None):
    """ Function that generates a network and return the percentage of unique neighborhoods in it.

    See also:
    -------
    generateNetwork : function to generate a random network from a given network model
    compute_uniqueness : function to compute the percentage of unique structures in a network
    """
    net = generateNetwork(size, name_model, deg, par)
    return compute_uniqueness(net)


def containedInInterval(value, low, up, tolerance=0):
    """ This function returns True if the given value (plus or minus a tolerance) is contained in the interval
    delimited by low and up, and False otherwise.

    Parameters
    ----------
    value : int or float
        value to evaluate
    low: int or float
        lower extreme of the interval
    up : int or float
        upper extreme of the interval
    tolerance: int or float
        the tolerance level
    """
    if tolerance == 0:
        return (True if (value >= low and value <= up) else False)
    else:
        return containedInInterval(value, low, up) or containedInInterval(value - tolerance, low,
                                                                          up) or containedInInterval(value + tolerance,
                                                                                                     low, up)


def binarySearchUnique(lowervalue, uppervalue, uniqval, size, name_model, model_par_prob=None, tolerance_threshold=0.05,
                       n_decisions=0, z_value=2.58, single_sim_number=10, max_sim=50):
    def evaluate(deg, sim_number=single_sim_number, simulation_list=[]):
        mean, mean_conf_lower, mean_conf_upper = doSimulations(deg, sim_number=sim_number,
                                                               simulation_list=simulation_list)
        if (containedInInterval(uniqval, mean_conf_lower, mean_conf_upper, tolerance=tolerance_threshold) and
            not containedInInterval(mean, uniqval - tolerance_threshold, uniqval + tolerance_threshold)) and \
                len(simulation_list) < max_sim:
            return evaluate(deg, sim_number=3, simulation_list=simulation_list)
        else:
            return mean, mean_conf_lower, mean_conf_upper, (len(simulation_list) >= max_sim)

    def doSimulations(deg, sim_number=single_sim_number, simulation_list=[]):
        for i in range(0, sim_number):
            count_n = computeUniquenessInNetwork(size, name_model, deg, model_par_prob)
            simulation_list.append(count_n)
        mean, std_dev = np.mean(simulation_list), np.std(simulation_list)
        s_error = std_dev / np.sqrt(sim_number)
        return mean, mean - std_dev * z_value, mean + std_dev * z_value

    def endEvaluation(mean_conf_lower, mean_conf_upper, middlevalue, n_decisions, lowervalue, uppervalue):
        if containedInInterval(uniqval, mean_conf_lower, mean_conf_upper, tolerance=tolerance_threshold):
            return middlevalue, n_decisions + 1, lowervalue, uppervalue
        else:
            print("Method failed")
            return -1, -1, -1, -1

    if lowervalue < 0 or uppervalue < 0 or uppervalue < lowervalue or uppervalue > (size - 1):
        print("Error: extreme values not valid")
        return -1, -1, -1, -1
    elif uniqval <= 0 or uniqval >= 1:
        print("Targeted uniqueness value not valid: give a target value greater than 0 and lower than 1")
        return -1, -1, -1, -1

    middlevalue = float(uppervalue + lowervalue) / 2.0

    if (uppervalue - lowervalue) < 0.02:
        mean, mean_conf_lower, mean_conf_upper = doSimulations(middlevalue,
                                                               sim_number=single_sim_number)
        return endEvaluation(mean_conf_lower, mean_conf_upper, middlevalue, n_decisions, lowervalue, uppervalue)

    if n_decisions == 0:
        for val in [lowervalue, uppervalue]:
            mean_val, mean_conf_lower, mean_conf_upper, reachedLimit = evaluate(lowervalue, simulation_list=[])
            if containedInInterval(mean_val, uniqval - tolerance_threshold, uniqval + tolerance_threshold):
                return val, n_decisions + 1, lowervalue, uppervalue
            if reachedLimit:
                return endEvaluation(mean_conf_lower, mean_conf_upper, val, n_decisions, lowervalue, uppervalue)

    meanvalue, mean_conf_lower, mean_conf_upper, reachedLimit = evaluate(middlevalue, simulation_list=[])

    if containedInInterval(meanvalue, uniqval - tolerance_threshold, uniqval + tolerance_threshold):
        return middlevalue, n_decisions + 1, lowervalue, uppervalue
    elif reachedLimit:
        return endEvaluation(mean_conf_lower, mean_conf_upper, middlevalue, n_decisions, lowervalue, uppervalue)
    else:
        low_extreme, up_extreme = (lowervalue, middlevalue) if (meanvalue > uniqval) else (middlevalue, uppervalue)
        return binarySearchUnique(low_extreme, up_extreme, uniqval, size, name_model, model_par_prob=model_par_prob,
                                  n_decisions=n_decisions + 1, z_value=z_value)


if __name__ == "__main__":
    target_uniqueness_value = 0.5
    net_size = 1000
    deg_low_value, deg_up_value = 2.0, 90.0
    name_model = "er"
    confidence_level, z_value = 0.99, 2.58
    deg, n_decisions, lowervalue, uppervalue = binarySearchUnique(deg_low_value, deg_up_value, target_uniqueness_value,
                                                                  net_size, name_model, model_par_prob=[directory_path],
                                                                  z_value=z_value)
    print("The average degree value that gives an " + str(name_model) + " network with " + str(
        net_size) + " nodes and with " + str(target_uniqueness_value) + " uniqueness is: " + str(deg))
    print("Probability of correct evaluation: " + str(confidence_level ** n_decisions))
    print("Extremes of the final interval: ", lowervalue, uppervalue)
