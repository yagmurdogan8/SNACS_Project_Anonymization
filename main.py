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
            next(file)  # Skip the header line if there is any
            for line in file:
                try:
                    edge = tuple(map(int, line.strip().split()))
                    edges.append(edge)
                except ValueError:
                    print(f"Skipping line with non-integer values: {line.strip()}")

    RDatasetGraph = nx.Graph(edges)
    return RDatasetGraph


# def generateNetwork(size, name_model, avg_degree, model_par_prob=None):
#     """ This function generates a random network with a given size and average dedgree
#     according to the specified network model.
#
#     Parameters
#     ----------
#     size : int
#         size of the network
#     name_model : str
#         name of the network model
#     avg_degree : int or float
#         average degree of the network
#     model_par_prob : list
#         list with the parameters of the model (e.g. probability of having an edge for Erdos-Renyi (ER) model,
#         or probability of rewiring for Watts-Strogatz (WS) model)
#
#     Returns
#     ------
#     the generated graph
#     """
#     # edges = int((avg_degree * size) / 2.000)  # compute the needed number of edges
#     # if name_model == 'er':
#     #     return pymnet.models.er(size, edges=[edges])  # a pymnet function to generate a ER network
#     # if name_model == 'ws':
#     #     return pymnet.models.ws(size, [edges], p=model_par_prob[0])  # a pymnet function to generate a WS network
#     if name_model == 'random':
#         edges = int((avg_degree * size) / 2.000)
#         return pymnet.models.er(size, edges=[edges])
#     elif name_model in ['er', 'ws']:
#         if not model_par_prob:
#             raise ValueError("model_par_prob must be provided for 'er' or 'ws' models.")
#         file_path = model_par_prob[0]
#         return readNetworkFromRLD(file_path)
#     else:
#         raise ValueError("Invalid network model name. Supported names are 'random', 'er', and 'ws'.")
#
#
# def generateNetwork(size, name_model, avg_degree, model_par_prob=None):
#     if name_model == 'random':
#         edges = int((avg_degree * size) / 2.000)
#         return pymnet.models.er(size, edges=[edges])
#     elif name_model in ['er', 'ws']:
#         if not model_par_prob:
#             raise ValueError("model_par_prob must be provided for 'er' or 'ws' models.")
#         file_path = model_par_prob[0]
#         return readNetworkFromRLD(file_path)
#     else:
#         raise ValueError("Invalid network model name. Supported names are 'random', 'er', and 'ws'.")

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
    represented by complete graph invariant (equivalent to a canonical labelling).
    It then stores the number of neighborhoods for each isomorphism class
    (in a dictionary with the complete invariant as a key), and return the number of classes occcuring just one time

    Parameters
    ----------
    net : Multilayer network (also single-layer networks are acceptable)
        the network

    Returns -------- float the percentage of unique neighborhoods in the graph (e.g.: 1.00 if all the neighborhoods
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
    for k in dic_count_n.keys():
        if dic_count_n[k] == 1:
            count_n += 1
    return float(count_n) / float(len(list(net)))


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


# def binarySearchUnique(lowervalue, uppervalue, uniqval, size, name_model, model_par_prob=None, tolerance_threshold=0.05,
#                        n_decisions=0, z_value=2.58, single_sim_number=10, max_sim=50):
#     meanvalue, mean_conf_lower, mean_conf_upper, reachedLimit = evaluate(middlevalue,
#                                                                          simulation_list=[])  # evaluate the middle value
#     # ...
#     low_extreme, up_extreme = (lowervalue, middlevalue) if (meanvalue > uniqval) else (middlevalue, uppervalue)
#     return binarySearchUnique(low_extreme, up_extreme, uniqval, size, name_model, model_par=model_par,
#                               n_decisions=n_decisions + 1, z_value=z_value)
#
#     def doSimulations(deg, sim_number=single_sim_number, simulation_list=[]):
#         """ This function performs the simulation (i.e. generates a network with a given average degree) a specified
#         number of times and compute the number of unique neighborhoods for each of those. It returns the mean and the
#         confidence interval of the results. If previous simulation were carried out, a list with the corresponding
#         uniqueness results can be passed as a parameter
#         """
#         for i in range(0, sim_number):  # run the simulations
#             count_n = computeUniquenessInNetwork(size, name_model, deg, model_par)  # compute the uniqueness value
#             simulation_list.append(count_n)  # add the computed value to the ones in the list
#         mean, std_dev = np.mean(simulation_list), np.std(simulation_list)  # compute the mean and the standard deviation
#         s_error = std_dev / np.sqrt(sim_number)  # compute the standard error
#         return mean, mean - std_dev * z_value, mean + std_dev * z_value  # return the mean, the lower and upper
#         # values of the confidence interval at confidence level corresponding to z-value
#
#     def evaluate(deg, sim_number=single_sim_number, simulation_list=[]):
#         """ This function computes the uniqueness of a network with a given average degree and checks if more
#         simulation are needed or not, in order to be confident (at 99% confidence level) that the value of the
#         average degree is or is not the one we are looking for.
#         """
#         mean, mean_conf_lower, mean_conf_upper = doSimulations(deg, sim_number=sim_number,
#                                                                simulation_list=simulation_list)
#         # if the target value is contained in the confidence interval (but the mean value it's not), then we need to
#         # do more simulations
#         if (containedInInterval(uniqval, mean_conf_lower, mean_conf_upper, tolerance=tolerance_threshold) and \
#             not containedInInterval(mean, uniqval - tolerance_threshold, uniqval + tolerance_threshold)) and \
#                 len(simulation_list) < max_sim:
#             return evaluate(deg, sim_number=3, simulation_list=simulation_list)
#         else:  # return the mean and the computed confidence interval extremes, and a boolean value indicating
#             # whether the maximum limit of simulation has been reached
#             return mean, mean_conf_lower, mean_conf_upper, (len(simulation_list) >= max_sim)
#
#     def endEvaluation(mean_conf_lower, mean_conf_upper, middlevalue, n_decisions, lowervalue, uppervalue):
#         """ This function evaluates if a result reached at the end of the searching process is acceptable or not. The
#         function is called when the maximum number of simulations has been reached or when the two extremes values
#         are close to each other. If the uniqueness value (with the tolerance level) is within the confidence interval
#         of the so far computed mean, then the result is acceptable. Otherwise the method fails.
#         """
#         # if the target value (plus or minus the tolerance) is contained in the interval, then return the middle
#         # value, otherwise the method fails
#         if containedInInterval(uniqval, mean_conf_lower, mean_conf_upper, tolerance=tolerance_threshold):
#             return middlevalue, n_decisions + 1, lowervalue, uppervalue
#         else:
#             print
#             "Method failed"
#             return -1, -1, -1, -1
#
#     # check if the extremes of the interval and the targeted uniqueness value are valid
#     if lowervalue < 0 or uppervalue < 0 or uppervalue < lowervalue or uppervalue > (
#             size - 1):  # the uppervalue cannot be greater than the maximum allowed degree
#         print
#         "Error: extreme values not valid"
#         return -1, -1, -1, -1
#     elif uniqval <= 0 or uniqval >= 1:
#         print
#         "Targeted uniqueness value not valid: give a target value greater than 0 and lower than 1"
#         return -1, -1, -1, -1
#
#     middlevalue = float(uppervalue + lowervalue) / 2.000  # compute the middle value (midpoint of the interval)
#     # if the two extreme values are very close to each other, evaluate the middle value. If it is not acceptable,
#     # the method fails.
#     if (uppervalue - lowervalue) < 0.02:
#         mean, mean_conf_lower, mean_conf_upper = doSimulations(middlevalue,
#                                                                sim_number=single_sim_number)  # simulate with the
#         # middle value
#         return endEvaluation(mean_conf_lower, mean_conf_upper, middlevalue, n_decisions, lowervalue, uppervalue)
#
#     if n_decisions == 0:  # at the beginning, we need to evaluate the extreme values of the interval to check if they
#         # are the values we are looking for
#         for val in [lowervalue, uppervalue]:
#             mean_val, mean_conf_lower, mean_conf_upper, reachedLimit = evaluate(lowervalue, simulation_list=[])
#             if containedInInterval(mean_val, uniqval - tolerance_threshold, uniqval + tolerance_threshold):
#                 return val, n_decisions + 1, lowervalue, uppervalue
#             if reachedLimit == True:
#                 return endEvaluation(mean_conf_lower, mean_conf_upper, val, n_decisions, lowervalue, uppervalue)
#
#     meanvalue, mean_conf_lower, mean_conf_upper, reachedLimit = evaluate(middlevalue,
#                                                                          simulation_list=[])  # evaluate the middle value
#     # does the confidence interval contain the uniqueness value we are looking for? If yes, we are done. Otherwise,
#     # if the maximum limit of simulation has not been reached in the previous evaluation, we need to make a decision
#     # where to move with the binary search
#     if containedInInterval(meanvalue, uniqval - tolerance_threshold, uniqval + tolerance_threshold):
#         return middlevalue, n_decisions + 1, lowervalue, uppervalue
#     elif reachedLimit == True:
#         return endEvaluation(mean_conf_lower, mean_conf_upper, middlevalue, n_decisions, lowervalue, uppervalue)
#     else:  # make a decision about where to move with the binary search
#         low_extreme, up_extreme = (lowervalue, middlevalue) if (meanvalue > uniqval) else (middlevalue, uppervalue)
#         return binarySearchUnique(low_extreme, up_extreme, uniqval, size, name_model, model_par=model_par,
#                                   n_decisions=n_decisions + 1, z_value=z_value)
#

#
# def binarySearchUnique(lowervalue, uppervalue, uniqval, size, name_model, model_par=None, tolerance_threshold=0.05,
#                        n_decisions=0, z_value=2.58, single_sim_number=10, max_sim=50):
#     """ This function performs a binary search (continuos version), looking for an average degree value for a given
#     network model, with which the network would have a certain percentage of unique neighborhoods' structure (with a
#     given tolerance threshold). The delimiting average degree values are given. To estimate the uniqueness value in a
#     network model corresponding to a given average degree, the function performs simulations by generating random
#     graphs according to the specified model.
#
#     Parameters ---------- lowervalue : int or float lower value of average degree delimiting the interval in which
#     the binary search is performed uppervalue : int or float upper value of average degree delimiting the interval in
#     which the binary search is performed size : int network size (number of nodes in the graph) name_model : str name
#     of the graph model we want to evaluate model_par : list parameters of the graph model uniqval : float the target
#     uniqueness value tolerance_threshold: float the tolerance value for uniqueness (e.g. if we are looking for 0.5
#     uniqueness, 0.5-tolerance_threshold or 0.5+tolerance_threshold are also fine) n_decisions : int number of
#     decisions already taken in the previous calls of the algorithm z_value: float z-value corresponding to a certain
#     confidence level (the confidence level we want to make decisions with) single_sim_number : int number of initial
#     simulations (graphs generation) to perform for each degree value's evaluation max_sim : int maximum simulation
#     numbers for each degree value to evaluate
#
#     Returns
#     --------
#     deg: float
#         the average degree value corresponding to the target uniqueness value
#     n_decisions: int
#         number of decisions taken at the end of the binary search process
#     lowervalue: int or float
#         lower extreme of the last evaluated interval in the binary search process
#     uppervalue: int or float
#         upper extreme of the last evaluated interval in the binary search process
#
#     Notes ------ The method can fail if the given extremes value are not valid or if the degree value corresponding
#     to the targeted uniqueness is not found. In this case it returns -1, -1, -1, -1. The target uniqueness value
#     should be greater than 0 and lower than 1, since there is a wide region where the graph has no or all unique
#     neighborhoods, and a single average degree value cannot be determined.
#     """
#

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

# # main function example
# if __name__ == "__main__":
#     target_uniqueness_value = 0.5  # 0.5 corresponds to 50% neighborhoods uniqueness
#     net_size = 1000  # number of nodes in the network
#     deg_low_value, deg_up_value = 2.0, 90.0  # extremes of the interval (average degree values)
#     name_model = "er"  # Erdos-Renyi graph model (can also be "ws" for Watts-Strogatz graph)
#     # model_parameters = [0.3] #parameters of the graph model (not used for er graph)
#     confidence_level, z_value = 0.99, 2.58  # confidence level and corresponding z-value
#     deg, n_decisions, lowervalue, uppervalue = binarySearchUnique(deg_low_value, deg_up_value, target_uniqueness_value,
#                                                                   net_size, name_model, z_value=z_value)
#     print
#     "The average degree value that gives a " + str(name_model) + " network with " + str(
#         net_size) + " nodes and with " + str(target_uniqueness_value) + " uniqueness is: " + str(deg)
#     print
#     "Probability of correct evaluation: " + str(confidence_level ** n_decisions)
#     print
#     "Extremes of the final interval: ", lowervalue, uppervalue
