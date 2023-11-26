import numpy as np
import pandas as pd
import pymnet

def convertToNet(path, sep=','):
    # Function that converts a csv file into a pymnet network
    # path = input("Enter the path to the csv file: ")
    # sep = input("Enter the separator of the csv file: ")
    df = pd.read_csv(path, sep=sep)
    net = pymnet.MultilayerNetwork(aspects=0)
    for i in range(0, len(df)):
        net[df.iloc[i ,0], df.iloc[i ,1]] = 1
    return net


# We won't use this function, but it is useful to generate a random network with a given number of nodes and a given average degree.
def generateNetwork(size, name_model, avg_degree, model_par_prob=None):
    edges = int((avg_degree *size ) /2.000)  # compute the needed number of edges
    if name_model == 'er':
        return pymnet.models.er(size, edges=[edges])  # a pymnet function to generate a ER network
    if name_model == 'wdef binarySearchUnique(lowervalue, uppervalue, uniqval, size, name_model, model_par=None, tolerance_threshold=0.05, n_decisions=0, z_value=2.58, single_sim_number=10, max_sim=50):def binarySearchUnique(lowervalue, uppervalue, uniqval, size, name_model, model_par=None, tolerance_threshold=0.05, n_decisions=0, z_value=2.58, single_sim_number=10, max_sim=50):def binarySearchUnique(lowervalue, uppervalue, uniqval, size, name_model, model_par=None, tolerance_threshold=0.05, n_decisions=0, z_value=2.58, single_sim_number=10, max_sim=50):s':
        return pymnet.models.ws(size, [edges], p=model_par_prob[0])  # a pymnet function to generate a WS network

def compute_uniqueness(net):
    dic_layer_1_neigh = {}  # dictionary to store the list of neighbors for every node
    for n in list(net):  # list of nodes
        dic_layer_1_neigh[n] = []

    # store the list of neighbors of every node
    for e in list(net.edges):
        if e[0] != e[1]:
            dic_layer_1_neigh[e[0]].append(e[1])
            dic_layer_1_neigh[e[1]].append(e[0])

    dic_count_n = {}  # dictionary to store the number of occurences for each isomorphism class
    for k in dic_layer_1_neigh.keys():  # go through all nodes
        neigh_net = pymnet.MultilayerNetwork \
            (aspects=0)  # create a temporary network to store the neighborhood of a node
        for neigh in dic_layer_1_neigh[k]:  # go through the neighbors
            for sec_neigh in dic_layer_1_neigh[neigh]:  # go trough the neighbors of the neighbor
                if sec_neigh in dic_layer_1_neigh[k]:  # if the node
                    neigh_net[neigh, sec_neigh] = 1  # this adds an edge between the two nodes

        compl_inv_n = str(pymnet.get_complete_invariant(neigh_net))  # compute the complete invariant
        # increment the count of isomorphism classes
        try:
            dic_count_n[compl_inv_n] += 1
        except KeyError as e:
            dic_count_n[compl_inv_n] = 1

    # count the number of classes occurring one single time
    count_n = 0
    for k in dic_count_n.keys():
        if dic_count_n[k] == 1:
            count_n += 1
    return float(count_n) / float(len(list
        (net)))  # dividing the number of unique neighborhoods by the number of nodes in the network (such as the total number of neighborhoods)


# def computeUniquenessInNetwork(size, name_model, deg, par=None):
def computeUniquenessInNetwork(net):
    # Function that generates a network and return the percentage of unique neighborhoods in it.
    # net = generateNetwork(size, name_model, deg, par)
    net = net
    return compute_uniqueness(net)

def containedInInterval(value, low, up, tolerance=0):
    # returns True if the given value (plus or minus a tolerance) is contained in the interval delimited by low and up, and False otherwise.
    if tolerance == 0:
        return (True if (value >= low and value <= up) else False)
    else:
        return (containedInInterval(value, low, up) or containedInInterval(value -tolerance, low, up) or
                containedInInterval(value +tolerance, low, up))


# def binarySearchUnique(lowervalue, uppervalue, uniqval, size, name_model, model_par=None, tolerance_threshold=0.05, n_decisions=0, z_value=2.58, single_sim_number=10, max_sim=50):
def binarySearchUnique(lowervalue, uppervalue, uniqval, net, name_model ,tolerance_threshold=0.05, n_decisions=int(0), z_value=2.58, single_sim_number=int(10), max_sim=int(50)):

    size = len(list(net))  # number of nodes in the network

    # def doSimulations(deg, sim_number=single_sim_number, simulation_list=[]):
    def doSimulations(sim_number=int(single_sim_number), simulation_list=[]):
        for i in range(0, int(sim_number)):  # run the simulations
            # count_n = computeUniquenessInNetwork(size, name_model, deg, model_par) #compute the uniqueness value
            count_n = compute_uniqueness(net)
            simulation_list.append(count_n)  # add the computed value to the ones in the list
        mean, std_dev = np.mean(simulation_list), np.std(simulation_list)  # compute the mean and the standard deviation
        s_error = std_dev / np.sqrt(sim_number)  # compute the standard error
        return mean, mean - std_dev * z_value, mean + std_dev * z_value  # return the mean, the lower and upper values of the confidence interval at confidence level corresponding to z-value

    def evaluate(sim_number=single_sim_number, simulation_list=[]):
        mean, mean_conf_lower, mean_conf_upper = doSimulations(sim_number=sim_number, simulation_list=simulation_list)
        # if the target value is contained in the confidence interval (but the mean value it's not), then we need to do more simulations
        if (containedInInterval(uniqval, mean_conf_lower, mean_conf_upper, tolerance=tolerance_threshold) and \
            not containedInInterval(mean, uniqval -tolerance_threshold, uniqval +tolerance_threshold)) and \
                len(simulation_list) < max_sim:
            return evaluate(sim_number=3, simulation_list=simulation_list)
        else:  # return the mean and the computed confidence interval extremes, and a boolean value indicating whether the maximum limit of simulation has been reached
            return mean, mean_conf_lower, mean_conf_upper, (len(simulation_list) >= max_sim)

    def endEvaluation(mean_conf_lower, mean_conf_upper, middlevalue, n_decisions, lowervalue, uppervalue):
        # if the target value (plus or minus the tolerance) is contained in the interval, then return the middle
        # value, otherwise the method fails
        if containedInInterval(uniqval, mean_conf_lower, mean_conf_upper, tolerance=tolerance_threshold):
            return middlevalue, n_decisions +1, lowervalue, uppervalue
        else:
            print("Method failed")
            return -1, -1, -1, -1

    # check if the extremes of the interval and the targeted uniqueness value are valid
    if lowervalue < 0 or uppervalue < 0 or uppervalue < lowervalue or uppervalue > \
            (size - 1):  # the uppervalue cannot be greater than the maximum allowed degree
        print("Error: extreme values not valid")
        return -1, -1, -1, -1
    elif uniqval <= 0 or uniqval >= 1:
        print("Targeted uniqueness value not valid: give a target value greater than 0 and lower than 1")
        return -1, -1, -1, -1

    middlevalue = float(uppervalue + lowervalue) / 2.000  # compute the middle value (midpoint of the interval)
    # if the two extreme values are very close to each other, evaluate the middle value. If it is not acceptable, the method fails.
    if (uppervalue - lowervalue) < 0.02:
        mean, mean_conf_lower, mean_conf_upper = doSimulations(middlevalue,
                                                               sim_number=single_sim_number)  # simulate with the middle value
        return endEvaluation(mean_conf_lower, mean_conf_upper, middlevalue, n_decisions, lowervalue, uppervalue)

    if n_decisions == 0:  # at the beginning, we need to evaluate the extreme values of the interval to check if they are the values we are looking for
        for val in [lowervalue, uppervalue]:
            mean_val, mean_conf_lower, mean_conf_upper, reachedLimit = evaluate(lowervalue, simulation_list=[])
            if containedInInterval(mean_val, uniqval - tolerance_threshold, uniqval + tolerance_threshold):
                return val, n_decisions + 1, lowervalue, uppervalue
            if reachedLimit == True:
                return endEvaluation(mean_conf_lower, mean_conf_upper, val, n_decisions, lowervalue, uppervalue)

    meanvalue, mean_conf_lower, mean_conf_upper, reachedLimit = evaluate(middlevalue,
                                                                         simulation_list=[])  # evaluate the middle value
    # does the confidence interval contain the uniqueness value we are looking for? If yes, we are done. Otherwise, if the maximum limit
    # of simulation has not been reached in the previous evaluation, we need to make a decision where to move with the binary search
    if containedInInterval(meanvalue, uniqval - tolerance_threshold, uniqval + tolerance_threshold):
        return middlevalue, n_decisions + 1, lowervalue, uppervalue
    elif reachedLimit == True:
        return endEvaluation(mean_conf_lower, mean_conf_upper, middlevalue, n_decisions, lowervalue, uppervalue)
    else:  # make a decision about where to move with the binary search
        low_extreme, up_extreme = (lowervalue, middlevalue) if (meanvalue > uniqval) else (middlevalue, uppervalue)
        return binarySearchUnique(low_extreme, up_extreme, uniqval, net, n_decisions=n_decisions + 1, z_value=z_value)


path = "data/SNAP_facebook_clean_data/new_sites_edges.csv"
net = convertToNet(path)
target_uniqueness_value = 0.5  # float(input("Enter the target uniqueness value: "))
deg_low_value = 2  # input("Enter the lower value of the degree interval: ")
deg_up_value = 90  # input("Enter the upper value of the degree interval: ")
deg_low_value, deg_up_value = float(deg_low_value), float(deg_up_value)
name_model = "test"  # input("Enter the name of the network model: ")
confidence_level, z_value = 0.99, 2.58
tolerance_threshold = 0.05
max_sim = int(50)
single_sim_number = int(10)
deg, n_decisions, lowervalue, uppervalue = binarySearchUnique(deg_low_value, deg_up_value, target_uniqueness_value, net,
                                                              name_model, z_value=z_value)

print("The average degree value that gives a " + str(name_model) + " network with " + str(
    len(list(net))) + " nodes and with " + str(target_uniqueness_value) + " uniqueness is: " + str(deg))
print("Probability of correct evaluation: " + str(confidence_level ** n_decisions))
print("Extremes of the final interval: ", lowervalue, uppervalue)



