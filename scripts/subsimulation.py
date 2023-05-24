import networkx as nx
import random
import logging, sys
import warnings
warnings.filterwarnings('ignore')
import operator


def propagation(H, K, exposed, trust, time_prop):  # The sum of trust will happen with the same node at each time step
    forgetting = 1.3
    threshold = 30  # should be user defined with and with default value added
    discount = 0.5
    if trust == False:
        append = exposed.append
        for i in H.nodes(data=True):
            node = i[0]
            if i[1]["state"] == 0:
                neighbors_nodes = list(K.neighbors(node))
                product_prob = 1
                for j in neighbors_nodes:
                    if K.nodes[j]['state'] == 1:
                        val = K.get_edge_data(node, j, 'zero_occur_prob')
                        prob_zero_occ = val['zero_occur_prob']
                        product_prob *= prob_zero_occ
                prob_infection = 1 - product_prob
                # Graph H is changed
                if prob_infection >= random.uniform(0, 1):
                    i[1]['state'] = 1
                else:
                    i[1]['state'] = 0

                if i[1]['state'] == 1:
                    append(node)
        return exposed
    else:
        append = exposed.append  # Append serves as the list such that I can call append() with the item I want to add
        for i in H.nodes(data=True):
            new_trust = 0
            node = i[0]
            if i[1]["state"] == 0:  # Check if the node i has no information
                neighbors_nodes = list(K.neighbors(node))
                for j in neighbors_nodes:
                    if K.nodes[j]['state'] == 1:
                        val = K.get_edge_data(node, j, 'zero_occur_prob')
                        prob_zero_occ = val['zero_occur_prob']
                        prob_infection = 1 - prob_zero_occ
                        if prob_infection >= random.uniform(0, 1):
                            # check if this node has been visited by the same value.
                            if j not in K.nodes[node]['number_of_meets']:

                                K.nodes[node]['number_of_meets'][j] = [0]  # number of times i have met j
                                K.nodes[node]['time_step'][j] = [0]  # the list of time steps of the meeting

                                trust_value = K.edges[node, j]['trust']  # trust between i and j

                                K.nodes[node]['number_of_meets'][j].append(K.nodes[node]['number_of_meets'][j][-1] + 1)
                                meeting_number = K.nodes[node]['number_of_meets'][j][-1]  # taking the last meet number
                                time_steps_past = 0
                                trust_in_step = 0  # use variable to hold the sum of trust using formula

                                for meeting in range(0, meeting_number):
                                    trust_in_step += (discount ** meeting) * (forgetting ** (time_prop - time_steps_past))
                                new_trust += (trust_in_step * trust_value)

                            else:
                                # print("THIS IS NOT FIRST MEETING")
                                # recalculate trust_values

                                K.nodes[node]['number_of_meets'][j].append(K.nodes[node]['number_of_meets'][j][-1] + 1)
                                K.nodes[node]['time_step'][j].append(time_prop)
                                trust_value = K.edges[node, j]['trust']
                                meeting_number = K.nodes[node]['number_of_meets'][j][-1]
                                trust_in_step = 0
                                for meeting in range(0, meeting_number):
                                    list_meets = K.nodes[node]['number_of_meets'][j]
                                    list_meets_index = list_meets.index(meeting)
                                    time_steps_past = K.nodes[node]['time_step'][j][list_meets_index]
                                    trust_in_step += (discount ** meeting) * (forgetting ** (time_prop - time_steps_past))

                                new_trust += (trust_in_step * trust_value)


                                # calculate the trust values of all these nodes
                            sum_trusts = new_trust
                            if sum_trusts >= threshold:
                                i[1]['state'] = 1
                            else:
                                i[1]['state'] = 0
                            if i[1]['state'] == 1:
                                append(node)
                            return exposed

def sub_simulation(G, total_time, pop_percent, trust):
    K = G.copy()
    time_prop = 0
    exposed = []
    append = total_time.append
    while len(exposed) < pop_percent:
        H = K.copy()
        exposed = propagation(H, K, exposed, trust, time_prop)
        K = H.copy()  # This copy is done so that graph K is updated with the changes in graph H
        time_prop += 1
    append(time_prop)
    return total_time

