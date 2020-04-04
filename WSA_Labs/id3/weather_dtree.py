#!/bin/usr/python

import copy
import math
import numpy as np
import pandas

from anytree import Node, RenderTree
from dtree import Dnode as D, Dtree as DT


class Dnode(D):
    ''' rewrote __eq__ method not to compare decision nodes
        i.e. nodes with yes and no data
    '''
    def __eq__(self, other):
        """ Compares two instances of this Class """
        if not isinstance(other, Dnode):
            # comparision against unrelated types
            return NotImplemented

        if self.data == 'yes' or self.data == 'no':
            return False
        return self.data == other.data


class Dtree(DT):
    ''' rewrote display_tree method to incorporate
        multiple nodes with same data
    '''
    def display_tree(self):
        if self.root == None:
            print("Root not set.")
            return

        decision_nodes = { 'yes': Node('yes'), 'no': Node('no') }

        node_dict = {}

        # creates anytree nodes from Dnode
        for node in self.nodes:
            if node.data != 'yes' and node.data != 'no':
                node_dict[node.data] = Node(node.data)

        # update parent for anytree nodes from Dnode
        for node in self.nodes:
            if node.data != 'yes' and node.data != 'no':
                if node.parent != None:
                    node_dict[node.data].parent = node_dict[node.parent.data]
            else:
                new_decision_node = copy.deepcopy(decision_nodes[node.data])
                new_decision_node.parent = node_dict[node.parent.data]

        # setting anytree root node
        root_node = node_dict[self.root.data]

        # displays anytree
        print("\nDecision Tree\n-------------\n")
        for pre, fill, node in RenderTree(root_node):
            print("%s%s" % (pre, node.name))


DECISION_NODES = { 'yes': Dnode('yes'), 'no': Dnode('no') }

def get_attribute_count_dict(df, headers, num_records):
    ''' returns decision dict for given dataframe 'df'
        helper method for select_an_attribute method
    '''
    attribute = [None] * len(headers)
    unique_values = [None] * len(headers)

    # separate decision column from dataframe
    decision = list(df[df.columns[-1]])

    for index in range(len(headers)):
        attribute[index] = list(df[headers[index]])

    dict_decision_1 = [None] * len(headers)
    dict_decision_2 = [None] * len(headers)

    for index in range(len(headers)):
        dict_decision_1[index] = {}
        dict_decision_2[index] = {}
        for index2 in range(num_records):
            if decision[index2] == 'yes':
                try:
                    dict_decision_1[index][attribute[index][index2]] = int(dict_decision_1[index][attribute[index][index2]]) + 1
                except:
                    dict_decision_1[index][attribute[index][index2]] = 1
            elif decision[index2] == 'no':
                try:
                    dict_decision_2[index][attribute[index][index2]] = int(dict_decision_2[index][attribute[index][index2]]) + 1
                except:
                    dict_decision_2[index][attribute[index][index2]] = 1

    # print(dict_decision_1)
    # print(dict_decision_2)
    return dict_decision_1, dict_decision_2


def calculate_gains(headers, num_records, decision, dict_decision_1, dict_decision_2):
    ''' returns gain values for the attributes
        helper method for select_an_attribute method
    '''

    attribute_entropies = [None] * len(headers)
    attribute_count = [None] * len(headers)

    # calculates attribute entropies and count
    for index in range(len(headers)):
        attribute_entropies[index] = {}
        attribute_count[index] = {}
        for key in dict_decision_1[index].keys():
            try:
                decision_yes = dict_decision_1[index][key]
                # decision_yes = dict_decision_1[index]['overcast']
            except:
                decision_yes = 0

            try:
                decision_no = dict_decision_2[index][key]
                # decision_no = dict_decision_2[index]['overcast']
            except:
                decision_no = 0

            decision_total = decision_yes + decision_no

            decision_entropy = - ((decision_yes * 1.0) / decision_total * np.log2((decision_yes * 1.0)/decision_total) + (decision_no * 1.0)/decision_total * np.log2((decision_no * 1.0)/decision_total))
            if math.isnan(decision_entropy):
                decision_entropy = 0

            attribute_entropies[index][key] = decision_entropy
            attribute_count[index][key] = decision_total

    # calculates I_ATTR
    i_attr = [None] * len(headers)
    for index in range(len(headers)):
        i_attr_sum = 0
        for key in attribute_entropies[index].keys():
            res = attribute_entropies[index][key] * attribute_count[index][key]
            res = 0.0 if math.isnan(res) else res
            i_attr_sum += res
            i_attr[index] = i_attr_sum / num_records
    # print(i_attr)

    # calculates gain and returns gain list (if nan return 0, no gain)
    total_yes = decision.count('yes')
    total_no = decision.count('no')
    total = total_yes + total_no
    entropy_total = - ((total_yes * 1.0) / total * np.log2((total_yes * 1.0)/total) + (total_no * 1.0)/total * np.log2((total_no * 1.0)/total))
    gain_values = [entropy_total - x for x in i_attr]
    # print(entropy_total)
    # print(gain_values)
    return gain_values


def select_an_attribute(df, headers, num_records):
    ''' returns one attribute with highest gain with its unique values '''

    attribute = [None] * len(headers)
    unique_values = [None] * len(headers)

    decision = list(df[df.columns[-1]])
    unique_decisions = set(decision)

    # best case: if dataframe contains only one decision, return that decision
    if len(unique_decisions) == 1:
        return list(unique_decisions)[0], -1

    for index in range(len(headers)):
        attribute[index] = list(df[headers[index]])
        unique_values[index] = list(set(attribute[index]))

    # calling helper functions
    dict_decision_1, dict_decision_2 = get_attribute_count_dict(df, headers, num_records)
    gain_values = calculate_gains(headers, num_records, decision, dict_decision_1, dict_decision_2)

    # if gain_values == [] or not isinstance(gain_values, list):
    #     return None, -1

    # select attribute with maximum gain
    selected_index = gain_values.index(max(gain_values))
    selected_attribute = headers[selected_index]

    selected_attribute_values = unique_values[selected_index]
    # selected_attribute_values_num = len(selected_attribute_values)

    return selected_attribute, selected_attribute_values


def predict_decision(dtree, input_values):
    ''' predict decision for given input_values '''

    attribute_node = dtree.get_root()
    while(True):
        try:
            if attribute_node.data == 'yes' or attribute_node.data == 'no':
                return attribute_node.data
            attribute_value = input_values[attribute_node.data]
            # print(attribute_value)

            attribute_children = attribute_node.children
            if len(attribute_children) == 1:
                return attribute_children[0].data
            else:
                for node in attribute_children:
                    if node.data == attribute_value:
                        input_values.pop(attribute_node.data, "attribute not found")
                        attribute_node = node.children[0]
        except:
            break
    return "Cannot Decide"


def main():
    ''' MAIN FUNCTION (ID3 WEATHER DATA) '''

    print("Brihat Ratna Bajracharya\n19/075\nCDCSIT\n")

    ''' read weather dataset '''
    df = pandas.read_csv('weather.csv')
    num_records = len(df.index)
    headers = list(df.head())[1:-1]

    print("Weather Dataset\n---------------\n")
    print(df)
    print("")

    # dictionary maintained for Dnode nodes
    node_dict = {}

    # init decision tree
    dtree = Dtree()

    ''' selecting first attribute (root attribute) '''
    selected_root_attribute, selected_root_attribute_values = select_an_attribute(df, headers, num_records)

    # creating dtree root from selected root attribute
    dtree_root = Dnode(selected_root_attribute)

    # update node_dict
    node_dict[selected_root_attribute] = dtree_root

    # add dtree root to dtree and set it root of dtree
    dtree.add_node(dtree_root)
    dtree.set_root(dtree_root)

    # remove selected attribute from headets
    headers.remove(selected_root_attribute)

    for root_attribute_value in selected_root_attribute_values:
        # add each attribute to node dict as child of root
        root_child_node = Dnode(root_attribute_value)
        node_dict[root_attribute_value] = root_child_node

        dtree.add_node(root_child_node)
        dtree_root.append_child(root_child_node)

    ''' show decision tree '''
    # dtree.display_tree()

    ''' loop may be nested upto 3 times because there are four columns and one is selected root '''
    parent_of_parent = dtree_root

    # new_headers = copy.deepcopy(headers)
    # new_df = copy.deepcopy(df)

    for attribute_value in selected_root_attribute_values:
        if attribute_value in node_dict:
            ''' if attribute value is in node_dict '''
            parent_node = node_dict[attribute_value]
            dtree.add_node(parent_node)
        else:
            ''' if attribute value is not in node_dict '''
            parent_node = Dnode(attribute_value, parent_of_parent)
            dtree.add_node(parent_node)

        ''' new dataframe for selected attribute value and removed attribute '''
        df2 = df.loc[df[selected_root_attribute] == attribute_value]
        df2 = df2.drop(columns=selected_root_attribute)

        num_records = len(df2.index)

        selected_second_attribute, selected_second_attribute_values = select_an_attribute(df2, headers, num_records)
        if selected_second_attribute_values != -1:
            # remove second attribute from header
            headers.remove(selected_second_attribute)

            if selected_second_attribute in node_dict:
                ''' if second attribute is in node_dict '''
                second_parent_node = node_dict[selected_second_attribute]
            else:
                ''' if second attribute is not in node_dict '''
                second_parent_node = Dnode(selected_second_attribute)
                node_dict[selected_second_attribute] = second_parent_node

            second_parent_node.set_parent(parent_node)
            dtree.add_node(second_parent_node)
            # dtree.display_tree()

            parent_of_parent = node_dict[selected_second_attribute]

            # new_df_2 = copy.deepcopy(df2)
            # new_headers_2 = copy.deepcopy(headers)
            for second_attribute_value in selected_second_attribute_values:
                second_attribute_value_node = Dnode(second_attribute_value)
                node_dict[second_attribute_value] = second_attribute_value_node
                dtree.add_node(second_attribute_value_node)
                second_parent_node.append_child(second_attribute_value_node)

            # for second_attribute_value in selected_second_attribute_values:
                if second_attribute_value in node_dict:
                    ''' if second attribute value is in node_dict '''
                    third_parent_node = node_dict[second_attribute_value]
                else:
                    ''' if second attribute value is not in node_dict '''
                    third_parent_node = Dnode(attribute_value, parent_of_parent)

                dtree.add_node(third_parent_node)

                ''' new dataframe for second selected attribute value and removed attribute '''
                df3 = df2.loc[df2[selected_second_attribute] == second_attribute_value]
                df3 = df3.drop(columns=selected_second_attribute)

                num_records = len(df3.index)

                selected_third_attribute, selected_third_attribute_values = select_an_attribute(df3, headers, num_records)

                try:
                    ''' try to remove third attribute (if exist) '''
                    headers.remove(selected_third_attribute)
                except:
                    ''' print("leaf detected") '''
                    pass

                if selected_third_attribute_values != -1:
                    if selected_third_attribute in node_dict:
                        ''' if third attribute is in node_dict '''
                        third_parent_node = node_dict[selected_third_attribute]
                    else:
                        ''' if third attribute is not in node_dict '''
                        third_parent_node = Dnode(selected_third_attribute)
                        node_dict[selected_third_attribute] = third_parent_node

                    third_parent_node.set_parent(node_dict[second_attribute_value])
                    dtree.add_node(third_parent_node)
                    # dtree.display_tree()

                else:
                    ''' if selected_third_attribute is decision (leaf) '''
                    decision_node = copy.deepcopy(DECISION_NODES[selected_third_attribute])
                    decision_node.set_parent(node_dict[second_attribute_value])
                    dtree.add_node(decision_node)
                    # dtree.display_tree()

                # df2 = copy.deepcopy(new_df_2)
                # headers = copy.deepcopy(new_headers_2)

        else:
            ''' if selected_second_attribute is decision (leaf) '''
            decision_node = copy.deepcopy(DECISION_NODES[selected_second_attribute])
            decision_node.set_parent(node_dict[attribute_value])
            dtree.add_node(decision_node)
            # dtree.display_tree()

        # headers = copy.deepcopy(new_headers)
        # df = copy.deepcopy(new_df)

    ''' show final decision tree '''
    # dtree.show_tree()
    dtree.display_tree()


    ''' Testing '''
    input_test = {'outlook':'sunny', 'temperature':'mild', 'humidity':'high', 'windy':'true'}
    result = predict_decision(dtree, copy.deepcopy(input_test))

    print("\nFor input value\n")
    print(input_test)
    print("\nResult: "),
    print(result)

    print("\nDONE.")


if __name__ == "__main__":
    main()
