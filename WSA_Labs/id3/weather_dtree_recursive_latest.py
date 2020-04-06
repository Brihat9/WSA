#!/bin/usr/python

import copy
import math
import numpy as np
import pandas

from anytree import Node, RenderTree
from dtree import Dnode as D, Dtree as DT


''' global variables '''
HEADERS = []
CONST_HEADERS = []
NODE_DICT = {}          # dictionary maintained for Dnode nodes


class Dnode(D):
    ''' rewrote __eq__ method not to compare decision nodes  '''
    def __eq__(self, other):
        """ Compares two instances of this Class """
        if not isinstance(other, Dnode):
            # comparision against unrelated types
            return NotImplemented
        return False


class Dtree(DT):
    ''' rewrote display_tree method to incorporate
        multiple nodes with same data
    '''
    def append_child_nodes(self, dnode, node):
        ''' recursive method to append child for anytree decision tree '''
        for child in dnode.children:
            new_node = Node(child.data)
            new_node.parent = node
            if(child.children != []):
                self.append_child_nodes(child, new_node)
            else:
                # print("Leaf Node")
                return


    def display_tree(self):
        if self.root == None:
            print("Root not set.")
            return

        dnode_root = DTREE.get_root()
        root_node = Node(DTREE.get_root().data)

        ''' recursively append nodes to decision tree '''
        self.append_child_nodes(dnode_root, root_node)

        # displays anytree
        print("\nDecision Tree\n-------------\n")
        for pre, fill, node in RenderTree(root_node):
            print("%s%s" % (pre, node.name))


''' more global variables '''
DTREE = Dtree()         # init decision tree
# DECISION_NODES = { 'yes': Dnode('yes'), 'no': Dnode('no') }


def get_attribute_count_dict(df, num_records):
    ''' returns decision dict for given dataframe 'df'
        helper method for select_an_attribute method
    '''
    attribute = [None] * len(HEADERS)
    unique_values = [None] * len(HEADERS)

    # separate decision column from dataframe
    decision = list(df[df.columns[-1]])

    for index in range(len(HEADERS)):
        attribute[index] = list(df[HEADERS[index]])

    dict_decision_1 = [None] * len(HEADERS)
    dict_decision_2 = [None] * len(HEADERS)

    for index in range(len(HEADERS)):
        dict_decision_1[index] = {}
        dict_decision_2[index] = {}
        for index2 in range(num_records):
            try:
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
            except:
                pass

    # print(dict_decision_1)
    # print(dict_decision_2)
    return dict_decision_1, dict_decision_2


def calculate_gains(num_records, decision, dict_decision_1, dict_decision_2):
    ''' returns gain values for the attributes
        helper method for select_an_attribute method
    '''

    attribute_entropies = [None] * len(HEADERS)
    attribute_count = [None] * len(HEADERS)

    # calculates attribute entropies and count
    for index in range(len(HEADERS)):
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
    i_attr = [None] * len(HEADERS)
    for index in range(len(HEADERS)):
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


def select_an_attribute(df, num_records):
    ''' returns one attribute with highest gain with its unique values '''

    attribute = [None] * len(HEADERS)
    unique_values = [None] * len(HEADERS)

    decision = list(df[df.columns[-1]])
    unique_decisions = set(decision)

    # best case: if dataframe contains only one decision, return that decision
    if len(unique_decisions) == 1:
        return list(unique_decisions)[0], -1

    for index in range(len(HEADERS)):
        attribute[index] = list(df[HEADERS[index]])
        unique_values[index] = list(set(attribute[index]))

    # calling helper functions
    dict_decision_1, dict_decision_2 = get_attribute_count_dict(df, num_records)
    gain_values = calculate_gains(num_records, decision, dict_decision_1, dict_decision_2)

    # if gain_values == [] or not isinstance(gain_values, list):
    #     return 'no', -1

    # select attribute with maximum gain
    try:
        selected_index = gain_values.index(max(gain_values))
        selected_attribute = HEADERS[selected_index]

        selected_attribute_values = unique_values[selected_index]
        # selected_attribute_values_num = len(selected_attribute_values)
    except:
        print("Unexpected Error Occured")
        import pdb; pdb.set_trace()

    return selected_attribute, selected_attribute_values


def select_root_attribute(df, num_records):
    ''' returns root attribute for decision tree and
        corresponding attribute unique values
    '''
    global DTREE
    global HEADERS
    global NODE_DICT

    selected_root_attribute, selected_root_attribute_values = select_an_attribute(df, num_records)

    # creating dtree root from selected root attribute
    dtree_root = Dnode(selected_root_attribute)

    # update NODE_DICT
    NODE_DICT[selected_root_attribute] = dtree_root

    # add dtree root to DTREE and set it root of DTREE
    DTREE.add_node(dtree_root)
    DTREE.set_root(dtree_root)

    # remove selected attribute from headets
    HEADERS.remove(selected_root_attribute)

    # can remove this ...
    # parent_of_parent = dtree_root

    return selected_root_attribute, selected_root_attribute_values


def make_decision_tree(old_df, old_attribute, old_attribute_values, old_parent_node):
    ''' recursive method to update decision tree for ID3 '''
    global DTREE
    global HEADERS
    global CONST_HEADERS
    global NODE_DICT

    # import pdb; pdb.set_trace()
    for old_attribute_value in old_attribute_values:
        old_attribute_value_node = Dnode(old_attribute_value)
        NODE_DICT[old_attribute_value] = old_attribute_value_node
        DTREE.add_node(old_attribute_value_node)
        old_parent_node.append_child(old_attribute_value_node)

    # for old_attribute_value in selected_old_attribute_values:
        new_parent_node = NODE_DICT[old_attribute_value]

        DTREE.add_node(new_parent_node)

        ''' new dataframe for third selected attribute value and removed attribute '''
        new_df = copy.deepcopy(old_df)
        new_df = old_df.loc[old_df[old_attribute] == old_attribute_value]
        new_df = new_df.drop(columns=old_attribute)

        num_records = len(old_df.index)

        # considering all remaining columns as headers
        HEADERS = new_df.columns.values.tolist()[1:-1]
        selected_new_attribute, selected_new_attribute_values = select_an_attribute(new_df, num_records)

        try:
            ''' try to remove third attribute (if exist) '''
            HEADERS.remove(selected_new_attribute)
        except:
            ''' print("leaf detected") '''
            pass

        if selected_new_attribute_values != -1:
            new_parent_node = Dnode(selected_new_attribute)
            NODE_DICT[selected_new_attribute] = new_parent_node

            new_parent_node.set_parent(NODE_DICT[old_attribute_value])
            DTREE.add_node(new_parent_node)

            ''' recursive function call here '''
            make_decision_tree(new_df, selected_new_attribute, selected_new_attribute_values, new_parent_node)

        else:
            ''' if old_attribute is decision (leaf) '''
            # decision_node = copy.deepcopy(DECISION_NODES[selected_new_attribute])
            decision_node = Dnode(selected_new_attribute)
            decision_node.set_parent(NODE_DICT[old_attribute_value])
            DTREE.add_node(decision_node)
            # DTREE.display_tree()
    # DTREE.display_tree()


def predict_decision(input_values):
    ''' predict decision for given input_values (uses DTREE) '''

    attribute_node = DTREE.get_root()
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
    global HEADERS
    global CONST_HEADERS

    print("Brihat Ratna Bajracharya\n19/075\nCDCSIT\n")

    ''' read weather dataset '''
    try:
        df = pandas.read_csv('weather.csv')
    except:
        print("CSV file not found.\nExiting ...")
        return

    num_records = len(df.index)
    HEADERS = list(df.head())[1:-1]
    CONST_HEADERS = copy.deepcopy(HEADERS)

    print("Weather Dataset\n---------------\n")
    print(df)
    print("")

    ''' selecting first attribute (root attribute) '''
    selected_root_attribute, selected_root_attribute_values = select_root_attribute(df, num_records)
    root_parent_node = NODE_DICT[selected_root_attribute]

    ''' recursive function call here '''
    make_decision_tree(df, selected_root_attribute, selected_root_attribute_values, root_parent_node)

    ''' show final decision tree '''
    # DTREE.show_tree()
    DTREE.display_tree()

    ''' Testing '''
    input_test = {'outlook':'sunny', 'temperature':'mild', 'humidity':'high', 'windy':'true'}
    result = predict_decision(copy.deepcopy(input_test))

    print("\nFor input value\n")
    print(input_test)
    print("\nResult: "),
    print(result)

    print("\nDONE.")


if __name__ == "__main__":
    main()
