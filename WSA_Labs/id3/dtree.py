#!/usr/bin/env python
# -*- coding: utf-8 -*-

from anytree import Node, RenderTree

''' THIS IS DTREE IMPLEMENTATION '''

class Dnode:
    ''' Node Class '''
    def __init__(self,  data, parent=None, children=None):
        self.data = data
        self.parent = None
        self.parent = self.set_parent(parent)
        self.children = []
        self.children = self.set_children(children)
        self.level = 0

    def set_parent(self, new_parent):
        ''' sets parent of Dnode, parent should be Dnode '''
        if new_parent:
            if self.parent:
                print("Replace current parent? (Y/n): "),
                choice = str(raw_input())

                if choice.lower() == 'y' or choice == '':
                    self.parent = new_parent
                    self.parent.children.append(self)
                    # print("Set Parent Success.\n")
                else:
                    print("Parent not changed.\n")
            else:
                self.parent = new_parent
                self.parent.children.append(self)
                # print("Set Parent Success.\n")
        else:
            return None

    def get_parent(self):
        ''' return parent data of Dnode '''
        if self.parent == None:
            # print("No parent.\n")
            return
        else:
            return str(self.parent.data)

    def set_children(self, new_children):
        ''' set children of Dnode, must be list of Dnode '''
        if new_children:
            if self.children:
                print("Replace current children? (Y/n): "),
                choice = str(raw_input())

                if choice.lower() == 'y' or choice == '':
                    self.children = new_children
                    for child in new_children:
                        child.parent = self
                    # print("Set Children Success.\n")
                else:
                    print("Children not changed.\n")
            else:
                self.children = new_children
                for child in new_children:
                    child.parent = self
                # print("Set Children Success.\n")
                return new_children
        else:
            return []

    def append_child(self, new_child):
        ''' adds one Dnode child to the Dnode '''
        if new_child:
            self.children.append(new_child)
            new_child.parent = self
        else:
            print("Error in child.")

    def get_children(self):
        ''' returns list of child Dnode of Dnode '''
        if self.children == []:
            return None
        else:
            child = []
            for node in self.children:
                child.append(node.data)

            return self.children

    def get_children_data(self):
        ''' returns list of children Dnode data '''
        if self.children == []:
            return None
        else:
            child = []
            for node in self.children:
                child.append(node.data)

            return child

    def set_level(self):
        ''' set level of tree for Dnode (0 for root) '''
        level = 0
        cursor = self
        while(cursor.parent != None):
            level += 1
            cursor = cursor.parent
        self.level = level

    def get_level(self):
        ''' returns level of Dnode '''
        return self.level

    def __str__(self):
        ''' prints Dnode content '''
        has_parent = isinstance(self.get_parent(), str)
        has_children = isinstance(self.get_children(), list)

        res = "\nNode Content:\n data: " + str(self.data)
        res += "\n parent: " + str(self.get_parent()) if has_parent else "\n parent: X"
        res += "\n children: " + str(self.get_children_data()) if has_children else "\n children: X"
        res += "\n level: " + str(self.get_level())
        return res

    def __eq__(self, other):
        """ Compares two instances of this Class """
        if not isinstance(other, Dnode):
            # comparision against unrelated types
            return NotImplemented

        return self.data == other.data

class Dtree:
    ''' Implementation of DTree '''
    def __init__(self):
        self.root = None
        self.nodes = []
        self.count = int(0)

    def set_root(self, root_node):
        ''' sets root Dnode for Dtree, must be Dnode '''
        if root_node.parent != None:
            print("Root node cannot have parent.")
            print("Root node not set.")
            return
        else:
            self.root = root_node

    def get_root(self):
        ''' returns root Dnode of Dtree '''
        if self.root == None:
            print("Root not set.")
            return
        else:
            return self.root

    def add_node(self, new_node):
        ''' adds given Dnode to Dtree, must be Dnode '''
        if new_node in self.nodes:
            ''' print("Node already exist") '''
            return
        self.nodes.append(new_node)
        # print("Node with data " + str(new_node.data) + " added.\n")
        for node in self.nodes:
            node.set_level()
        return

    def remove_node(self, rem_node):
        ''' remove given Dnode from Dtree '''
        for node in self.nodes:
            if node == rem_node:
                self.nodes.remove(rem_node)
                return "Node removal success."
        return "Node not found in tree."

    def display(self):
        ''' displays content of Dtree, detailed view (data) '''
        print("\nDTREE CONTENT: ")
        if self.nodes == []:
            print("Empty")
        else:
            for node in self.nodes:
                print(node)
        print("DTREE CONTENT END\n")

    def show_tree(self):
        ''' display Dtree tree view (mine) '''
        if self.root == None:
            print("Root not set")
            return
        else:
            stack = []
            stack.append(self.root)
            while(stack != []):
                node = stack.pop()
                if node != self.root:
                    print("  " * (node.level - 1) + "\'-" + str(node.data))
                else:
                    print(str(node.data))
                node_children = node.children
                node_children.reverse()
                for child_node in node_children:
                    stack.append(child_node)

    def display_tree(self):
        ''' displays Dtree tree view using anytree (https://anytree.readthedocs.io/en/latest/index.html) '''
        if self.root == None:
            print("Root not set.")
            return

        node_dict = {}
        for node in self.nodes:
            node_dict[node.data] = Node(node.data)

        for node in self.nodes:
            if node.parent != None:
                node_dict[node.data].parent = node_dict[node.parent.data]

        root_node = node_dict[self.root.data]

        for pre, fill, node in RenderTree(root_node):
            print("%s%s" % (pre, node.name))


def main():
    ''' MAIN FUNCTION '''
    dtree = Dtree()

    ''' top down '''
    # node1 = Dnode(1)
    # node2 = Dnode(2, node1)
    # node3 = Dnode(3, node1)
    # node4 = Dnode(4, node2)
    # node5 = Dnode(5, node2)
    # node6 = Dnode(6, node3)
    # node7 = Dnode(7, node3)
    # node8 = Dnode(8, node3)
    # node9 = Dnode(9, node4)

    ''' bottom up '''
    node9 = Dnode(9)
    node8 = Dnode(8)
    node7 = Dnode(7)
    node6 = Dnode(6)
    node5 = Dnode(5)
    node4 = Dnode(4, children=[node9])
    node3 = Dnode(3, children=[node6, node7, node8])
    node2 = Dnode(2, children=[node4, node5])
    node1 = Dnode(1, children=[node2, node3])
    nodeA = Dnode(10)
    nodeB = Dnode(11)

    ''' adding nodes to dtree '''
    dtree.add_node(node1)
    dtree.add_node(node2)
    dtree.add_node(node3)
    dtree.add_node(node4)
    dtree.add_node(node5)
    dtree.add_node(node6)
    dtree.add_node(node7)
    dtree.add_node(node8)
    dtree.add_node(node9)
    dtree.add_node(nodeA)
    dtree.add_node(nodeB)

    node6.append_child(nodeA)
    node6.append_child(nodeB)

    dtree.display()

    dtree.set_root(node1)

    # dtree.show_tree()
    dtree.display_tree()

    print('\n\nEND\n\n')

if __name__ == '__main__':
    main()
