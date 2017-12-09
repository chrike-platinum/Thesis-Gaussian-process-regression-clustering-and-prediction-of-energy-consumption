__author__ = 'christiaanleysen'
'''
Class which represents a tree structure: Used for the visuealisation of the clusterresults
'''
class Node:
    value = ""
    leftChild = None
    rightChild = None
    def __init__(self, parent):
        self.parent = parent
    def isLeaf(self):
        if (self.leftChild == None and self.rightChild == None):
            return True
        else:
            return False

    def __repr__(self, level=0):
        ret = "\t"*level+repr(self.value)+"\n"
        if(self.leftChild != None):
            ret += self.leftChild.__repr__(level+1)
        if(self.rightChild != None):
            ret += self.rightChild.__repr__(level+1)
        return ret

def convertTreeAux(node, newick):
    if node.isLeaf():
        printValue = str(node.value).replace(',','|')
        newick.append(printValue)
    if node.leftChild != None:
        newick.append("(")
        convertTreeAux(node.leftChild, newick)
        newick.append(",")
    if node.rightChild != None:
        convertTreeAux(node.rightChild, newick)
        newick.append(")")

def insertBF(node, item):

        if node.value == item:
            return #we do nothing because the item is already here
        else:

            if set(item).issubset(set(node.value)):

                if node.leftChild == None:
                     node.leftChild = Node(node)
                     node.leftChild.value = item
                     #self.left.parent = self

                elif node.rightChild == None:
                    node.rightChild = Node(node)
                    node.rightChild.value = item
                    #self.right.parent = self


                elif(set(item).issubset(set(node.leftChild.value))):
                        insertBF(node.leftChild,item)
                else:
                    insertBF(node.rightChild,item)
            else:
                print("Stephanie error, should never come here")




