import numpy as np
import random
import sys
import warnings



# For dealing with warnings
warnings.filterwarnings('error')

# In case we want to print the whole array
np.set_printoptions(threshold=sys.maxsize)


class CTW:
    def __init__(self, symbols, seq, depth = None, beta = None):
        
        # number of predicted symbols (for keeping counts)
        self.M = symbols 
        
        # Stores the sequence in order to compute 
        # the count vector for each context
        self.seq = seq
        
        n = self.seq.size
        
        # Stores the tree
        self.tree = {}
        
        # tree depth
        if depth == None:
            
            self.D = int(np.floor(np.log(n)))
            
        else: 
            
            self.D = depth
        
        
        if beta == None:
            
            self.beta = 1 - 2**(-symbols + 1)
        
        else:
            
            self.beta = beta
            
       
        # Putting the leaves in a matrix
        
        # Creates a matrix with the amount of rows equal to the maximum
        # number of non repeated leaves in the sequence
        leaves_matrix = -1 * np.ones((n - self.D + 1, self.D)).astype(int)
        
        # Number of different rows
        count = 0
        
        for i in range(n - self.D + 1):
            
            leaf_i = np.array([self.seq[i: i + self.D]])
            
            is_in_leaves = False
            
            for j in range(count):
                
                if (leaves_matrix[j] == leaf_i).all():
                    
                    is_in_leaves = True
                    
                    break
            
            if not is_in_leaves:
            
                leaves_matrix[count] = leaf_i
                
                count += 1
                
                
        # Gets only the rows different from -1s
        leaves_matrix = leaves_matrix[:count]
        
        
        #print(leaves_matrix)
        
        self.leaves_matrix = leaves_matrix
        
        #print(self.leaves_matrix)
        
        # Includes the root node
        self.root = Node(self, np.array([]).astype(int))
        
        self.add_node(self.root)
        
        # Includes the children of the root node
        for i in range(self.M):
            
            self.add_node(Node(self, np.array([i]).astype(int)))
        
        #Creates the maximal tree.
        leaves_number = len(self.leaves_matrix)
        
        for i in range(leaves_number):
            # Start by the end of the leaf because that's how
            # we're reading the contexts
            for j in range(-1, -1 - self.D, -1):
                
                current_context = self.leaves_matrix[i, j:]
                
                self.aux_tree(current_context.astype(int), j)
                                        
        
        #print(self.tree)
        
        # Get the children of all nodes so we can use them to 
        # compute the weighted probability. The parents will be
        # used in the MCMC algorithm
        self.root.get_children_parent()
        
        
        
        # List of leaves
        self.list_leaves = []
        
        for key in self.tree:
            
            if self.tree[key].children == {}:
                
                self.list_leaves.append(key)
        
        
        # In order to compute the mixture probability using sequential updates
        # we first need to compute de count vectors and the estimated probabilities
        # of the contexts in the first self.D elements of the sequence the "ordinary way"
        
        # Nodes that are going to have their count vector modified
        # count_mod = []
        
        # Computes the count vectors only for the first self.D observations of the sequence
        # because we'll use sequential updates to count the rest
        for i in range(self.D + 1):
            for j in range(self.D - i):
                
                current_context = str(self.seq[j : i + j].astype(int))
                
                if current_context in self.tree:
                
                    node = self.tree[current_context]
            
                    node.counts[self.seq[i + j]] += 1
                
                    node.log_P_hat = node.estimated_prob()
        
        
        self.seq_update()
        
        self.root.weighted_prob(self.beta)
        
        
        
    # Adds a node to the tree
    def add_node(self, node):
        
        self.tree[str(node.context)] = node
    
    
    def aux_tree(self, current_context, j):
        
        if str(current_context) not in self.tree:
                    
            self.add_node(Node(self, current_context))
            
        # Check if the node context isn't in the maximal depth and if
        # its children are not in the tree, so we can add them
        if j > -self.D and str(np.insert(current_context, 0, 0)) not in self.tree:
                        
            for k in range(self.M):
                            
                child_context = np.insert(current_context, 0, k)
                    
                self.add_node(Node(self, child_context.astype(int)))
        
    
    def seq_update(self):
        
        k = 0
        
        for i in range(self.D, self.seq.size):
            
            for j in range(self.D + 1):
                
                context = self.seq[j + k: i].astype(int)
                
                node = self.tree[str(context)]
                
                if i < self.seq.size - (self.D - context.size):
                
                    node.next_update(self.beta, i)
            
            k += 1   
            
        
        # Returns the final weighted probability of the root node
        return self.root.log_P_w
        
        
        
class Node:
    
    # Leaf_array is the index of the leaf in the leaves_matrix array 
    # to which the current context corresponds
    #def __init__(self, ctw, context, parent, leaf_array = 0):
    def __init__(self, ctw, context):   
        
        self.ctw = ctw
        
        # The count vector
        self.counts = np.zeros((self.ctw.M))
        
        self.context = context
        
        # Initiate the variable which will be an object
        self.parent = None
        
        # Sum of the elements of the count vector
        self.M_s = 0

        
        # We will sum the logarithms of the estimated probabilities
        # computed with the sequential update algorithm
        self.log_P_hat = 0
        
        # Logarithm of the Weighted probability of the node.
        self.log_P_w = 0
        
        # LEogarithm of the Maximal probability of the node for the BCT algorithm
        self.log_P_m = 0
        
        # This variable is an indicator. If the first term is the maximum in the 
        # BCT algorithm then a method in class Node will make it be equal to 1
        self.max_term = 0
        
        
        if self.context.size == 0:
            
            self.root = True
            
        else:
            
            self.root = False
        
        
        
        self.children = {}
        
        
        
    def get_children_parent(self):
        
        self.ctw.tree['[]'].parent = None 
        
        # tree elements
        keys = self.ctw.tree.keys()
        
        for tree_el in keys:
                
            node = self.ctw.tree[tree_el]
                    
            n = node.context.size
            
            # If one child is in the tree, then the others are too
            if n < self.ctw.D and str(np.insert(node.context, 0, 0)) in keys:
                
                for k in range(self.ctw.M):
                            
                    child_context = str(np.insert(node.context, 0, k))
                    
                    child = self.ctw.tree[child_context]
                    
                    node.children[child_context] = child
                    
                    child.parent = node
                        
    
    # Get the children of the self node only
    def get_node_children(self):
        
        n = self.context.size
        
        for tree_el in self.ctw.tree:

            node = self.ctw.tree[tree_el]
                
            if n + 1 == node.context.size:
                    
                if (self.context == node.context[-n:].astype(int)).all():    
                    
                    self.children[tree_el] = node
                    
        
        
    def estimated_prob(self):
        
        m = self.ctw.M
        
        self.M_s = self.counts.sum().astype(int)
        
        # Numerator
        log_num = 0
        
        # Denominator
        log_den = 0
        
        for i in range(m):
            
            for j in range(self.counts[i].astype(int)):
                
                log_num += np.log(j + 0.5)
                
        for k in range(self.M_s):
            
            log_den += np.log(m/2 + k)
        
        return log_num - log_den
    
    
        
    def weighted_prob(self, beta):
        
        # Checking if self is a leaf
        if self.children == {}:
            
            self.log_P_w = self.log_P_hat
            #self.P_w = self.estimated_prob()
            
            return self.log_P_w
        
        else:

            log_Pw_children = 0
            
            for child in self.children:
                    
                node = self.children[child]
                
                log_Pw_children += node.weighted_prob(beta)

            self.log_P_w = self.log_sum(np.log(beta) + self.log_P_hat, np.log(1 - beta) + log_Pw_children)
            
            return self.log_P_w
    
    
    #ln(a + b) = ln{exp[ln(a) - ln(b)] + 1} + ln(b)
    def log_sum(self, log_a, log_b):    
    
        # 2^52-1 = 4503599627370495. log of that is 36.043653389117155867651465390794
        # This branch is necessary to avoid shifted_a_ln = a_ln - b_ln having a too big value
        if abs(log_a - log_b) >= 36.043653389117155: 

            return max(log_a, log_b)  

        else:
            shifted_log_a = log_a - log_b
            
            shifted_sum = np.exp(shifted_log_a) + 1
            
            shifted_sum_log = np.log(shifted_sum)
            
            unshifted_sum_log = shifted_sum_log + log_b;
            
            return unshifted_sum_log
        
        
        
    def next_update(self, beta, i):
        
        m = self.ctw.M
        
        index = self.ctw.seq[i]
        
        self.counts[index] += 1
        
        self.M_s += 1
        
        self.log_P_hat += np.log((self.counts[index] - 0.5)/(m/2 + self.M_s - 1))
        
    
    
    def maximal_probability(self):
        
        log_beta = np.log(self.ctw.beta)
        
        # if the node is a leaf
        if self.children == {}:
            
            # if it's at depth D
            if self.context.size == self.ctw.D:
                
                self.log_P_m = self.log_P_hat
                
            else: self.log_P_m = log_beta
        
        else: 
            
            log_P_m_children = 0
            
            for child_key in self.children:
                
                child = self.children[child_key]
                
                log_P_m_children += child.maximal_probability()[0]
            
            # Log of the max is the max of the logs because log is an increasing function
            
            first_elem = log_beta + self.log_P_hat
            
            self.log_P_m = max(first_elem, np.log(1 - self.ctw.beta) + log_P_m_children)
            
            
            if self.log_P_m == first_elem: 
                
                self.max_term = 1
            
            
        return self.log_P_m, self.max_term
    
    


class BCT:
    
    def __init__(self, symbols, seq, depth = None, beta = None):
    
        self.ctw = CTW(symbols, seq, depth, beta)
        
        self.ctw.root.maximal_probability()

        self.prune_tree(self.ctw.root, self.ctw.D)
        
        self.ctw.list_leaves = []
        
        for key in self.ctw.tree.keys():
            
            if self.ctw.tree[key].children == {}:
                
                self.ctw.list_leaves.append(key)
                
        root = '[]'
        
        self.ctw.log_posterior = self.ctw.tree[root].log_P_m - self.ctw.tree[root].log_P_w
        
        self.ctw.posterior = np.exp(self.ctw.log_posterior)
        
        
        self.list_leaves = self.ctw.list_leaves
        
        self.posterior = self.ctw.posterior
        
        print("List of leaves: ", self.list_leaves)
        
        print("\nPosterior: ", self.posterior)
    
    
    
    def prune_tree(self, node, min_depth):
            
        if node.max_term == 1 and node.context.size < min_depth:
            
            min_depth = node.context.size
                    
                
        children_deleted = False
        
        
        for child_key in node.children:
                
            child = node.children[child_key]
                
            if child.children != {}:
                
                self.prune_tree(child, min_depth)
                
            if child.context.size > min_depth:
                
                del child
                
                del self.ctw.tree[child_key]
                
                children_deleted = True
                
            
        if children_deleted:
            
            node.children = {}

            
    # Checks if the tree estimated by BCT is the same as the inputted one
    def tree_is_equal(self, original_tree):

        len_list1 = len(self.list_leaves)

        len_list2 = len(original_tree)

        count = 0

        if len_list1 == len_list2:

            for leaf in original_tree:

                if leaf in self.list_leaves:

                    count += 1

        if count == len_list1:

            return True

        return False




# Implementation of the MCMC sampler

# Create class new_Tree with "artificial" subtree

# Takes as input a list of nodes which will form a new sub-tree (a dictionary)
# of the maximal tree with the same instances as the values

class new_Tree:
    def __init__(self, ctw, subtree):
        
        self.nodes = subtree
        
        self.ctw = ctw

        self.size = len(self.nodes)
        
        self.list_leaves = self.leaves()
        
        self.L_D = self.leaves_D()
        
        # List of internal nodes with only m descendants
        self.list_i_D = self.internal_D()
        
    
    def add_children(self, node):
        
        new_tree = {}

        # Copies the current tree on to the new tree
        for key in self.nodes:

            new_tree[key] = self.nodes[key]
        
        cont = str(np.insert(node.context, 0, 0))
        # Check if the node children are in the maximal tree
        if cont in self.ctw.tree.keys():
            
            # Get the children of the node
            node.get_node_children()

            # Adds the node children to the new tree
            for key in node.children:

                new_tree[key] = node.children[key]
                
            return new_Tree(self.ctw, new_tree)
        
                
        else:
            
            for k in range(self.ctw.M):
                
                # Gets the context
                child_context = np.insert(node.context, 0, k)
                
                # Creates the node object
                child = Node(self.ctw, child_context)
                
                child.parent = node
                
                new_tree[str(child_context)] = child
                
                node.children[str(child_context)] = child

            return new_Tree(self.ctw, new_tree)




    def remove_children(self, node):

        new_tree = {}

        # Copies the current tree on to the new tree except 
        # for the children we want to remove
        for key in self.nodes:

            if not key in node.children:

                new_tree[key] = self.nodes[key]
        
        # Removes the children of the node
        node.children = {}
        
        return new_Tree(self.ctw, new_tree)



    # Compares the two trees taken as inputs.
    # If they're the same, the function returns True,
    # and False otherwise

    def compares_trees(self, tree_A):

        count = 0

        # Number of leaves in each tree
        n_leaves_self = len(self.list_leaves)
        
        n_leaves_A = len(tree_A.list_leaves)
        
        if n_leaves_self == n_leaves_A:

            for leaf in self.list_leaves:

                if leaf in tree_A.list_leaves:

                    count += 1
        
        if count == n_leaves_self:

            return True


        return False
    
    
    # Returns list o leaves
    def leaves(self):
        
        list_leaves = []
        
        for key in self.nodes:
            
            if self.nodes[key].children == {}:

                list_leaves.append(key)


        return list_leaves
        

    # Returns the number of leaves the tree has at depth D
    def leaves_D(self):
        
        L_D = 0

        for key in self.list_leaves:

            if self.nodes[key].context.size == self.ctw.D:

                L_D += 1


        return L_D


    # Returns a list of the keys of the internal
    # nodes of the tree having only m descendants
    def internal_D(self):

        int_nodes = []
        
        for key in self.list_leaves:
            
            # In case the root node is a leaf
            if self.nodes[key].parent == None:
                
                break
                
            
            parent_node = str(self.nodes[key].parent.context)

            
            # Check if it's not a duplicate
            if not parent_node in int_nodes:
                
                # counts the number of children with no children
                desc = 0
                
                # Check if the node has exactly m descendants
                for child in self.nodes[parent_node].children:
                    
                    if self.nodes[child].children == {}:
                        
                        desc += 1
                    
                if desc == self.ctw.M:
                    
                    int_nodes.append(parent_node)

        return int_nodes


    # Computes the logarithm of the a priori probability of the tree and the
    # the logarithm of the likelihood probability, summing those two and resulting
    # in the logarithm of the numerator of Bayes formula
    def num_bayes(self):

        alpha = (1 - self.ctw.beta)**(1/(self.ctw.M - 1))

        # Number of leaves
        n_leaves = len(self.list_leaves)

        log_priori_prob = (n_leaves - 1) * np.log(alpha) + (n_leaves - self.L_D) * np.log(self.ctw.beta)

        # Logarithm of the produtory of the estimated probabilities of the
        # leaves which is equal to the likelihood
        log_prod = 0

        for key in self.list_leaves:

            log_prod += self.nodes[key].log_P_hat
            

        return log_priori_prob + log_prod
    
    
# Returns the number that multiplies the quocient of the
# two a posteriori probabilities, so we get the acceptance
# probability.
def mult_ratio(tree_i, new_tree, ctw, ctrl, max_size):
    
    len_i = len(tree_i.list_leaves)
    
    len_new = len(new_tree.list_leaves)
    
    len_max = max_size
    
    if ctrl == 1:
        
        return 1/2
    
    elif ctrl == 2:
        
        return ctw.M**(ctw.D - 1)/2
    
    elif ctrl == 3:
        
        if len_new != len_max:
            
            # Number of leaves that the tree i has at depth D
            tree_i.L_D
            
            # Number of internal nodes in the new tree having
            # only m descendants
            N_D = len(new_tree.list_i_D)

            return (len_i - tree_i.L_D)/N_D
        
        return 2 * ctw.M**(-ctw.D + 1)
    
    
    elif ctrl == 4:
        
        if len_new != 1:
            
            # Number of leaves that the new tree has at depth D
            new_tree.L_D
            
            # Number of internal nodes in the tree i having
            # only m descendants
            N_D = len(tree_i.list_i_D)
            
            return N_D/(len_new - new_tree.L_D)
        
        return 2
    


def RW_sampler(init_tree, n, ctw):
    
    # Maximum size of a tree of depth D
    max_size = 0
    
    for d in range(ctw.D + 1):
        
        max_size += ctw.M**d
    
    # This part is for removing the children of the elements that
    # are in init_tree but not in the maximal tree
    for key in ctw.tree:
        
        if key not in init_tree:
            
            ctw.tree[str(ctw.tree[key].parent.context)].children = {}
            
    # This is for the case in which the leaves of the maximal tree
    # have children from a past call of the RW_sampler function because
    # in our implementation we end up modifying the maximal tree.
    for key in ctw.list_leaves:
        
        ctw.tree[key].children = {}
            
    
    tree_i = new_Tree(ctw, init_tree)
    
    # List of trees
    tree_list = [tree_i]
    
    
    # We're gonna need these variables outside the if and elif
    # statements. That's why we define them here.
    new_tree = 0
    
    node = 0
    
    # Control variable to know from which if or else
    # statement the newly proprosed tree came from
    ctrl = 0
    
    for i in range(n):
        
        # Get the children the were removed so we could 
        # compute the weighted probability at the root node
        #tree_i.nodes['[]'].get_children()
        
        u = np.random.uniform()
        
        if len(tree_i.nodes) == 1: 
            
            node = tree_i.nodes['[]']
            
            new_tree = tree_i.add_children(node)
        
            ctrl = 1
            
            
        # If the tree has the same number of elements as the
        # maximal tree, then it means that by definition that tree
        # is equal to the maximal tree
        elif len(tree_i.nodes) == max_size:
            
            # Get nodes at depth D - 1
            nodes_D_1 = []
            
            for key in tree_i.nodes:
                
                if tree_i.nodes[key].context.size == ctw.D - 1:
                    
                    nodes_D_1.append(key)
                    
                    
            node = tree_i.nodes[random.choice(nodes_D_1)]
            
            new_tree = tree_i.remove_children(node) 
            
            
            ctrl = 2
            
        # Using the inverse transform method
        elif u <= 1/2:
            
            
            nodes = []
            
            for key in tree_i.nodes:
                
                # Get only the leaves at depth less than or equal to D - 1
                if (tree_i.nodes[key].children == {} and tree_i.nodes[key].context.size <= ctw.D - 1):  
                    
                    nodes.append(key)
                    
            
            node = tree_i.nodes[random.choice(nodes)]
            
            new_tree = tree_i.add_children(node) 
            
            ctrl = 3
        
        else: 
            
            # Gets the list of the internal nodes of the
            # current tree i having oly m descendants
            int_nodes = tree_i.list_i_D

            node = tree_i.nodes[random.choice(int_nodes)]
            
            new_tree = tree_i.remove_children(node)
        
            ctrl = 4
        
        
        # Computing the acceptance probability
        
        # Computing the log of the probabilities ratio, because otherwise
        # an overflow could happen here
        log_P_ratios = new_tree.num_bayes() - tree_i.num_bayes() + np.log(mult_ratio(tree_i, new_tree, ctw, ctrl, max_size))
        
        log_accept_prob = min(0, log_P_ratios)
        
        accept_prob = np.exp(log_accept_prob)
        
        u = np.random.uniform()
        
        # Using the inverse transform method
        if u <= accept_prob:
            
            tree_i = new_tree
            
            tree_list.append(tree_i)
        
        
        else: 
            
            tree_list.append(tree_i)
            
            # In case the new tree wasn't accepted we need to erase the new
            # children added or get back the children removed
            if ctrl == 1 or ctrl == 3:
                
                
                node.children = {}
            
            # Getting back the children attribute
            else:
                
                # Case in which the children are in the maximal tree
                if str(np.insert(node.context, 0, 0)) in ctw.tree.keys():
                    
                    # Get the children of the node
                    node.get_node_children()

                else:
                    
                    for k in range(ctw.M):
                        
                        # Gets the context
                        child_context = np.insert(node.context, 0, k)
                
                        node.children[str(child_context)] = tree_i.nodes[str(child_context)]
            
    
    # Returns the chain realization
    return tree_list



'''Returns a list whose elements are the tree nodes with the proportion it 
    appeared throughout the list of tree objects taken as input.'''

def time_tree(tree_list):
    
    # List to be returned
    prop_trees = []
    
    list_size = len(tree_list)
    
    # List of tree objects that were already counted so
    # we don't append duplicates
    already_counted = []
    
    for j in range(list_size):
        
        is_counted = False
        
        for counted in already_counted:
            
            if tree_list[j].compares_trees(counted):
                
                is_counted = True
                
                break
        
        if not is_counted:
        
            # Counts number of times the tree j
            # has appeared
            count = 0

            for k in range(list_size):

                if tree_list[j].compares_trees(tree_list[k]):

                    count += 1
            
            prop_trees.append([count/list_size, tree_list[j].list_leaves, tree_list[j]])
                
                
                
        already_counted.append(tree_list[j])
        
        
        
    
    # Sorting prop_trees so that the most seen trees in the list
    # come first
    
    # Function for sorting list according to the proportion
    def sorting_criterion(element):
        
        return element[0]
    
    
    prop_trees.sort(reverse = True, key = sorting_criterion)
    
    
    return prop_trees

# Function that combines the last two functions
def Bayesian_MCMC(init_tree, n, ctw):

    return time_tree(RW_sampler(init_tree, n, ctw))




