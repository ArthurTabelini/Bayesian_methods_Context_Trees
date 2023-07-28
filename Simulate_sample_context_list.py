import numpy as np
import random
import sys
import warnings



# For dealing with warnings
warnings.filterwarnings('error')

# In case we want to print the whole array
np.set_printoptions(threshold=sys.maxsize)




'''Relates the context of the chain to the respective row of the
context matrix so the next function can compute the next state
of the chain.
The parameters are the array with the last n elements of the
chain, where n is the greatest length of a context and the context
list.'''

def context_row(cont, cont_list):
    
    m = len(cont_list)

    n = len(cont)
    
    # Starts by the last row

    for j in range(m - 1, -1, -1):
        
        aux = np.array(cont_list[j])
        
        k = len(aux)
        
        # Tests if the context corresponds to a certain row
        count = 0
        
        context = cont[n - k:] # with cont[n - k:], it gets the last k elements
        
        for i in range(k):

            if aux[i] == context[i]: 

                count += 1

        if count == k:  

            # Returns the row number
            return j
        

'''Computes the next state of the chain, but doesn't return anything.
Its parameters are: the transition matrix, the current context, the context list, the whole chain and
the current index of the chain array.'''

def next_state(P, cont, cont_list, chain, i):
    
    n_states = len(P[0])
    
    # Gets row number
    j = context_row(cont, cont_list)

    u = np.random.uniform()
    
    # Uses inverse transform sampling to compute
    # the next state
    
    if u < P[j, 0]:
        
        chain[i] = 0
        
    else:
    
        for k in range(1, n_states):
        
            if np.sum(P[j, 0:k]) <= u < np.sum(P[j, 0:k + 1]):
            
                chain[i] = k
                
                # Breaks the loop so it doesn't run unecessary
                # tests in the above if statement
                break

                
                
                
                
''' What follows is the main function of the file. Its parameters are the length of the sample,
    the transition matrix and the context list.'''

def simulate_sample(n, P, cont_list):
    
    # Length of the initial context which is always
    # the length of the longest context 
    k1 = len(max(cont_list, key=len)) 
    
    # The initial context that will start the chain must
    # come from a leaf at the maximum depth, D, because otherwise
    # there would be a possibility we don't find a context matching
    # one in the context list within the first D observations.
    # We now iterate over the cont_list array to get these leaves and
    # choose one of them uniformly.
    
    # Array that will store the leaves
    choose_leaf = -1 * np.ones((1, k1))
    
    for j in range(len(cont_list)):
        
        if len(cont_list[j]) == k1:
            
            choose_leaf = np.append(choose_leaf, np.array([cont_list[j]]), 0)

        
    # Delete the sub-array with only -1
    choose_leaf = np.delete(choose_leaf, 0, 0)
    
    # The initial context
    cont = random.choice(choose_leaf)
    
    
    # Initiates chain array
    chain = np.empty(n)
    
    chain[:] = np.NaN
    
    
    # Adds the initial context to the chain array
    chain[:k1] = cont
    
    for i in range(k1, n):
        
        # Gets the last "k1" elements of the chain
        cont = chain[i - k1:i]
            
        next_state(P, cont, cont_list, chain, i)
            
            
            
    return chain.astype(int) 
