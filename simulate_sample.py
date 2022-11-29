import numpy as np
import random
import sys
import warnings



# For dealing with warnings
warnings.filterwarnings('error')

# In case we want to print the whole array
np.set_printoptions(threshold=sys.maxsize)


# Given an array which begins with null elements (represented by the keyword "None"), 
# the function extract context returns an array with only the elements which are not 
# null of the array passed as a parameter.

def extract_context(cont):
    
    i = 0
    
    while cont[i] == None:
        
        i += 1
       
    return cont[i:]


'''Relates the context of the chain to the respective row of the
context matrix so the next function can compute the next state
of the chain.
The parameters are the array with the last n elements of the
chain, where n is the greatest length of a context and the context
matrix.'''

def context_row(cont, mat_cont):
    
    m = len(mat_cont[:, 0])

    n = len(cont)
    
    # Starts by the last row

    for j in range(m - 1, -1, -1):
        
        aux = extract_context(mat_cont[j, :])
        
        k = len(aux)
        
        # Tests if the context corresponds to a certain row
        count = 0
        
        #print(cont)
        
        context = cont[n - k:] # with cont[n - k:], it gets the last k elements
        
        #print(context.size, k)
        
        for i in range(k):

            if aux[i] == context[i]: 

                count += 1

        if count == k:  

            # Returns the row number
            return j

        
'''Computes the next state of the chain, but doesn't return anything.
Its parameters are: the transition matrix, the current context, the context matrix, the whole chain,
the current index of the chain array and the array of the uniforms(for checking the results).'''

def next_state(P, cont, mat_cont, chain, i, unif_array):
    
    n_states = len(P[0])
    
    # Gets row number
    j = context_row(cont, mat_cont)

    u = np.random.uniform()
    
    unif_array[i] = u 
    
    # Uses inverse transform sampling to compute
    # the next state
    #print(u, P[j, 0])
    
    if u < P[j, 0]:
        
        chain[i] = 0
        
    else:
    
        for k in range(1, n_states):
        
            if np.sum(P[j, 0:k]) <= u < np.sum(P[j, 0:k + 1]):
            
                chain[i] = k
                
                # Breaks the loop so it doesn't run unecessary
                # tests in the above if statement
                break

                
                
                
                
''' What follows is the main function of the code. Its parameters are the length of the chain,
    the transition matrix, the initial context to kick-start the chain and the context matrix.'''

def simulate_sample(n, P, mat_cont):
    
    # Length of the initial context
    #k1 = len(cont)
    k1 = len(mat_cont[0,:])
    
    # The initial context that will start the chain must
    # come from a leaf at the maximum depth, D, because otherwise
    # there would be a possibility we don't find a context matching
    # one in the context matrix within the first D observations.
    # We now iterate over the mat_cont array to get these leaves and
    # choose one of them uniformly.
    
    # Array that will store the leaves
    choose_leaf = -1 * np.ones((1, k1))
    
    for j in range(len(mat_cont[:, 0])):
        
        if mat_cont[j, 0] != None:
            
            choose_leaf = np.append(choose_leaf, np.array([mat_cont[j, :]]), 0)

        
    # Delete the sub-array with only -1
    choose_leaf = np.delete(choose_leaf, 0, 0)
    
    # The initial context
    cont = random.choice(choose_leaf)
    
    
    # Initiates chain array
    chain = np.empty(n)
    
    chain[:] = np.NaN
    
    # Initiates array of the uniforms
    unif_array = np.empty(n)
    
    unif_array[:] = np.NaN
    
    # Greatest length of a context
    l = len(mat_cont[0, :])
    
    # Adds the initial context to the chain array
    chain[:k1] = cont
    
    for i in range(k1, n):
        
        # Gets the last "k1" elements of the chain
        cont = chain[i - k1:i]
            
        next_state(P, cont, mat_cont, chain, i, unif_array)
            
            
            
    return chain.astype(int) #, unif_array
