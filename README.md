# Bayesian_methods_Context_Trees

# VLMC simulation

This code is written in Python and simulates a stochastic chain with memory of variable length accepting any number of contexts and states.

In order for the code to work properly, we must respect the following conventions: 

- The context matrix ("cont_mat"), which is a parameter of most functions throughout the code must have its contexts written from right to left and if the number of elements in a given context is less than the greatest size of a context, the remaining entries of that row are filled with None objects.

- The probabilities of each row of the transition matrix correspond to the context of same row number of the context matrix.

Now, a brief description of the functions present in the code:

- extract_context: Given an array which begins with null elements (represented by the keyword "None" in python), the function returns an array with only the elements which are not null of the array passed as a parameter.

- context_row: Relates the context of the chain to the respective row of the context matrix so the next function can compute the next state of the chain.
The parameters are the array with the last n elements of the chain, where n is the greatest length of a context and the context matrix.

- next_state: Computes the next state of the chain, but doesn't return anything. Its parameters are: transition matrix, current context, context matrix, whole chain, current index of the chain array and the array of the uniforms (for checking the results).

- simulate_CMVL: The main function of the code. Its parameters are the length of the chain, the transition matrix, the initial context to kick-start the chain and the 
context matrix.

# Bayesian Methods
