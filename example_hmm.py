'''
This is the main file where we will implement an example of HMM 
using the foward-backward algorithm + viterbi algorithm
'''

# First we import the libraries
import numpy as np

# Then we import our methods module
import methods

# We define the transition matrix
transition_probs = np.array([
    [0.7, 0.3],
    [0.4, 0.6]
])

# We define the emission matrix
emission_probs = np.array([
    [0.2, 0.4, 0.4],
    [0.5, 0.4, 0.1]
])

# We define the initial probabilities
initial_probs = np.array([0.6, 0.4])

# We initialize the observations
observations = [2, 1, 0]

# Let's execute the algorithms
alpha = methods.forward_algorithm(transition_probs, emission_probs, initial_probs, observations)
beta = methods.backward_algorithm(transition_probs, emission_probs, initial_probs, observations)
states = methods.viterbi_algorithm(transition_probs, emission_probs, initial_probs, observations)

# Show the results 
print("Alpha (Forward probabilities):")
print(alpha)
print("\nBeta (Backward probabilities):")
print(beta)
print("\nMost likely states sequence (Viterbi):")
print(states)