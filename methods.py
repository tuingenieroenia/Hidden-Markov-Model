#On this file we describe the algorithms we will use to implement a Hidden Markov Models problem

#First we will import the libraries
import numpy as np

# Forward Algorithm
'''
Calculates the probability of observing a sequence of observations up to time t,
given a particular state at time t.
'''
def forward_algorithm(transition_probs, emission_probs, initial_probs, observations):
    N = len(initial_probs)
    T = len(observations)
    
    alpha = np.zeros((T, N))
    
    # Inicialization
    alpha[0, :] = initial_probs * emission_probs[:, observations[0]]
    
    # Recurrence
    for t in range(1, T):
        for j in range(N):
            alpha[t, j] = np.sum(alpha[t - 1, :] * transition_probs[:, j]) * emission_probs[j, observations[t]]
    
    return alpha

# Backward Algorithm 
'''
Calculates the probability of observing the sequence of observations
from time t+1 to the end, given a particular state at time t.
'''
def backward_algorithm(transition_probs, emission_probs, initial_probs, observations):
    N = len(initial_probs)
    T = len(observations)
    
    beta = np.zeros((T, N))
    
    # Inicialization
    beta[T - 1, :] = 1
    
    # Recurrence
    for t in range(T - 2, -1, -1):
        for i in range(N):
            beta[t, i] = np.sum(transition_probs[i, :] * emission_probs[:, observations[t + 1]] * beta[t + 1, :])
    
    return beta

# Viterbi Algorithm
'''
Find the most probable sequence of hidden states that could have produced a sequence of observations.
'''
def viterbi_algorithm(transition_probs, emission_probs, initial_probs, observations):
    N = len(initial_probs)
    T = len(observations)
    
    delta = np.zeros((T, N))
    psi = np.zeros((T, N), dtype=int)
    
    # Inicialization
    delta[0, :] = initial_probs * emission_probs[:, observations[0]]
    
    # Recurrence
    for t in range(1, T):
        for j in range(N):
            delta[t, j] = np.max(delta[t - 1, :] * transition_probs[:, j]) * emission_probs[j, observations[t]]
            psi[t, j] = np.argmax(delta[t - 1, :] * transition_probs[:, j])
    
    # Finish
    states = np.zeros(T, dtype=int)
    states[T - 1] = np.argmax(delta[T - 1, :])
    
    # Regression
    for t in range(T - 2, -1, -1):
        states[t] = psi[t + 1, states[t + 1]]
    
    return states