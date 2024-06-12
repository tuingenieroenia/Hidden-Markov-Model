import numpy as np

# Probabilidades de transición
trans_probs = np.array([[0.7, 0.3],  # De a a a, de a a b
                        [0.4, 0.6]]) # De b a a, de b a b

# Probabilidades de emisión
emission_probs = np.array([[0.8, 0.2],  # Probabilidad de observar "1" dado a, probabilidad de observar "2" dado a
                            [0.3, 0.7]]) # Probabilidad de observar "1" dado b, probabilidad de observar "2" dado b

# Probabilidades iniciales
initial_probs = np.array([0.6, 0.4])  # Probabilidad inicial de estar en el estado a o en el estado b

def forward_algorithm(observations, trans_probs, emission_probs, initial_probs):
    T = len(observations)
    n_states = trans_probs.shape[0]
    
    alpha = np.zeros((T, n_states))
    for t in range(T):
        if t == 0:
            alpha[t] = initial_probs * emission_probs[:, observations[t]]
        else:
            for j in range(n_states):
                alpha[t, j] = np.sum(alpha[t-1] * trans_probs[:, j]) * emission_probs[j, observations[t]]
    
    return alpha

def viterbi_algorithm(observations, trans_probs, emission_probs, initial_probs):
    T = len(observations)
    n_states = trans_probs.shape[0]
    
    delta = np.zeros((T, n_states))
    psi = np.zeros((T, n_states), dtype=int)
    
    # Paso inicial
    delta[0] = initial_probs * emission_probs[:, observations[0]]
    
    # Paso recursivo
    for t in range(1, T):
        for j in range(n_states):
            delta[t, j] = np.max(delta[t-1] * trans_probs[:, j]) * emission_probs[j, observations[t]]
            psi[t, j] = np.argmax(delta[t-1] * trans_probs[:, j])
    
    # Backtracking
    best_path = np.zeros(T, dtype=int)
    best_path[-1] = np.argmax(delta[-1])
    
    for t in range(T - 2, -1, -1):
        best_path[t] = psi[t + 1, best_path[t + 1]]
    
    return best_path

# Secuencia de observaciones (características de voz)
observations = [0, 1, 0, 1]  # Por ejemplo, "1 2 1 2"

# Aplicamos el algoritmo Forward
alpha = forward_algorithm(observations, trans_probs, emission_probs, initial_probs)
print("Alpha (Forward):\n", alpha)

# Aplicamos el algoritmo Viterbi
best_path = viterbi_algorithm(observations, trans_probs, emission_probs, initial_probs)
print("Viterbi Path:", best_path)
