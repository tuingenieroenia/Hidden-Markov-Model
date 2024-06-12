import numpy as np

class HMM:
    def __init__(self, n_states, n_symbols):
        self.n_states = n_states
        self.n_symbols = n_symbols
        self.trans_probs = np.zeros((n_states, n_states))
        self.emission_probs = np.zeros((n_states, n_symbols))
        self.initial_probs = np.zeros(n_states)

    def set_transitions(self, trans_probs):
        self.trans_probs = trans_probs

    def set_emissions(self, emission_probs):
        self.emission_probs = emission_probs

    def set_initial(self, initial_probs):
        self.initial_probs = initial_probs

    def forward(self, observations):
        T = len(observations)
        alpha = np.zeros((T, self.n_states))

        # Paso forward
        for t in range(T):
            if t == 0:
                alpha[t] = self.initial_probs * self.emission_probs[:, observations[t]]
            else:
                for j in range(self.n_states):
                    alpha[t, j] = np.sum(alpha[t-1] * self.trans_probs[:, j]) * self.emission_probs[j, observations[t]]

        return alpha

    def viterbi(self, observations):
        T = len(observations)
        delta = np.zeros((T, self.n_states))
        psi = np.zeros((T, self.n_states), dtype=int)

        # Paso inicial
        delta[0] = self.initial_probs * self.emission_probs[:, observations[0]]

        # Paso recursivo
        for t in range(1, T):
            for j in range(self.n_states):
                delta[t, j] = np.max(delta[t-1] * self.trans_probs[:, j]) * self.emission_probs[j, observations[t]]
                psi[t, j] = np.argmax(delta[t-1] * self.trans_probs[:, j])

        # Backtracking para encontrar el camino óptimo
        best_path = np.zeros(T, dtype=int)
        best_path[-1] = np.argmax(delta[-1])

        for t in range(T - 2, -1, -1):
            best_path[t] = psi[t + 1, best_path[t + 1]]

        return best_path

# Ejemplo de uso
n_states = 2
n_symbols = 3

hmm = HMM(n_states, n_symbols)
hmm.set_transitions(np.array([[0.7, 0.3],
                               [0.4, 0.6]]))
hmm.set_emissions(np.array([[0.1, 0.4, 0.5],
                             [0.6, 0.3, 0.1]]))
hmm.set_initial(np.array([0.6, 0.4]))

observations = [0, 2, 1, 1, 2]

alpha = hmm.forward(observations)
best_path = hmm.viterbi(observations)

print("Alpha (Forward):\n", alpha)
print("Viterbi Path:", best_path)
