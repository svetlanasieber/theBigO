import numpy as np

# Define states and observations
states = ["Noun", "Verb", "Adjective"]
observations = ["The cat", "jumps", "high"]


start_prob = {"Noun": 0.6, "Verb": 0.3, "Adjective": 0.1}


trans_prob = {
    "Noun": {"Noun": 0.4, "Verb": 0.4, "Adjective": 0.2},
    "Verb": {"Noun": 0.5, "Verb": 0.3, "Adjective": 0.2},
    "Adjective": {"Noun": 0.3, "Verb": 0.3, "Adjective": 0.4}
}


emit_prob = {
    "Noun": {"The cat": 0.8, "jumps": 0.2, "high": 0.3},
    "Verb": {"The cat": 0.1, "jumps": 0.7, "high": 0.1},
    "Adjective": {"The cat": 0.1, "jumps": 0.1, "high": 0.6}
}


T = len(observations)
viterbi = np.zeros((len(states), T))
backpointer = np.zeros((len(states), T), dtype=int)


for i, state in enumerate(states):
    viterbi[i, 0] = start_prob[state] * emit_prob[state][observations[0]]


for t in range(1, T):
    for i, state in enumerate(states):
        max_prob, best_state = max(
            (viterbi[j, t-1] * trans_prob[prev_state][state] * emit_prob[state][observations[t]], j)
            for j, prev_state in enumerate(states)
        )
        viterbi[i, t] = max_prob
        backpointer[i, t] = best_state


best_path_prob = max(viterbi[:, T-1])
best_last_state = np.argmax(viterbi[:, T-1])


best_path = [best_last_state]
for t in range(T-1, 0, -1):
    best_path.insert(0, backpointer[best_path[0], t])


best_path_states = [states[state] for state in best_path]


print("The best path:", best_path_states)
print("Maximum probability:", best_path_prob)
