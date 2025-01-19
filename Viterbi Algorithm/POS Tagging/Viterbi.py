import numpy as np


states = ["N", "V", "Adj"]
observations = ["Котката", "скача", "високо"]

start_prob = {"N": 0.6, "V": 0.3, "Adj": 0.1}
trans_prob = {
    "N": {"N": 0.4, "V": 0.4, "Adj": 0.2},
    "V": {"N": 0.5, "V": 0.3, "Adj": 0.2},
    "Adj": {"N": 0.3, "V": 0.3, "Adj": 0.4}
}
emit_prob = {
    "N": {"Котката": 0.8, "скача": 0.2, "високо": 0.3},
    "V": {"Котката": 0.1, "скача": 0.7, "високо": 0.1},
    "Adj": {"Котката": 0.1, "скача": 0.1, "високо": 0.6}
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

print("Най-добър път:", best_path_states)
print("Максимална вероятност:", best_path_prob)
