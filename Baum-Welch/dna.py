import numpy as np
from hmmlearn import hmm

# DNA -> (A=0, T=1, C=2, G=3)
sequence = np.array([[0, 3, 2, 2, 1, 0, 2, 0, 3, 3, 2, 0, 3, 0, 2, 3, 3, 2, 0, 1, 2]]).T


model = hmm.MultinomialHMM(n_components=2, n_iter=100, tol=0.01, random_state=42)


model.startprob_ = np.array([0.5, 0.5])


model.transmat_ = np.array([
    [0.8, 0.2],  # C -> C, C -> N
    [0.1, 0.9]   # N -> C, N -> N
])


model.emissionprob_ = np.array([
    [0.2, 0.3, 0.3, 0.2],  # Кодиращо (C)
    [0.3, 0.3, 0.2, 0.2]   # Некодиращо (N)
])


model.fit(sequence)


logprob, states = model.decode(sequence, algorithm="viterbi")

print("Sequence:", "ATGCGGCTTACGATAGGCTAC")
print("States (0=Coding, 1=Non-coding):", states)
