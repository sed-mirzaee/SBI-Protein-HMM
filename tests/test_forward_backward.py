import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.inference.forward_backward import forward_backward


seq = "ALEKGV"

posterior = forward_backward(seq)

print(posterior)
print(posterior.shape)
print(posterior.sum(axis=1))