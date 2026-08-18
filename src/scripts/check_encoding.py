import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.preprocessing import encode_for_neural_network

seq = ["A", "L", "E", "K", "G"]

x_int = encode_for_neural_network(seq, max_length=10)
x_onehot = encode_for_neural_network(seq, max_length=10, one_hot=True)

print(x_int)
print(x_int.shape)
print(x_onehot.shape)