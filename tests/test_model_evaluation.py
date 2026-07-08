import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation import evaluate_model_prediction
from src.inference.forward_backward import forward_backward


seq = "ALEKGV"

# This is the true posterior from Forward-Backward
y_true = forward_backward(seq)

# Fake model prediction for now
# Later, replace this with BayesFlow prediction
y_pred = y_true + 0.02

# Normalize again so each row sums to 1
y_pred = y_pred / y_pred.sum(axis=1, keepdims=True)

results = evaluate_model_prediction(y_true, y_pred)

print("\nEvaluation Results")
print("-" * 30)

for metric, value in results.items():
    print(f"{metric:20s}: {value:.6f}")
