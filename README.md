# Neural Posterior Estimation for Protein Secondary-Structure Motifs

Course project for **Simulation-Based Inference** at TU Dortmund University.

This project studies position-wise inference of two hidden protein
secondary-structure states:

- `other`
- `alpha`

A two-state Hidden Markov Model (HMM) is used as the simulator.
For each simulated amino-acid sequence, the Forward–Backward algorithm
computes the exact posterior probabilities under the specified HMM.
A neural posterior estimator is then trained with BayesFlow to approximate
these posterior probabilities for new protein sequences.

---

## Contributors

- Forough Asgari
- Sara Davoodabadi
- Anna Dustert
- Sedigheh Mirzaei

---

## Inference Problem

For an observed amino-acid sequence

```text
x_1:T = (x_1, ..., x_T)
```

the hidden state at every position is

```text
z_t ∈ {other, alpha}.
```

The goal is to estimate

```text
P(z_t = other | x_1:T)
P(z_t = alpha | x_1:T)
```

for every sequence position.

The state order used throughout the project is:

```text
0 = other
1 = alpha
```

---

## Project Workflow

```text
Two-state HMM
      │
      ▼
Simulate hidden states and amino-acid sequences
      │
      ▼
Forward–Backward algorithm
Exact posterior under the specified HMM
      │
      ▼
Synthetic training pairs
(sequence, position-wise posterior)
      │
      ▼
BayesFlow Adapter
      │
      ▼
BiLSTM Summary Network
      │
      ▼
BayesFlow ScoringRuleNetwork
      │
      ▼
ScoringRuleApproximator
      │
      ▼
Amortized position-wise posterior inference
      │
      ├── Synthetic test sequences
      └── Real protein sequence
```

After training, inference requires only a forward pass through the neural
network. The Forward–Backward algorithm is used as the reference method
for synthetic data and for model-based comparison on a real protein
sequence.

---

## Model and Data

### HMM simulator

The simulator generates:

- a hidden state sequence,
- an amino-acid sequence conditioned on the hidden states.

Every sequence starts in the `other` state.

### Synthetic data

The stored datasets contain:

```text
x              padded one-hot protein sequences
y              Forward–Backward posterior probabilities
mask           valid-position mask
lengths        original sequence lengths
hidden_states  simulated hidden state labels
```

Main shapes:

```text
x:              (N, 250, 20)
y:              (N, 250, 2)
mask:           (N, 250)
lengths:        (N,)
hidden_states:  (N, 250)
```

Sequence lengths vary uniformly between 50 and 250.
All sequences are padded to length 250.

The mask is used to:

- prevent padded positions from affecting the BiLSTM representation,
- exclude padded positions from the training loss,
- exclude padded positions from evaluation metrics.

### Data splits

```text
Training:    15,000 synthetic sequences
Validation:   2,000 synthetic sequences
Test:         2,000 synthetic sequences
```

Smaller training runs with 2,000 and 8,000 sequences were also performed
before the final 15,000-sequence run.

---

## Native BayesFlow Architecture

The final BayesFlow pipeline is implemented in:

```text
src/bayesflow_model/
```

Main components:

```text
adapter.py
    Maps project-specific arrays to BayesFlow keys.

data.py
    Loads and validates offline synthetic datasets.

summary_network.py
    Sequence-preserving bidirectional LSTM.

inference_network.py
    BayesFlow ScoringRuleNetwork for posterior probabilities.

approximator.py
    Connects the adapter, summary network, and inference network
    through a ScoringRuleApproximator.

train_native_bayesflow.py
    Trains, validates, checkpoints, and saves the native BayesFlow model.
```

The network predicts two probabilities at every valid sequence position:

```text
[P(other | sequence), P(alpha | sequence)]
```

The two probabilities sum to one.

---

## Repository Structure

```text
SBI-Protein-HMM/
│
├── data/
│   └── synthetic/
│
├── docs/
├── notebooks/
│
├── outputs/
│   └── native_bayesflow/
│       ├── evaluation___/
│       ├── figures___/
│       ├── real_protein/
│       ├── training_2000/
│       ├── training_8000/
│       └── training_15000/
│
├── src/
│   ├── bayesflow_model/
│   ├── configs/
│   ├── evaluation/
│   ├── inference/
│   ├── preprocessing/
│   ├── scripts/
│   └── simulator/
│
├── requirements.txt
└── README.md
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/sed-mirzaee/SBI-Protein-HMM.git
cd SBI-Protein-HMM
git checkout bayesflow-training
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Linux or macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The project was developed with the BayesFlow Torch backend.

---

## Component Checks

Run commands from the repository root.

### Simulator

```bash
python -m src.scripts.check_simulator
```

### Sequence encoding

```bash
python -m src.scripts.check_encoding
```

### Forward–Backward algorithm

```bash
python -m src.scripts.check_forward_backward
```

### Dataset generation and validation

```bash
python -m src.scripts.check_generate_dataset
```

### BayesFlow adapter

```bash
python -m src.scripts.check_adapter
```

### Summary network

```bash
python -m src.scripts.check_summary_network
```

### Inference network

```bash
python -m src.scripts.check_inference_network
```

### Complete BayesFlow approximator

```bash
python -m src.scripts.check_approximator
```

### Saved-model loading and target-leakage check

```bash
python -m src.scripts.check_saved_native_model
```

### Real-protein inference

```bash
python -m src.scripts.check_real_protein_inference
```

---

## Training

The native BayesFlow model is trained using:

```bash
python -m src.bayesflow_model.train_native_bayesflow
```

Training configuration, dataset limits, output directory, batch size,
learning rate, and number of epochs can be changed in:

```text
src/bayesflow_model/train_native_bayesflow.py
```

The training script provides:

- offline BayesFlow datasets,
- training and validation loss,
- early stopping,
- learning-rate reduction,
- best-model checkpointing,
- saved training history,
- total training time.

Saved training runs are available in:

```text
outputs/native_bayesflow/training_2000/
outputs/native_bayesflow/training_8000/
outputs/native_bayesflow/training_15000/
```

---

## Evaluation

Run the synthetic test evaluation with:

```bash
python -m src.bayesflow_model.evaluate_native_bayesflow
```

Evaluation is performed only on valid, non-padding positions.

The reported metrics include:

- mean absolute error,
- mean squared error,
- root mean squared error,
- soft cross-entropy,
- alpha-probability MAE,
- neural hidden-state accuracy,
- Forward–Backward hidden-state accuracy,
- neural–Forward–Backward state agreement.

Evaluation outputs are saved under:

```text
outputs/native_bayesflow/evaluation___/
```

---

## Real-Protein Inference

A real amino-acid sequence is:

1. validated,
2. one-hot encoded,
3. padded to length 250,
4. passed through the trained BayesFlow model,
5. compared with the Forward–Backward posterior.

Run:

```bash
python -m src.scripts.check_real_protein_inference
```

Position-wise results are saved under:

```text
outputs/native_bayesflow/real_protein/
```

The comparison with Forward–Backward is a model-based comparison under
the specified HMM. It should not be interpreted as a complete biological
validation of protein secondary structure.

---

## Figures

Create the final figures with:

```bash
python -m src.evaluation.make_final_figures
```

Generated figures include:

1. training and validation history,
2. synthetic posterior example,
3. predicted versus reference posterior scatter plot,
4. posterior error distribution,
5. synthetic accuracy comparison,
6. synthetic Forward–Backward confusion matrix,
7. synthetic BayesFlow confusion matrix,
8. real-protein posterior comparison,
9. real-protein accuracy comparison,
10. real-protein Forward–Backward confusion matrix,
11. real-protein BayesFlow confusion matrix.

Figures are saved under:

```text
outputs/native_bayesflow/figures___/
```

---

## Current Status

- [x] HMM simulator implemented and checked
- [x] Sequence encoding implemented and checked
- [x] Forward–Backward inference implemented and checked
- [x] Variable-length synthetic datasets generated
- [x] Padding and masking implemented
- [x] Native BayesFlow adapter implemented
- [x] BiLSTM summary network implemented
- [x] BayesFlow inference network implemented
- [x] ScoringRuleApproximator implemented
- [x] Models trained on 2,000, 8,000, and 15,000 sequences
- [x] Saved-model loading verified
- [x] Inference without true targets verified
- [x] Target-leakage check passed
- [x] Synthetic test evaluation completed
- [x] Real-protein inference completed
- [x] Final figures generated

---

## Important Interpretation

The Forward–Backward posterior is exact for the specified HMM parameters.
The neural network learns an amortized approximation to this posterior.

Therefore, the project demonstrates:

```text
simulation
→ exact model-based posterior computation
→ neural posterior approximation
→ fast amortized inference
```

It does not claim that the simplified two-state HMM is a complete
biological model of protein secondary structure.