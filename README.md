# SBI-Protein-HMM
Course project on simulation-based inference using Hidden Markov Models and BayesFlow for protein sequence analysis.

## Project Overview

## Project Structure

## Installation
## ِDevelopment

@sara: training issue
The data generation pipeline is now ready and connected to BayesFlow.

You can obtain a training batch using:

from src.simulator.bayesflow_simulator import bayesflow_simulator

batch = bayesflow_simulator.sample(batch_size=32)

x = batch["encoded_sequence"]
y = batch["state_probabilities"]

The batch contains:

encoded_sequence
Shape: (batch_size, sequence_length, 20)
One-hot encoded amino acid sequences.

state_probabilities
Shape: (batch_size, sequence_length, 2)
Posterior state probabilities computed by the Forward–Backward algorithm.
Column order:
0 = other
1 = alpha

The simulator, encoding, Forward–Backward algorithm, and BayesFlow wrapper have been implemented and tested.

The remaining work is the BayesFlow model itself:

Define the neural architecture.
Train the amortized posterior estimator.
Save/load the trained model.
Predict state probabilities for new sequences.

models/
│
├── amortized_posterior.py
│     Placeholder for the trained BayesFlow model.
│
└── train_bayesflow.py
      Placeholder for BayesFlow model training.

## Workflow

                HMM

                 │

          Simulator

                 │

        Protein Sequence

                 │

        Forward-Backward
       (Ground Truth Posterior)

                 │

      Training Dataset 

                 │

            BayesFlow

                 │

      Neural Posterior Estimator

                 │

         New Protein Sequence

                 │

        Predicted Posterior

                 │

       Compare with Ground Truth

## Contributors

    * Sara Davoodabadi, Matriculation: 262114
    * Sedigheh Mirzaei, Matriculation: 274433 
    * Forough Asgari, Matriculation: 270161
    * Anna Dustert, Matriculation: 209940
