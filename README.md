# SBI-Protein-HMM
Course project on simulation-based inference using Hidden Markov Models and BayesFlow for protein sequence analysis.

## Contributors

    * Sara Davoodabadi, Matriculation: 262114
    * Sedigheh Mirzaei, Matriculation: 274433 
    * Forough Asgari, Matriculation: 270161
    * Anna Dustert, Matriculation: 209940

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

## Current Status

- HMM simulator implemented
- Forward–Backward inference implemented
- Offline synthetic datasets generated
- BiLSTM posterior estimator trained on 15,000 sequences
- Synthetic and real-protein evaluations completed
- Unit tests added

## Run Tests

pytest -q

## Run Inference

python -m src.scripts.check_trained_model

## Run Final Evaluation

python -m src.evaluate_final

## Create Figures

python -m src.evaluation.make_final_figures

State order:
0 = other
1 = alpha

Sequence lengths:
50–250

Splits:
15,000 train
2,000 validation
2,000 synthetic test
