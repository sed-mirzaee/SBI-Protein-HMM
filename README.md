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

          Simulator (Offline)

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

## Check Components

src.scripts.check_simulator.py
src.scripts.check_encoding.py
src.scripts.check_forward_backward.py
src.scripts.check_generate_dataset.py 
src.scripts.check_adapter
src.scripts.check_summary_network
src.scripts.check_inference_network
src.scripts.check_approximator
src.scripts.check_real_protein_inference


## Run Inference

python -m src.scripts.check_saved_native_model

## Run Final Evaluation

python -m 
src.bayesflow_model.evaluate_native_bayesflow

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
