To test if R functions produce results identical to those in python:


- Run notebook [./experiments/generate_replication.ipynb](https://github.com/mollyow/contextual_bandits_evaluation/tree/main/experiments/generate_replication.ipynb)
  - This will run an experiment, and save experimental data:
    - `./experiments/results/yobs.csv`
    - `./experiments/results/ws.csv`
    - `./experiments/results/xs.csv`
    - `./experiments/results/ys.csv`
    - `./experiments/results/muxs.csv`
    - `./experiments/results/muhat.csv`
    - `./experiments/results/gammahat.csv`
    - `./experiments/results/probs.npy`
   - It will also produce estimated results within the notebook. 

 - Compile R script [./r-scripts/generate_replicationR.R](https://github.com/mollyow/contextual_bandits_evaluation/tree/main/r-scripts/generate_replicationR.R) using the identical data. 
 - Compare estimates. 
 