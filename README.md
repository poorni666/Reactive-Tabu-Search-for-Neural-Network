# Goal:
Implementing the paper of Reactive Tabu Search in Python and then using it as an Optimization method for Neural Network.
- Training neural nets with the reactive tabu search - https://pubsonline.informs.org/doi/10.1287/ijoc.6.2.126  
- The Reactive Tabu Search - https://ieeexplore.ieee.org/document/410361  

# Directory Structure :  
basic_tabu.py ├──      # Basic Tabu Search skeleton for solving the Quadratic Assignment Problem (QAP)   
RTS.py        ├──      # Reactive Tabu Search implementation for solving DQp (escape strategy)  
nn.ipynb      ├──      # Notebook for testing the trained model on MNIST  
Models/  
│
├── Checkpoint/    # Saved trained model checkpoints  
└── nn.py          # Neural network implementation using Reactive Tabu Search 
