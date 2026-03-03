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

# Reactive Tabu search :
In non convex search space , traditional hill climbing algorithm or gradient-based methods can get struck in local minima. We want to escape the local min in the case shown below in the graph and we have to find the global minimum far away from the local search space.  Reactive Tabu Search (RTS) improves upon standard Tabu Search by dynamically adjusting the tabu list size and introducing an escape mechanism to avoid repeated solutions and escape local minima.
<img width="1400" height="804" alt="image" src="https://github.com/user-attachments/assets/ed78008e-2ac8-49fd-a627-4444e473c50e" />



