# DeepKidney
A geometric deep learning approach to the kidney exchange matching problem.

## Background
Contained within this repository is code that generates data (compatibility graphs), an implementation of Graph Neural Networks using PyTorch Geometric, and a Monte Carlo Tree Search (MCTS) implementation. There are also several notebooks included to get you started on the basics, such as the format of the random compatiblity graphs and the usage of our MCTS algorithm. 

## Installation

Start by cloning this repository by using `git clone`.

These repository requries several Python 3 Dependencies. The dependencies are listed in the requirements.txt file. We recommend using Conda to install the packages, and resort to pip if necessary.

There are several dependencies that require special installation. Firstly, Gurobi and PySCIPopt require special academic licenses. Although we are planning to move away from Gurobi, the current codebase still relies on it. The python packages are merely wrappers for these commericial solvers. 

Pytorch Geometric can be quite tricky to setup, and also requires special dependencies of its own. 
## License
MIT License
