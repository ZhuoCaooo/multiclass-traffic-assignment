# MULTICLASS-Traffic-Assignment-Zhuo
FEATURES:

This is modified based on code from `MatteoBettini/Traffic-Assignment-Frank-Wolfe-2021` to simply test the multiclass traffic assignment problems.
-----Traffic Assignments on multicalss Vehicles
-----Data Extraction
-----Data Visuallization

It can compute the **User Equilibrium (UE)** assignment or the **System Optimal (SO)** assignment.

The travel time cost function that models the effect of congestion on travel time is pluggable and definable by the users. And there are two kinds of BPRcostfunctions for CAV (connected and autonomous vehicles) and HDV (human driven vehciles) respectively with corresponding flow sensitivity expression.

Currently, three cost function implementations are available:
* BPR cost function ([see more](https://rdrr.io/rforge/travelr/man/bpr.function.html)) -----used for multiclass traffic assignment.
* Greenshields cost function (see Greenshields, B. D., et al. "A study of traffic capacity." Highway research board proceedings. Vol. 1935. National Research Council (USA), Highway Research Board, 1935.)
* Constant cost function (no congestion effects)

Our implementation has been tested against all the networks for which a solution is available on [TransportationNetworks](https://github.com/bstabler/TransportationNetworks) and has always obtained the correct solution.

# How to use

First, clone the repository to a local directory.

To use the algorithm, simply call the `computeAssingment()` method in the `assignment.py` script.

The documentation of the method provides a through description of all the available parameters and their meaning.

# Importing networks
 Networks and demand files must be specified in the TNTP data format.
 
 A through description of the TNTP format and a wide range of real transportation networks to test the algorithm on is avaialble at [TransportationNetworks](https://github.com/bstabler/TransportationNetworks).

 
 # Acknowledgments
 
* This work is based on [Traffic-Assignment](https://github.com/prameshk/Traffic-Assignment). and MatteoBettini/Traffic-Assignment-Frank-Wolfe-2021.  I focused on implementation of multiclass traffic assignments based on the work from precursors.
* All the networks used for testing the correctness of the algorithm are available at [TransportationNetworks](https://github.com/bstabler/TransportationNetworks).
* further implementation and development will keep uploading as the record of my thesis study in UNSW.
