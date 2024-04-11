
## Phase transitions in Hopfield Networks

The main objetive of this code is to test the abilities of Hopfield Networks to retain information (memories) as a function of their size $N$. Hopfield Networks are artificial neural networks that can be trained using **associative memories mechanisms** to recover memories aftyer being "corrupted" by noise and have numerous applications in pattern recognition and optimization problems. 


Read: https://en.wikipedia.org/wiki/Hopfield_network for more details

One interesting aspect of this networks is that as you increase the number of patterns that you want to store (a.k.a the network to remember) there is a critical point when the networks suddenly starts to fail and can't recover the patterns anymore. This behavior is very similar to phase transitions in magnetic systems and has been studied extensivly, for reference see this two papers:

1) Daniel Volk (1998): On the Phase Transition of Hopfield Networks — Another Monte Carlo Study

2) Violla Folli et al (2016) On the maximum storage capacity of the Hopfield Model


### Organization of the project

The Hopfield Networks are constructed using the Julia Porgramming Language mainly for speed and simplicity. The code for running the simulations is in **run_sim_hopfield.jl**, I have already prepared a bash file **reproduce_results** that re-reuns all the simulations but I also pre-simulated and stored the siulation results in the **results folder** if you just want to see the analysis in the *analysis.ipynb** jupyter-notebook .



### Results



