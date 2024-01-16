# inverse-map-karumuri-2023
This repository replicates the result of the paper "Learning to solve Bayesian inverse problems: An amortized variational inference approach using Gaussian and Flow guides."

## **Learning to solve Bayesian inverse problems: An amortized variational inference approach using Gaussian and Flow guides**
[Sharmila Karumuri](https://scholar.google.com/citations?user=uY1G-S0AAAAJ&hl=en) and [Ilias Bilionis](https://scholar.google.com/citations?user=rjXLtJMAAAAJ&hl=en).

Our paper proposes an approach to solve Bayesian inverse problems in real-time. Our method is intended to overcome the need for solving inverse problems from scratch for each set of new data. We do this by learning a map from data to posteriors (Bayesian inverse map).

#### The novel features of our approach are as follows:

1.	We parameterize the posterior distribution as a function of data. This work outlines two distinct approaches to do this -

    a. The first method involves parameterizing the posterior using an amortized full-rank Gaussian guide, implemented through neural networks.
    
  	b. The second method utilizes a Conditional Normalizing Flow guide, employing conditional invertible neural networks for cases where the target posterior is arbitrarily complex. 
2.	In both approaches,  we learn the network parameters by amortized variational inference which involves maximizing the expectation of evidence lower bound over all possible datasets compatible with the model.
3.	Once trained posterior estimates are available on-the-fly just at the cost of the forward pass of the network.

## Code outline

[examples](https://github.com/PredictiveScienceLab/paper-2023-inverse-map-karumuri/tree/main/examples): Contains the Python scripts that implements the examples discussed in the paper, along with the code for generating the corresponding plots. The three examples are organized into  separate folders.

Within each example folder, you will find a Jupyter notebook named 'AVI_fullrankGaussian.ipynb'. This notebook demonstrates the process of learning the Bayesian inverse map for the specific example. The results of the computations are saved in the ```./results``` folder.

Additionally, a comparison has been made between the posteriors learned from our approach and MCMC. The results of this comparison can be found in the ```./results_metrics``` folder. The estimation of the comparison metrics is carried out in the 'metrics_evaluation.ipynb' jupyter notebook.

The forward model, which maps parameters to observable quantities, is implemented in the following Jupyter notebooks:
* [Surrogate.ipynb](https://github.com/PredictiveScienceLab/paper-2023-inverse-map-karumuri/tree/main/examples/1-Damage_location_detection/Surrogate.ipynb): This notebook contains the implementation of the forward model for [example 1](https://github.com/PredictiveScienceLab/paper-2023-inverse-map-karumuri/tree/main/examples/1-Damage_location_detection).
* [pinn.ipynb](https://github.com/PredictiveScienceLab/paper-2023-inverse-map-karumuri/tree/main/examples/2-1D_heateqn_Wiener-Levy_process/pinn.ipynb): This notebook contains the implementation of the forward model for [example 2](https://github.com/PredictiveScienceLab/paper-2023-inverse-map-karumuri/tree/main/examples/2-1D_heateqn_Wiener-Levy_process).

## Installing

The code for examples is written in pytorch. Install dependencies at [requirements.txt](https://github.com/PredictiveScienceLab/paper-2023-inverse-map-karumuri/tree/main/requirements.txt) and clone our repository
```
git clone https://github.com/PredictiveScienceLab/paper-2023-inverse-map-karumuri.git
cd paper-2023-inverse-map-karumuri
```

### Citation:
If you use this code for your research, please cite our paper https://arxiv.org/abs/2305.20004.



