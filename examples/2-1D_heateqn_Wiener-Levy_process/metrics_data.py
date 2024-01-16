#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# get_ipython().system('pip install pyro-ppl')


# In[ ]:


# get_ipython().system('pip install scikit-learn==0.22.1')


# In[ ]:


import numpy as np
import torch
from torch import nn
import torch.distributions as dist
import matplotlib.pyplot as plt
import scipy.stats as st
import pandas as pd
import math
import os
import arviz as az
import random
# get_ipython().run_line_magic('matplotlib', 'inline')

from networks import *

import seaborn as sns
sns.set_style("white")
plt.rcParams.update({'font.size': 9})

import warnings
warnings.filterwarnings("ignore")

np.random.seed(37)
torch.manual_seed(37)
random.seed(37)


# In[ ]:


save = True


# ## Propagating Uncertainty Through a Differential Equation
# Consider the steady state heat equation on a heterogeneous rod with no heat sources:
# $$
# \frac{d}{dx}\left(a(x)\frac{d}{dx}u(x)\right) = 0,
# $$
# and boundary values:
# $$
# u(0) = 1\;\mbox{and}\;u(1) = 0.
# $$
# We are interested in cases in which we are uncertain about the conductivity, $a(x)$.
# Before we proceed, we need to put together all our prior beliefs and come up with a stochastic model for $a(x)$ that represents our uncertainty.
# This requires assigning a probability measure on a function space.
# For now, we will just give you a model.
# We will model $a = a(x;\boldsymbol{\xi})$ as:
# $$
# a(x;\boldsymbol{\xi}) = \exp\{g(x;\boldsymbol{\xi})\},
# $$
# where  $g(x;\boldsymbol{\xi})$ is a random field.
# The reason for the exponential is that $a(x;\boldsymbol{\xi})$ must be positive.
# We will assume that the random field ia a [Wiener-Lévy process](https://en.wikipedia.org/wiki/Wiener_process).
# This is a field that it is no-where continuous and it is actually a fractal (when you zoom in the spatial dimension, the field resembles itself at a larger scale).
# The Karhunen-Loeve expansion of the field is:
# $$
# g(x;\boldsymbol{\xi}) = \sum_{i=1}^d\xi_i\phi_i(x),
# $$
# where $\phi_i(x)$ are the eigenfunctions of $x$ and $\xi_i$ are independent standard normal random variables with zero mean and unit variance.
# For this particular example, we will assume that:
# $$
# \phi_i(x) = \frac{\sqrt{2}\sigma}{(i - \frac{1}{2})\pi}\sin\left((i-\frac{1}{2})\pi x\right),
# $$
# where $\sigma>0$ is a parameter controlling the variance of the random field.
# 
# The field is implemented as function 'g'.

# ## Inverse problem
# $$p(\boldsymbol{\xi}|\textbf{y}) \propto p(\textbf{y}|\boldsymbol{\xi}) p(\boldsymbol{\xi})$$
# $$\text{Prior: } p(\xi) = N(0, I)$$
# $$\text{Likelihood: } p(\textbf{y}|\boldsymbol{\xi}) =  N(\textbf{y}| f(x, \boldsymbol{\xi}), \gamma^2); \gamma = 0.015$$
# $$\text{Approximate posterior: }q(\boldsymbol{\xi}|\textbf{y})$$

# In[ ]:


d_xi = 5 #  number of terms to consider


# In[ ]:


x_gt = torch.linspace(0., 1, 100)
x_ticks = torch.linspace(0.15, 0.85, 9) # Measurements of output at this x-locations
print(x_ticks)


# In[ ]:


# loading trained pinn model
pinn_state_dict = torch.load('model_state_dict_pinn.pt')
pinn = DenseResNet(dim_in=1+(d_xi), dim_out=1, num_resnet_blocks=5, 
                 num_layers_per_block=3, num_neurons=40, activation=nn.SiLU())
pinn.load_state_dict(pinn_state_dict)

B1 = 1 # boundary_value_left
B2 = 0 # boundary_value_right
u_trail = lambda x, inputs, net: (B1*(1-x))+(B2*x)+(x*(1-x)*net(inputs))


def forward_process(xi): # N x d_xi
    """
    Implements the forward process f(xi) and 
    returns the output field values at the measurement locations.
    """
    xi = xi.reshape(-1,d_xi)
    outputs = torch.zeros(xi.shape[0], len(x_ticks))
    for i in range(xi.shape[0]):
        inputs_i = torch.hstack([x_ticks.reshape(-1,1), xi[i].repeat(len(x_ticks),1)])
        u_pred_i = u_trail(x_ticks.reshape(-1,1), inputs_i, pinn) 
        outputs[i] = u_pred_i.flatten()
    return outputs


# In[ ]:


class ResNet(torch.nn.Module):
    """
    Implements the residual network
    """
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, inputs):
        return self.module(inputs) + inputs


# In[ ]:


def unpack_cholesky(L_diag, L_offdiag, xi_dim):
    """
    Constructs cholesky L matrix from diagonal and offdiagonal elements.
    """
    chol_diag = torch.diag(L_diag)
    
    chol_offdiag = torch.zeros((xi_dim, xi_dim))
    tril_indices = torch.tril_indices(row=xi_dim, col=xi_dim, offset=-1)
    chol_offdiag[tril_indices[0], tril_indices[1]] = L_offdiag

    q_L = chol_diag + chol_offdiag # here q_L is L matrix
    return q_L 


# In[ ]:


class Amortized_VI(nn.Module):
    """
    Class that performs Amortized Variational inference.
    xi_dim: Number of dimensions in the variables to infer.
    y_dim: Number of measurement observations.
    prior_xi_dist: prior distribution defined using torch.distributions.
    f: function defining the forward process.
    noise_scale: Standard deviation of likelihood or the measurement process.
    """
    def __init__(self, xi_dim=2*5, y_dim=1, prior_xi_dist=None, f=forward_process, noise_scale=0.02):
        super().__init__()
        
        self.xi_dim = xi_dim
        self.y_dim = y_dim
        self.prior_xi_dist = prior_xi_dist
        self.f = f
        self.noise_scale = noise_scale
        
        self.mu = nn.Sequential(
            nn.Linear(self.y_dim, 50),
            nn.ReLU(),
            nn.Linear(50, 40),
            nn.ReLU(),
            nn.Linear(40, 30),
            nn.ReLU(),
            nn.Linear(30, 20),
            nn.ReLU(),
            nn.Linear(20, self.xi_dim))
        
        self.L_diag = nn.Sequential(
            nn.Linear(self.y_dim, 50),
            nn.ReLU(),
            nn.Linear(50, 40),
            nn.ReLU(),
            nn.Linear(40, 30),
            nn.ReLU(),
            nn.Linear(30, 20),
            nn.ReLU(),
            nn.Linear(20, self.xi_dim),
            nn.Softplus()
        )
        
        self.L_offdiag = nn.Sequential(
            nn.Linear(self.y_dim, 50),
            nn.ReLU(),
            nn.Linear(50, 40),
            nn.ReLU(),
            nn.Linear(40, 30),
            nn.ReLU(),
            nn.Linear(30, 20),
            nn.ReLU(),
            nn.Linear(20, int(self.xi_dim*(self.xi_dim+1)/2)-self.xi_dim)
        )
        
    def observed_data(self, n=30):
        xi_data = self.prior_xi_dist.sample([n])  
        y_data = self.f(xi_data) + self.noise_scale * torch.randn(*self.f(xi_data).shape)

        return xi_data, y_data
    
    def forward(self, num_particles=2):
        _, y = self.observed_data(n=64)
        
        q_mu = self.mu(y)
        q_L_diag = self.L_diag(y)
        q_L_offdiag = self.L_offdiag(y) #or 0.01*self.L_offdiag(y)

        #print(q_mu, q_L_diag, q_L_offdiag)
        
        loss = 0
        for j in range(y.shape[0]):
            
            ##### elbo for one data point 
            q_Lj = unpack_cholesky(q_L_diag[j,:], q_L_offdiag[j,:], self.xi_dim) # q_Lj is L matrix    
            
            xi_samples = torch.zeros(num_particles, self.xi_dim)
            for k in range(num_particles):
                zs = torch.randn_like(q_mu[j,:])
                xi_samples[k] = q_mu[j,:] + torch.matmul(q_Lj, zs)# reparametrization

            datafit = 0.
            for i in range(num_particles):
                log_prior = self.prior_xi_dist.log_prob(xi_samples[i]).sum()
                log_likelihood = dist.Normal(loc=self.f(xi_samples[i])[0], scale=self.noise_scale).log_prob(y[j,:]).sum()
                log_joint = log_prior + log_likelihood
                # print(log_prior, log_likelihood)
                datafit += log_joint
            datafit = datafit/num_particles

            entropy = dist.MultivariateNormal(loc=q_mu[j,:], scale_tril=q_Lj).entropy()
            # print(entropy)
            elbo = datafit + entropy
            #####
            loss += elbo
        
        return loss/y.shape[0]


# In[ ]:


# Defining prior distribution
prior_xi_dist = dist.Normal(loc=torch.zeros(d_xi), scale=torch.ones(d_xi))
noise_scale = 0.015
model = Amortized_VI(xi_dim=d_xi, y_dim=len(x_ticks), prior_xi_dist=prior_xi_dist, f=forward_process, noise_scale=noise_scale)
model.load_state_dict(torch.load('model_state_dict_AVI.pt'))


# In[ ]:


results_dir = os.path.join(os.getcwd(), 'results_metrics/')
if not os.path.isdir(results_dir):
    os.makedirs(results_dir)


# In[ ]:


xi_data_, y_data_ = model.observed_data(n=100)
print(xi_data_.shape, y_data_.shape)
print('Groundtruth:\n xi_data='+str(xi_data_.data.numpy())+',\n y_data='+str(y_data_.data.numpy()))

torch.save(xi_data_, results_dir +'xi_data.pt')
torch.save(y_data_, results_dir +'y_data.pt')


# In[ ]:


# This computes the random field given a xi
def g(x, xi, sigma=1.):
    """
    Arguments:
    x     -   Column containing the points at which you wish to evaluate the field.
    xi    -   Array of the random variables. The number of columns correspond to 
              corresponds to the ``d`` in the math above.
    sigma -   This is the variance of the field.
    """
    res = torch.zeros((xi.shape[0], 1))
    d = xi.shape[1]
    for i in range(1, d+1):
        res += xi[:, i-1].reshape(-1,1) * math.sqrt(2) * sigma * torch.sin((i - .5) * math.pi * x) / ((i - .5) * math.pi)
    return res


# In[ ]:


def trajectories_from_parameters(xi): 
    inputs = torch.hstack([x_gt.reshape(-1,1), xi.repeat(len(x_gt),1)])
    u_pred = u_trail(x_gt.reshape(-1,1), inputs, pinn)  
    return u_pred


# In[ ]:


def plot(xi_samples, xi_data, y_data, j, title):
    
    fig = plt.figure(figsize=(12,4))
    
    ax1 = fig.add_subplot(1, 2, 1)
    # corresponding input field samples
    sigma = 1.5
    ax1.plot(x_gt.reshape(-1, 1), g(x_gt.reshape(-1, 1),xi_data[0].repeat(len(x_gt),1),sigma=sigma).detach().numpy(), '-.', color = 'black', label ='ground-truth')
    field_values = torch.zeros(xi_samples.shape[0], len(x_gt))
    for i in range(xi_samples.shape[0]):
        field_values[i] = g(x_gt.reshape(-1, 1),xi_samples[i].repeat(len(x_gt),1),sigma=sigma).flatten()
        if i <= 2:
            ax1.plot(x_gt.data.numpy(), field_values[i].data.numpy(), color='green') #label=i
    mean_field = torch.mean(field_values, dim=0) 
    ax1.plot(x_gt.data.numpy(), mean_field.data.numpy(),'-.', label='mean', color='red')
    low_field = torch.quantile(field_values, 0.025, dim=0) 
    high_field = torch.quantile(field_values, 0.975, dim=0) 
    ax1.fill_between(x_gt.data.numpy(), low_field.data.numpy(), high_field.data.numpy(), alpha = 0.5, color = 'cornflowerblue', label='predictive interval')
    plt.xlabel('$x$')
    plt.ylabel(r'$log(a(x,\xi))$')
    plt.legend(loc='upper right');
    
    ax2 = fig.add_subplot(1, 2, 2)
    # corresponding output field samples
    u_pred_gt = trajectories_from_parameters(xi_data[0])
    ax2.plot(x_gt.data.numpy(), u_pred_gt.flatten().data.numpy(),'-.', label='ground-truth', color='black')
    u_preds = torch.zeros((xi_samples.shape[0], len(x_gt)))
    for i in range(xi_samples.shape[0]):
        u_preds[i] = trajectories_from_parameters(xi_samples[i]).flatten()
        if i <= 2:
            ax2.plot(x_gt.data.numpy(), u_preds[i].data.numpy(), color='green') #label=i
    ax2.plot(x_ticks, y_data[0].data.numpy(), 'x', markersize = 8, markeredgewidth=2.0, color = 'black', label="sensor data "+r'$y$')
    mean_u_pred = torch.mean(u_preds, dim=0) 
    ax2.plot(x_gt.data.numpy(), mean_u_pred.data.numpy(),'-.', label='mean', color='red')
    low_u_pred = torch.quantile(u_preds, 0.025, dim=0) 
    high_u_pred = torch.quantile(u_preds, 0.975, dim=0) 
    ax2.fill_between(x_gt.data.numpy(), low_u_pred.data.numpy(), high_u_pred.data.numpy(), alpha = 0.5, color = 'cornflowerblue', label='predictive interval')
    plt.xlabel('$x$')
    plt.ylabel(r'$u(x,\xi)$');
    plt.legend(loc='upper right');
    
    plt.suptitle(title, fontsize=12)
    
    if save==True:
        plt.savefig(results_dir +str(j)+'_plot'+'.pdf', dpi=300)
        torch.save(xi_samples, results_dir +str(j)+'_xi_samples.pt')
        #xi_samples = torch.load(results_dir +str(j)+'_xi_samples.pt')
    plt.show()
    plt.close()

def pair_plot(df, xi_data, j):
    fig, axes = plt.subplots(len(xi_data[0]), len(xi_data[0]), figsize = (14, 10), sharex="col", tight_layout=True)
    
    COLUMNS = list(df.columns)
    COLUMNS.remove('Sample type')

    for i in range(len(COLUMNS)):
        for k in range(len(COLUMNS)):
            
            # If this is the lower-triangule, add a scatterlpot for each group.
            if i > k:
                a = sns.scatterplot(data=df, x=COLUMNS[k], y=COLUMNS[i], 
                                      hue="Sample type", ax=axes[i, k], s=10, legend=False)
                a.set(xlabel=None)
                a.set(ylabel=None)
                
            # If this is the main diagonal, add kde plot
            if i == k:
                b = sns.kdeplot(data=df, x=COLUMNS[k], hue="Sample type",  common_norm=False, ax=axes[i, k])
                axes[i, k].axvline(x=xi_data[0][k], color = 'black', ls ='--')
                b.set(xlabel=None)
                b.set(ylabel=None)
                
                if k == 0:
                    sns.move_legend(b, "center right", bbox_to_anchor=(5,-1.25), title=None,frameon=True,)
                    #sns.move_legend(b, "lower center", bbox_to_anchor=(2.5, 1), ncol=3, title=None,frameon=True,)
                else:
                    axes[i, k].legend([],[], frameon=False)

            # If on the upper triangle
            if i < k:
                axes[i, k].remove()

    for i in range(len(COLUMNS)):
        k=0
        axes[i, k].set_ylabel(COLUMNS[i])
    
    for k in range(len(COLUMNS)):
        i=len(COLUMNS)-1
        axes[i, k].set_xlabel(COLUMNS[k])
    
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    if save==True:
        plt.savefig(results_dir +str(j)+'_pairplot'+'.pdf', dpi=300, bbox_inches='tight')
    # See the chart now
    plt.show()
    plt.close()


# ### Comparison of AVI vs MCMC:

# In[ ]:


import pyro
import pyro.distributions as dist_pyro
from pyro.infer import MCMC, HMC, NUTS

np.random.seed(37)
torch.manual_seed(37)
pyro.set_rng_seed(37)


# In[ ]:


n_samples = 1000 # number of posterior samples

for j in range(y_data_.shape[0]): 
    print(j)
    
    # Getting data
    xi_data, y_data = xi_data_[j].reshape(1,-1), y_data_[j].reshape(1,-1)
    print('Groundtruth:\n xi_data='+str(xi_data.data.numpy())+',\n y_data='+str(y_data.data.numpy()))
    
    print(str(j)+'_AVI')
    # AVI_Full-rank-Gaussian
    mean = (model.mu(y_data))[0, :]
    L_diag = model.L_diag(y_data)
    L_offdiag = model.L_offdiag(y_data) #or 0.01*self.L_offdiag(y)
    L = unpack_cholesky(L_diag[0,:], L_offdiag[0,:], model.xi_dim) # L matrix

    print('Estimated mean:\n'+str(mean.data.numpy()))
    print('Estimated covariance matrix:\n'+str(torch.matmul(L,L.T).data.numpy()))

    xi_samples_AVI = torch.zeros(n_samples,model.xi_dim)
    for k in range(n_samples):
        zs = torch.randn_like(mean)
        xi_samples_AVI[k] = mean + torch.matmul(L, zs)
    print(xi_samples_AVI.shape)

    plot(xi_samples_AVI, xi_data, y_data, str(j)+'_AVI', "AVI (Full-rank Gaussian)") 
    print('-'*90)

    print(str(j)+'_MCMC')
    # MCMC
    def model_MCMC(data):
        xi = pyro.sample( "input", dist_pyro.Normal(loc=torch.zeros(d_xi), scale=torch.ones(d_xi)) )
        pyro.sample("obs", dist_pyro.Normal(forward_process(xi), noise_scale), obs=data)

    hmc_kernel = HMC(model_MCMC, step_size=0.0855, num_steps=4)
    nuts_kernel= NUTS(model_MCMC, adapt_step_size=True)
    mcmc = MCMC(nuts_kernel, num_samples=int(n_samples*3), warmup_steps=300, num_chains=1)
    mcmc.run(y_data[0])
    
    thin = 3 #adjacent MCMC samples are correlated so computationally it can make sense to throw some samples out.
    xi_samples_MCMC = mcmc.get_samples()['input'][::thin,:]
    print(xi_samples_MCMC.shape)

    print(mcmc.summary())
    plot(xi_samples_MCMC, xi_data, y_data, str(j)+'_MCMC', "MCMC (NUTS)") 
    print('-'*90)    
    
    ######
    Prior_data = prior_xi_dist.sample([n_samples]).T.data.numpy()
    Prior_dict = dict() 
    for i in range(Prior_data.shape[0]):
        Prior_dict[r'$\xi_{%.0f}$' % (i+1)] = Prior_data[i] 

    AVI_data = xi_samples_AVI.T.data.numpy()
    AVI_dict = dict() 
    for i in range(AVI_data.shape[0]):
        AVI_dict[r'$\xi_{%.0f}$' % (i+1)] = AVI_data[i]  

    MCMC_data = xi_samples_MCMC.T.data.numpy()
    MCMC_dict = dict() 
    for i in range(MCMC_data.shape[0]):
        MCMC_dict[r'$\xi_{%.0f}$' % (i+1)] = MCMC_data[i]    

    df_Prior = pd.DataFrame(Prior_dict)
    df_MCMC = pd.DataFrame(MCMC_dict)
    df_AVI = pd.DataFrame(AVI_dict)
    df_Prior['Sample type'] = 'Prior'
    df_MCMC['Sample type'] = 'MCMC'
    df_AVI['Sample type'] = 'AVI'

    df = pd.concat([df_Prior, df_MCMC, df_AVI])
    df.reset_index(drop=True, inplace=True)
    pair_plot(df, xi_data, j)
    ######   
    
    print('*'*185)  


# In[ ]:


# get_ipython().system('zip -r ./results_metrics.zip ./results_metrics')


# In[ ]:


noise_scale


# In[ ]:




