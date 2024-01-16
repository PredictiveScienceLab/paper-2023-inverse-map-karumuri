#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().system('pip install pyro-ppl')


# In[2]:


# get_ipython().system('pip install scikit-learn==0.22.1')


# In[3]:


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
#get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns
sns.set_style("white")
plt.rcParams.update({'font.size': 9})

import warnings
warnings.filterwarnings("ignore")

np.random.seed(37)
torch.manual_seed(37)
random.seed(37)


# In[4]:


save = True


# ## Inverse kinematics problem
# 
# The goal is to find configurations of a multi-jointed 2D arm that end in a given position.
# The forward process takes a starting height $\xi_1$ and the three joint angles $\xi_2$, $\xi_3$, $\xi_4$, and returns the coordinate of the arm’s end point $f(\boldsymbol{\xi}) = [f_1(\boldsymbol{\xi}), f_2(\boldsymbol{\xi})]$ as 
# 
# $$f_1(\boldsymbol{\xi}) = l_1 \cos(\xi_2)+l_2 \cos(\xi_2+\xi_3)+l_3 \cos(\xi_2+\xi_3+\xi_4)$$
# $$f_2(\boldsymbol{\xi}) = \xi_1+l_1 \sin(\xi_2)+l_2 \sin(\xi_2+\xi_3)+l_3 \sin(\xi_2+\xi_3+\xi_4)$$
# 
# with arm lengths $l_1=0.5,l_2=0.5,l_3=1.0$.
# 
# Parameters $\boldsymbol{\xi}$ follow a Gaussian prior $\boldsymbol{\xi} \sim \mathcal{N} (0, \boldsymbol{\sigma}^2.\textbf{I})$ with $\boldsymbol{\sigma} = [0.25, 0.5, 0.5, 0.5]$.The inverse problem is to find the distribution $p(\boldsymbol{\xi} | \textbf{y})$
# of all arm configurations $\boldsymbol{\xi}$ that end at some observed 2D position $\textbf{y}$

# ### Formulation:
# $$\textbf{y} = \text{observed variable}$$ 
# $$\boldsymbol{\xi} = \text{latent variable}$$
# $$\boldsymbol{\xi} \in 	\mathbb{R}^4, \ \textbf{y} \in \mathbb{R}^2 $$
# $$p(\boldsymbol{\xi}|\textbf{y}) \propto p(\textbf{y}|\boldsymbol{\xi}) p(\boldsymbol{\xi})$$
# $$\text{Prior: } p(\boldsymbol{\xi}) = \mathcal{N} (0, \boldsymbol{\sigma}^2.\textbf{I})$$
# $$\text{Likelihood: } p(\textbf{y}|\boldsymbol{\xi}) =  \mathcal{N}(\textbf{y}| f(\boldsymbol{\xi}), \boldsymbol{\gamma}^2.\textbf{I}); \boldsymbol{\gamma} = [0.01, 0.01]$$

# In[5]:


xi_dim = 4 
y_dim = 2
noise_scale = torch.tensor([0.01, 0.01])
lens = torch.tensor([0.5, 0.5, 1.0]) # lengths of the kinematic links
prior_scale = torch.tensor([0.25, 0.5, 0.5, 0.5]) # prior scale

# Defining prior distribution
prior_xi_dist = dist.Normal(loc=torch.zeros(xi_dim), scale=prior_scale)


# In[6]:


def segment_points(p_, length, angle):
    p = torch.zeros(p_.shape)
    p[:,0] = p_[:,0] + length * torch.cos(angle)
    p[:,1] = p_[:,1] + length * torch.sin(angle)
    return p_, p

def forward_process(xi): # N x d_xi
    """
    Implements the forward process f(xi) and 
    and returns each of the arm’s end points as dictionary.
    """
    values = dict();
    xi = xi.reshape(-1,4)
    A = torch.stack([torch.zeros((xi.shape[0])), xi[:, 0]], axis=1)
    _, B = segment_points(A, lens[0], xi[:,1])
    _, C = segment_points(B, lens[1], xi[:,1] + xi[:,2])
    _, D = segment_points(C, lens[2], xi[:,1] + xi[:,2] + xi[:,3])
    values['A'] = A
    values['B'] = B
    values['C'] = C
    values['D'] = D
    return values;


# In[7]:


class ResNet(torch.nn.Module):
    """
    Implements the residual network
    """
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, inputs):
        return self.module(inputs) + inputs


# In[8]:


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


# In[9]:


class Amortized_VI(nn.Module):
    """
    Class that performs Amortized Variational inference.
    xi_dim: Number of dimensions in the variables to infer.
    y_dim: Number of measurement observations.
    prior_xi_dist: prior distribution defined using torch.distributions.
    f: function defining the forward process.
    noise_scale: Standard deviation of likelihood or the measurement process.
    """
    def __init__(self, xi_dim=4, y_dim=2, prior_xi_dist=None, f=forward_process, noise_scale=torch.tensor([0.01, 0.01])):
        super().__init__()
        
        self.xi_dim = xi_dim
        self.y_dim = y_dim
        self.prior_xi_dist = prior_xi_dist
        self.f = f
        self.noise_scale = noise_scale
        
        self.mu = nn.Sequential(
            nn.Linear(self.y_dim, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, self.xi_dim))
        
        self.L_diag = nn.Sequential(
            nn.Linear(self.y_dim, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, self.xi_dim),
            nn.Softplus()
        )
        
        self.L_offdiag = nn.Sequential(
            nn.Linear(self.y_dim, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, int(self.xi_dim*(self.xi_dim+1)/2)-self.xi_dim)
        )
        
    def observed_data(self, n=30):
        xi_data = self.prior_xi_dist.sample([n])  
        y_data = self.f(xi_data)['D'] + self.noise_scale * torch.randn(*self.f(xi_data)['D'].shape)

        return xi_data, y_data
    
    def forward(self, num_particles=2):
        _, y = self.observed_data(n=32)
        
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
                log_likelihood = dist.Normal(loc=self.f(xi_samples[i])['D'][0], scale=self.noise_scale).log_prob(y[j,:]).sum()
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


# In[10]:


noise_scale = torch.tensor([0.01, 0.01])
model = Amortized_VI(xi_dim=xi_dim, y_dim=y_dim, prior_xi_dist=prior_xi_dist, f=forward_process, noise_scale=noise_scale)
model.load_state_dict(torch.load('model_state_dict_AVI.pt'))


# In[11]:


results_dir = os.path.join(os.getcwd(), 'results_metrics/')
if not os.path.isdir(results_dir):
    os.makedirs(results_dir)


# In[12]:


xi_data_, y_data_ = model.observed_data(n=100)
print(xi_data_.shape, y_data_.shape)
print('Groundtruth:\n xi_data='+str(xi_data_.data.numpy())+',\n y_data='+str(y_data_.data.numpy()))

torch.save(xi_data_, results_dir +'xi_data.pt')
torch.save(y_data_, results_dir +'y_data.pt')


# In[13]:


from post_processing import *    
def plot(xi_samples, xi_data, y_data, j, title, color_code):
    fig = update_plot(xi_samples.data.numpy(), xi_data.data.numpy(), y_data.data.numpy(), lens.data.numpy(), target_label=True, color_code=color_code)
    plt.title(title, fontsize=12)

    if save==True:
        plt.savefig(results_dir +str(j)+'_plot'+'.pdf', dpi=300)
        torch.save(xi_samples, results_dir +str(j)+'_xi_samples.pt')
        #xi_samples = torch.load(results_dir +str(j)+'_xi_samples.pt')
    plt.show()
    plt.close()


def pair_plot(df, xi_data, j):
    fig, axes = plt.subplots(len(xi_data[0]), len(xi_data[0]), figsize = (12, 8), sharex="col", tight_layout=True)

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
                    sns.move_legend(b, "center right", bbox_to_anchor=(5.2,-1.25), title=None,frameon=True,)
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

    if save==True:
        plt.savefig(results_dir +str(j)+'_pairplot'+'.pdf', dpi=300)
    # See the chart now
    plt.show()
    plt.close()


# ### Comparison of AVI vs MCMC:

# In[14]:


import pyro
import pyro.distributions as dist_pyro
from pyro.infer import MCMC, HMC, NUTS

np.random.seed(37)
torch.manual_seed(37)
pyro.set_rng_seed(37)


# In[15]:


n_samples = 1000 # number of posterior samples

for j in range(y_data_.shape[0]): 
    print(j)
    
    # Getting data
    xi_data, y_data = xi_data_[j].reshape(1,-1), y_data_[j].reshape(1,-1)
    print('Groundtruth:\n xi_data='+str(xi_data.data.numpy())+',\n y_data='+str(y_data.data.numpy()))
    
    # color selection for plotting
    prior=False
    if prior:
        color_code = 4
    else:
        color_code = random.randint(0, 3)

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

    plot(xi_samples_AVI, xi_data, y_data, str(j)+'_AVI', "AVI (Full-rank Gaussian)", color_code)
    print('-'*90)

    print(str(j)+'_MCMC')
    # MCMC
    def model_MCMC(data):
        xi = pyro.sample( "input", dist_pyro.Normal(loc=torch.zeros(len(prior_scale)), scale=prior_scale) )
        pyro.sample("obs", dist_pyro.Normal(forward_process(xi)['D'].flatten(), scale=noise_scale), obs=data)

    hmc_kernel = HMC(model_MCMC, step_size=0.0855, num_steps=4)
    nuts_kernel= NUTS(model_MCMC, adapt_step_size=True)
    mcmc = MCMC(nuts_kernel, num_samples=n_samples*3, warmup_steps=300, num_chains=1)
    mcmc.run(y_data[0])
    
    thin = 3 #adjacent MCMC samples are correlated so computationally it can make sense to throw some samples out.
    xi_samples_MCMC = mcmc.get_samples()['input'][::thin,:]
    print(xi_samples_MCMC.shape)

    print(mcmc.summary())
    plot(xi_samples_MCMC, xi_data, y_data, str(j)+'_MCMC', "MCMC (NUTS)", color_code)
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


# In[16]:


#get_ipython().system('zip -r ./results_metrics.zip ./results_metrics')

