#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import torch
from torch import nn
import torch.distributions as dist
import matplotlib.pyplot as plt
import scipy.stats as st
import math
import os
import pickle
from sklearn.utils import shuffle
# get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
import pandas as pd
import arviz as az
import pyro
import pyro.distributions as dist_pyro
from pyro.infer import MCMC, HMC, NUTS
import argparse
import json

pyro.set_rng_seed(37)

np.random.seed(120)
torch.manual_seed(120)

len_flow, num_neurons = 15, 100
job = 'a'
print(len_flow, num_neurons, job)

# Specify the directory name
results_dir = os.path.join(os.getcwd(), 'results-'+str(job)) 
# Check if the directory already exists; if not, create it
if not os.path.exists(results_dir):
    os.makedirs(results_dir)


# In[3]:


xi_dim = 4 
y_dim = 2
noise_scale = torch.tensor([0.01, 0.01])
lens = torch.tensor([0.5, 0.5, 1.0]) # lengths of the kinematic links
prior_scale = torch.tensor([0.25, 0.5, 0.5, 0.5]) # prior scale

# Defining prior distribution
prior_xi_dist = dist.Normal(loc=torch.zeros(xi_dim), scale=prior_scale)


# In[4]:


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


# In[5]:


class RealNVP(nn.Module):
    def __init__(self, base_dist, split_dim, len_flow, nets, nett, neta, netb, perm):
        super(RealNVP, self).__init__()
        
        self.base_dist = base_dist
        self.split_dim = split_dim
        self.len_flow = len_flow
        self.t = torch.nn.ModuleList([nett() for _ in range(self.len_flow)])
        self.s = torch.nn.ModuleList([nets() for _ in range(self.len_flow)])
        self.a = torch.nn.ModuleList([neta() for _ in range(self.len_flow)])
        self.b = torch.nn.ModuleList([netb() for _ in range(self.len_flow)])
        self.perm = perm

    # forward transformation z, y --> xi, xi = g(z, y), log_det_J here is new/old
    def g(self, z, y):
        log_det_J = z.new_zeros(z.shape[0])
        xi = z
        for i in range(self.len_flow):
            u1, u2 = torch.split(xi, split_size_or_sections= [self.split_dim, z.shape[1]-self.split_dim], dim=-1)
            v1 = ( u1 * torch.exp(self.s[i]( torch.cat((u2, y), dim=1) )) ) + self.t[i]( torch.cat((u2, y), dim=1) )
            v2 = ( u2 * torch.exp(self.a[i]( torch.cat((v1, y), dim=1) )) ) + self.b[i]( torch.cat((v1, y), dim=1) )
            log_det_J +=  self.s[i]( torch.cat((u2, y), dim=1) ).sum(-1) + self.a[i]( torch.cat((v1, y), dim=1) ).sum(-1) 
            xi = torch.cat((v1, v2), dim=-1)
            if i < self.len_flow-1:
                xi = torch.matmul(xi, self.perm[i])   
        return xi, log_det_J
        
    def sample(self, z, y): 
        xi = self.g(z, y)[0]
        return xi


# In[6]:

class AVI(nn.Module):
    def __init__(self, xi_dim=4, y_dim=2, f=forward_process, noise_scale=0.01):
        super().__init__()
        
        self.xi_dim = xi_dim
        self.y_dim = y_dim
        self.f = f
        self.noise_scale = noise_scale
        
        self.input_dim = xi_dim
        self.split_dim = int(xi_dim/2)

        self.nets = lambda: nn.Sequential(nn.Linear(self.input_dim-self.split_dim+self.y_dim, num_neurons), nn.LeakyReLU(), nn.Linear(num_neurons, num_neurons), nn.LeakyReLU(), nn.Linear(num_neurons, self.split_dim), nn.Tanh())
        self.nett = lambda:  nn.Sequential(nn.Linear(self.input_dim-self.split_dim+self.y_dim, num_neurons), nn.LeakyReLU(), nn.Linear(num_neurons, num_neurons), nn.LeakyReLU(), nn.Linear(num_neurons, self.split_dim))

        self.neta = lambda: nn.Sequential(nn.Linear(self.split_dim+self.y_dim, num_neurons), nn.LeakyReLU(), nn.Linear(num_neurons, num_neurons), nn.LeakyReLU(), nn.Linear(num_neurons, self.input_dim-self.split_dim), nn.Tanh())
        self.netb = lambda: nn.Sequential(nn.Linear(self.split_dim+self.y_dim, num_neurons), nn.LeakyReLU(), nn.Linear(num_neurons, num_neurons), nn.LeakyReLU(), nn.Linear(num_neurons, self.input_dim-self.split_dim))

        self.len_flow = len_flow

        self.base_dist = dist.MultivariateNormal(torch.zeros(self.input_dim), torch.eye(self.input_dim))
        
        perm_file_path = os.path.join(results_dir, 'perm.pkl')
        if os.path.exists(perm_file_path):
            with open(perm_file_path, 'rb') as file:
                self.perm = pickle.load(file)
        else:
            self.perm = []
            arr =  torch.eye(self.input_dim)
            for i in range(self.len_flow-1):
                self.perm.append(shuffle(arr)) #random_state=i
            with open(perm_file_path, 'wb') as file:
                pickle.dump(self.perm, file)

        self.model = RealNVP(self.base_dist, self.split_dim, self.len_flow, self.nets, self.nett, self.neta, self.netb, self.perm)
    
    def observed_data(self, n=30):
        xi_data = dist.Normal(loc=torch.zeros(self.xi_dim), scale=prior_scale).sample([n])
        y_data = self.f(xi_data)['D'] + self.noise_scale * torch.randn(*self.f(xi_data)['D'].shape)
        return xi_data, y_data
    
    def log_joint(self, xi, y): # xi is z_k
        
        log_prior = dist.Normal(loc=torch.zeros(self.xi_dim), scale=prior_scale).log_prob(xi).sum(axis=1)
        log_likelihood = dist.Normal(loc=self.f(xi)['D'], scale=self.noise_scale).log_prob(y).sum(axis=1)
        log_joint = log_prior + log_likelihood

        return log_joint
    
    def forward(self, num_particles=2):
        
        _, y = self.observed_data(n=n_y)
        
        loss = 0
        for i in range(y.shape[0]):
            
            y_samples = y[i,:].repeat(num_particles, 1)
            z_samples = torch.autograd.Variable( self.base_dist.sample((num_particles, )) )
            z_k, log_det_J = self.model.g(z_samples, y_samples)
            loss_ = (- self.log_joint(z_k, y[i,:]) + self.base_dist.log_prob(z_samples) - log_det_J).mean()
            #or
            #loss_ = (- self.log_joint(z_k, y[i,:]) - log_det_J).mean()
            loss += loss_
    
        return loss/y.shape[0]
        


# In[7]:


m = AVI(xi_dim=xi_dim, y_dim=y_dim, f=forward_process, noise_scale=noise_scale)


# In[8]:


from post_processing import *    
def plot(xi_samples, xi_data, y_data, j, title, color_code, show=False, save=True):
    fig = update_plot(xi_samples.data.numpy(), xi_data.data.numpy(), y_data.data.numpy(), lens.data.numpy(), target_label=True, color_code=color_code)
    plt.title(title, fontsize=12)
    if save is True:
        plt.savefig(os.path.join(results_metrics_dir,str(j)+'_plot'+'.pdf'), dpi=300)
        torch.save(xi_samples, os.path.join(results_metrics_dir,str(j)+'_xi_samples.pt'))
        #xi_samples = torch.load(os.path.join(results_metrics_dir,str(j)+'_xi_samples.pt'))
    if show is True:
        plt.show()
    plt.close()


def pair_plot(df, xi_data, j, show=False, save=True):
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

    if save is True:
        plt.savefig(os.path.join(results_metrics_dir,str(j)+'_pairplot'+'.pdf'), dpi=300)
    if show is True:
        plt.show()
        
    plt.close()


# ### Comparison

# In[9]:


import pyro
import pyro.distributions as dist_pyro
from pyro.infer import MCMC, HMC, NUTS

np.random.seed(37)
torch.manual_seed(37)
pyro.set_rng_seed(37)


# In[11]:
results_metrics_dir = os.path.join(os.getcwd(), 'results_metrics/')
if not os.path.isdir(results_metrics_dir):
    os.makedirs(results_metrics_dir)


xi_data_, y_data_ = model.observed_data(n=100)
print(xi_data_.shape, y_data_.shape)
# print('Groundtruth:\n xi_data='+str(xi_data_.data.numpy())+',\n y_data='+str(y_data_.data.numpy()))
# # Getting data
# xi_data_, y_data_ = torch.load(os.path.join(results_metrics_dir,'xi_data.pt')), torch.load(os.path.join(results_metrics_dir,'y_data.pt'))
# print(xi_data_.shape, y_data_.shape)

save = True
if save is True:
    torch.save(xi_data_, os.path.join(results_metrics_dir,'xi_data.pt'))
    torch.save(y_data_, os.path.join(results_metrics_dir,'y_data.pt'))


# In[12]:


n_samples = 1000 # Number of posterior samples to visualize.

m = AVI(xi_dim=xi_dim, y_dim=y_dim, f=forward_process, noise_scale=noise_scale)
cnf_model_state_dict_loaded = torch.load(os.path.join(results_dir,'cnf_model_state_dict.pt'))
m.load_state_dict(cnf_model_state_dict_loaded)

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
        
    print(str(j)+'_MCMC')
    # MCMC
    def model_MCMC(data):
        xi = pyro.sample( "input", dist_pyro.Normal(loc=torch.zeros(len(prior_scale)), scale=prior_scale) )
        pyro.sample("obs", dist_pyro.Normal(forward_process(xi)['D'].flatten(), scale=noise_scale), obs=data)

    hmc_kernel = HMC(model_MCMC, step_size=0.0855, num_steps=4)
    nuts_kernel= NUTS(model_MCMC, adapt_step_size=True)
    mcmc = MCMC(nuts_kernel, num_samples=int(n_samples*3), warmup_steps=300, num_chains=1)
    mcmc.run(y_data[0])

    thin = 3 #adjacent MCMC samples are correlated so computationally it can make sense to throw some samples out.
    xi_samples_MCMC = mcmc.get_samples()['input'][::thin,:]
    print(xi_samples_MCMC.shape)
    # xi_samples_MCMC = torch.load(os.path.join(results_metrics_dir, str(j)+'_MCMC'+'_xi_samples.pt'))
    # print(xi_samples_MCMC.shape)

    # print(mcmc.summary())
    plot(xi_samples_MCMC, xi_data, y_data, str(j)+'_MCMC', "MCMC (NUTS)", color_code, show=True, save=True)
    print('-'*90) 
    
    print(str(j)+'_AVI')
    # AVI_Standard-Normalizing-Flows-RealNVP
    z_samples = m.base_dist.sample((n_samples, ))
    y_samples = y_data[0,:].repeat(n_samples, 1)
    xi_samples_AVI = m.model.sample(z_samples, y_samples)
    print(xi_samples_AVI.shape)
    
    plot(xi_samples_AVI, xi_data, y_data, str(j)+'_AVI', "AVI (Conditional Normalizing Flows)", color_code, show=True, save=True)
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
    pair_plot(df, xi_data, j, show=True, save=True)
    ######  

    print('*'*230)  


# In[ ]:





# In[ ]:





# In[ ]:




