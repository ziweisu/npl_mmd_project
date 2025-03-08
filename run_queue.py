import sys
sys.path.append("./src")

from utils import sample_Queue
from plot_functions import SeabornFig2Grid, plot_gauss_4d, plot_posterior_marginals_mmd_vs_mabc
import NPL
import models
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.gridspec as gridspec
import time
import matplotlib.pyplot as plt
import os
import jax
import seaborn as sns

# Before running: 
# 1) Set paths 
# 2) Indicate whether you want a fresh dataset or to load existing one 
# 3) experiments are run for multiple runs - index which run you want plots for

# Paths
data_path = "./data/queueing_model/"
results_path = "./results/queueing_model/"
plots_path = "./plots/queueing_model/"

# Set to True to generate and save fresh datasets or False to load saved datasets
sample_data_bool = True

# Index which run you want to plot results for
r = 0 

# check if the GPU is online
print(jax.__version__)
print(jax.devices())

# Set model 
model_name = 'queue' 
n = 1000 # number of samples from the target system, m in the paper
theta_star = np.array([1, 1])
m = 100 # number of samples, n in the paper
p = 2   # number of unknown parameters
B = 1000 # number of bootstrap iterations 
model = models.QueueModel(m)
R = 1 # number of independent runs

# Create the directory if it doesn't exist
if not os.path.exists(data_path):
    os.makedirs(data_path)

if sample_data_bool:
    for j in range(R):
        X = sample_Queue(theta_star, n, constant_seed=j)
        file_path = os.path.join(data_path, f'run_{j}')
        np.savetxt(file_path, X)
        
# Load data
datasets = np.zeros((R,n))
for j in range(R):
    X = np.loadtxt(data_path+'run_{}'.format(j))
    datasets[j,:] = X
times = []

# Obtain and save results 
if not os.path.exists(results_path):
    os.makedirs(results_path)
else:
    if not os.path.exists("./results/queueing_model/NPL_MMD/"):
        os.makedirs("./results/queueing_model/NPL_MMD/")
if not os.path.exists(plots_path):
    os.makedirs(plots_path)

l = -1 # bandwidth picked with median heuristic
summary_stats = np.zeros((R, p, 4)) # collect mean, median, mode, st.dev for each bootstrap sample
for j in range(R):
    print("-----Run ", j)
    X =datasets[j,:].reshape((n,1))
    npl = NPL.npl(X,B,m,p,l, model = model, model_name = model_name)
    t0 = time.time()
    npl.draw_samples()
    t1 = time.time()
    total = t1-t0
    times.append(total)
    print(total)
    sample = npl.sample
    summary_stats[j,:,0] = np.mean(sample, axis=0)
    summary_stats[j,:,1] = np.std(sample, axis=0)
    summary_stats[j,:,2] = np.median(sample, axis=0)
    summary_stats[j,:,3] = stats.mode(sample, axis=0)[0]
    np.savetxt(results_path+'NPL_MMD/thetas_mmd_run_{}.txt'.format(j), sample)

np.save(os.path.join(results_path, 'NPL_MMD', 'summary_stats.npy'), summary_stats)
np.savetxt(results_path+'NPL_MMD/cpu_times.txt', times)  

# Reshape results
r = 0 # which run you want to have the posterior plotted out
thetas_mmd = np.zeros((p,B))
for j in range(p):
    sample = np.loadtxt(results_path+'NPL_MMD/thetas_mmd_run_{}.txt'.format(r))
    if p>1:
        thetas_mmd[j,:] = sample[:,j]
    else:
        thetas_mmd[j,:] = sample
       
# Assuming thetas_mmd is a 2D array with shape (num_samples, 2)
service_rate = thetas_mmd[0, :]
arrival_rate = thetas_mmd[1, :]

# True parameter values
true_service_rate = 1
true_arrival_rate = 1

# Calculate posterior statistics
service_rate_mean = np.mean(service_rate)
service_rate_median = np.median(service_rate)
service_rate_025 = np.quantile(service_rate, 0.025)
service_rate_975 = np.quantile(service_rate, 0.975)

arrival_rate_mean = np.mean(arrival_rate)
arrival_rate_median = np.median(arrival_rate)
arrival_rate_025 = np.quantile(arrival_rate, 0.025)
arrival_rate_975 = np.quantile(arrival_rate, 0.975)

# Create a figure and axis
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Plot the density plot for service rate
sns.kdeplot(service_rate, shade=True, ax=ax1, label='Posterior')
ax1.axvline(true_service_rate, color='r', linestyle='--', label='True value')
ax1.axvline(service_rate_mean, color='g', linestyle='--', label='Posterior mean')
ax1.axvline(service_rate_median, color='y', linestyle='--', label='Posterior median')
ax1.axvline(service_rate_025, color='b', linestyle='--', label='2.5% quantile')
ax1.axvline(service_rate_975, color='b', linestyle='--', label='97.5% quantile')
ax1.set_title('Posterior of service rate')
ax1.set_xlabel('Service rate')
ax1.legend()

# Plot the density plot for arrival rate
sns.kdeplot(arrival_rate, shade=True, ax=ax2, label='Posterior')
ax2.axvline(true_arrival_rate, color='r', linestyle='--', label='True value')
ax2.axvline(arrival_rate_mean, color='g', linestyle='--', label='Posterior mean')
ax2.axvline(arrival_rate_median, color='y', linestyle='--', label='Posterior median')
ax2.axvline(arrival_rate_025, color='b', linestyle='--', label='2.5% quantile')
ax2.axvline(arrival_rate_975, color='b', linestyle='--', label='97.5% quantile')
ax2.set_title('Posterior of arrival rate')
ax2.set_xlabel('Arrival rate')
ax2.legend()

# Adjust spacing between subplots
plt.tight_layout()

# Show the plot
plt.show()
