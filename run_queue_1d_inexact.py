import sys
sys.path.append("./src")

from utils import sample_Queue
from plot_functions import SeabornFig2Grid, plot_gauss_4d, plot_posterior_marginals_mmd_vs_mabc
from credible_interval import joint_eti_region
import NPL
import models
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.gridspec as gridspec
import time
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import jax
import seaborn as sns
import warnings
warnings.filterwarnings("ignore") 

# check if the GPU is online
print(jax.__version__)
print(jax.devices())

service_shapes = [1, 0.8, 0.6, 0.4, 0.2]
print("Starting experiments for different values of service shape a...")

# Set model 
model_name = 'queue_1d' 
theta_opts = np.array([1.2, 1.46515538, 1.84569039, 2.49797974, 3.9806058])
m = 30 # number of samples, n in the paper
p = 1   # number of unknown parameters
B = 1000 # number of bootstrap iterations 
R = 100 # number of independent runs
alpha = 0.05
n = 500 # m in the paper

# Before running: 
# 1) Set paths 
# 2) Indicate whether you want a fresh dataset or to load existing one 
# 3) experiments are run for multiple runs - index which run you want plots for
index = 0
for service_shape in service_shapes:
    print(f"Running experiments for a={service_shape}...")
    # Paths
    data_path = f"./data/queueing_model_1d_inexact/a={service_shape}/"
    results_path = f"./results/queueing_model_1d_inexact/a={service_shape}/"
    plots_path = f"./plots/queueing_model_1d_inexact/a={service_shape}/"

    # Set to True to generate and save fresh datasets or False to load saved datasets
    sample_data_bool = True

    # Index which run you want to plot results for
    r = 0 

    # Set model 
    model = models.QueueModel_1d(m, shape=service_shape)

    # Create the directory if it doesn't exist
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    theta_opt = theta_opts[index]    

    if sample_data_bool:
        for j in range(R):
            X = sample_Queue(np.array([1.2, 1]), n, service_shape_parameter=service_shape, constant_seed=j)
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
    npl_mmd_path = os.path.join(results_path, 'NPL_MMD')
    os.makedirs(npl_mmd_path, exist_ok=True)
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)

    l = -1 # bandwidth picked with median heuristic
    summary_stats = np.zeros((R, 4)) # collect mean, median, mode, st.dev for each bootstrap sample
    coverage_results = 0 
    eti_width = np.zeros(R)
    eti_height = np.zeros(R)
    for j in range(R):
        print("-----Run ", j)
        X = datasets[j,:].reshape((n,1))
        npl = NPL.npl(X,B,m,p,l, model = model, model_name = model_name)
        t0 = time.time()
        npl.draw_samples()
        sample = npl.sample
        # Compute the ETI region
        eti_region = np.quantile(sample, [alpha/2, 1-alpha/2])

        # Check if theta_star is inside the HPD region
        if np.any((theta_opt >= eti_region[0]) & (theta_opt <= eti_region[1])):
            coverage_results += 1
        t1 = time.time()
        total = t1-t0
        times.append(total)
        eti_lower = eti_region[0]
        eti_upper = eti_region[1]
        eti_width[j] = eti_upper - eti_lower
        print(total)
        summary_stats[j, 0] = np.mean(sample)
        summary_stats[j, 1] = np.std(sample)
        summary_stats[j, 2] = np.median(sample)
        summary_stats[j, 3] = stats.mode(sample)[0]
        np.savetxt(results_path+'NPL_MMD/thetas_mmd_run_{}.txt'.format(j), sample)
        print(f"  Run {j+1}/{R} for a={service_shape} completed.")


    # save results
    np.save(os.path.join(results_path, 'NPL_MMD', 'summary_stats.npy'), summary_stats)
    np.savetxt(results_path+'NPL_MMD/cpu_times.txt', times)  
    np.savetxt(os.path.join(results_path, 'NPL_MMD', 'eti_widths.txt'), eti_width)
    np.savetxt(os.path.join(results_path, 'NPL_MMD', 'eti_heights.txt'), eti_height)
    # Save the estimated coverage to a text file
    with open(os.path.join(results_path, 'NPL_MMD', 'eti_coverage.txt'), 'w') as f:
        f.write(f"Estimated ETI Coverage: {coverage_results / R}\n")
    with open(os.path.join(results_path, 'NPL_MMD', 'cpu_times_avg.txt'), 'w') as f:
        f.write(f"Estimated mean CPU time: {np.mean(times)}\n")
    with open(os.path.join(results_path, 'NPL_MMD', 'eti_width_avg.txt'), 'w') as f:
        f.write(f"mean CI width for service rate: {np.mean(eti_width)}\n")

    print(f"All runs for n={n} completed.")
    for r in range(min(5, R)):
        # Reshape results
        thetas_mmd = np.zeros((B))
        for j in range(p):
            sample = np.loadtxt(os.path.join(results_path, f'NPL_MMD/thetas_mmd_run_{r}.txt'))
            if p > 1:
                thetas_mmd[j, :] = sample[:, j]
            else:
                thetas_mmd = sample


        # Assuming thetas_mmd is a 2D array with shape (num_samples, 2)
        service_rate = thetas_mmd

        # True parameter values
        true_service_rate = theta_opt

        # Calculate posterior statistics
        service_rate_mean = np.mean(service_rate)
        service_rate_median = np.median(service_rate)
        service_rate_025 = np.quantile(service_rate, 0.025)
        service_rate_975 = np.quantile(service_rate, 0.975)

        # Create a figure and axis
        plt.figure(figsize=(8, 6))

        # Plot the density plot for service rate
        sns.kdeplot(service_rate, shade=True, label='Posterior')
        plt.axvline(true_service_rate, color='r', linestyle='--', label='Optimal value')
        plt.axvline(service_rate_mean, color='g', linestyle='--', label='Posterior mean')
        plt.axvline(service_rate_median, color='y', linestyle='--', label='Posterior median')
        plt.axvline(service_rate_025, color='b', linestyle='--', label='2.5% quantile')
        plt.axvline(service_rate_975, color='b', linestyle='--', label='97.5% quantile')
        plt.title('Posterior of service rate')
        plt.xlabel('Service rate')
        plt.ylabel('Density')
        plt.legend()

        # Save the plot as an image file
        plt.savefig(os.path.join(plots_path, f'posterior_densities_run_{r}.png'))
        plt.close()
    index += 1
print("Experiments completed for all values of n.")