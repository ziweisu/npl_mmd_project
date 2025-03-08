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
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import jax
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# check if the GPU is online
print(jax.__version__)
print(jax.devices())

n_values = [10, 50, 100, 500, 1000]

print("Starting experiments for different values of n...")

# Set model 
model_name = 'queue' 
theta_star = np.array([1, 1])
m = 30 # number of samples, n in the paper
p = 2   # number of unknown parameters
B = 1000 # number of bootstrap iterations 
R = 100 # number of independent runs
n = 500

# Before running: 
# 1) Set paths 
# 2) Indicate whether you want a fresh dataset or to load existing one 
# 3) experiments are run for multiple runs - index which run you want plots for

for n in n_values:
    print(f"Running experiments for n={n}...")
    # Paths
    data_path = f"./data/queueing_model_2d_nonstat/n={n}/"
    results_path = f"./results/queueing_model_2d_nonstat/n={n}/"
    plots_path = f"./plots/queueing_model_2d_nonstat/n={n}/"

    # Set to True to generate and save fresh datasets or False to load saved datasets
    sample_data_bool = True

    # Index which run you want to plot results for
    r = 0 

    # Set model 
    model = models.QueueModel(m)

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
    npl_mmd_path = os.path.join(results_path, 'NPL_MMD')
    os.makedirs(npl_mmd_path, exist_ok=True)
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)

    l = -1 # bandwidth picked with median heuristic
    summary_stats = np.zeros((R, p, 4)) # collect mean, median, mode, st.dev for each bootstrap sample
    coverage_results = 0 #
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
        eti_region = joint_eti_region(sample, alpha=0.05)
    
        # Check if theta_star is inside the HPD region
        if np.any((theta_star >= eti_region.min(axis=0)) & (theta_star <= eti_region.max(axis=0))):
            coverage_results += 1

        t1 = time.time()
        total = t1-t0
        times.append(total)
        eti_lower = eti_region[0]
        eti_upper = eti_region[1]
        eti_width[j] = eti_upper[0] - eti_lower[0]
        eti_height[j] = eti_upper[1] - eti_lower[1]
        print(total)
        if j <= 4:
            # Plot the joint ETI region
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(sample[:, 0], sample[:, 1], alpha=0.1)

            # Create a rectangle patch for the joint ETI region
            eti_rect = plt.Rectangle((eti_lower[0], eti_lower[1]), eti_width[j], eti_height[j],
                         linewidth=1, edgecolor='r', facecolor='none', linestyle='--', label='Joint ETI Region')
            ax.add_patch(eti_rect)

            ax.plot(theta_star[0], theta_star[1], 'go', label='True Parameter')
            ax.set_xlabel('Service Rate')
            ax.set_ylabel('Arrival Rate')
            ax.set_title('Joint ETI Region')
            ax.legend()

            # Save the joint ETI region plot as an image file
            plt.savefig(os.path.join(plots_path, 'joint_eti_region_{}.png'.format(j)))
            plt.close()
        summary_stats[j,:,0] = np.mean(sample, axis=0)
        summary_stats[j,:,1] = np.std(sample, axis=0)
        summary_stats[j,:,2] = np.median(sample, axis=0)
        summary_stats[j,:,3] = stats.mode(sample, axis=0)[0]
        np.savetxt(results_path+'NPL_MMD/thetas_mmd_run_{}.txt'.format(j), sample)
        print(f"  Run {j+1}/{R} for n={n} completed.")


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
    with open(os.path.join(results_path, 'NPL_MMD', 'eti_height_avg.txt'), 'w') as f:
        f.write(f"mean CI width for arrival rate: {np.mean(eti_height)}\n")
    
    print(f"All runs for n={n} completed.")
    for r in range(min(5, R)):
        # Reshape results
        thetas_mmd = np.zeros((p, B))
        for j in range(p):
            sample = np.loadtxt(os.path.join(results_path, f'NPL_MMD/thetas_mmd_run_{r}.txt'))
            if p > 1:
                thetas_mmd[j, :] = sample[:, j]
            else:
                thetas_mmd[j, :] = sample

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

        # Save the plot as an image file
        plt.savefig(os.path.join(plots_path, f'posterior_densities_run_{r}.png'))
        plt.close()
print("Experiments completed for all values of n.")
