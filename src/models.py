#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 17:19:23 2021

"""
from utils import boxmuller, normals, k_comp
import math
import numpy as np
from jax import grad, lax, jit, vmap, random
import jax.numpy as jnp
from jax.scipy import stats as jstats
#from jax.ops import index, index_update

class gauss_model():
    
    def __init__(self, m, d, s):
        self.m = m  # number of points sampled from P_\theta at each optim. iteration
        self.d = d  # data dim
        self.s = s  # standard deviation of the model
        
    def generator(self, unif, theta):

        unif[unif==0] = np.nextafter(0, 1)

        # if d is odd, add one dimension
        if self.d % 2 != 0:
            dim = self.d + 1
        else:
            dim = self.d

        # create standard normal samples
        u = np.zeros((self.m,dim))
        for i in np.arange(0,dim,2):
            u[:,i:(i+2)] = boxmuller(unif[:,i],unif[:,(i+1)])

        # if d is odd, drop one dimension
        if self.d % 2 != 0:
            u = np.delete(u,-1,1)

       # generate samples
        x = theta + u*self.s

        return x

    # gradient of the generator
    def grad_generator(self,theta):
        return np.broadcast_to(np.expand_dims(np.eye(theta.shape[0]),axis=2),(theta.shape[0],theta.shape[0],self.m))
    
    def sample(self,theta):

        # odd number of parameters
        if self.d % 2 != 0: 
            unif = np.random.rand(self.m,self.d+1)
        # even number of parameters
        else: 
            unif = np.random.rand(self.m,self.d)
        
        # use generator  
        x = self.generator(unif,theta)

        return x


class QueueModel():
    def __init__(self, m, shape=None):
        self.m = m  # number of points sampled from P_\theta at each optim. iteration
        self.shape = shape
    
    def generator(self, theta, service_shape_parameter=1, arrival_shape_parameter=0.5,
                  sample_period=20, burn_in_period=10, constant_seed=None):
        service_rate, arrival_rate = theta
        if self.shape is not None:
            service_shape_parameter = self.shape
        if constant_seed is not None:
            key = random.PRNGKey(constant_seed)
        else:
            key = random.PRNGKey(np.random.randint(2**32))

        recurrsionvar = jnp.zeros(self.m)
        totaltime = jnp.zeros(self.m)

        service_times, arrival_times = self.pull_simulation_drivers(service_shape_parameter, arrival_shape_parameter,
                                                                    sample_period + burn_in_period, key)

        service_ratio = service_shape_parameter / service_rate
        arrival_ratio = arrival_shape_parameter / arrival_rate

        for i in range(sample_period + burn_in_period):
            # Simulate from G/G/1
            recurrsionvar = jnp.maximum(recurrsionvar + service_ratio * service_times[:, i]
                                        - arrival_ratio * arrival_times[:, i], 0.0)

            if i >= burn_in_period:
                totaltime += recurrsionvar

        return totaltime / sample_period

    def pull_simulation_drivers(self, service_shape_parameter, arrival_shape_parameter, total_samples, key=None):
        # Simulate the Gamma arrival and service times
        service_shape_paras = jnp.full((self.m, total_samples), service_shape_parameter)
        arrival_shape_paras = jnp.full((self.m, total_samples), arrival_shape_parameter)
        
        key, subkey = random.split(key)
        service_times = random.gamma(key, service_shape_paras, None)
        arrival_times = random.gamma(subkey, arrival_shape_paras, None)

        return service_times, arrival_times
    
    def pull_simulation_drivers_single(self, service_shape_parameter, arrival_shape_parameter, total_samples, key=None):
        # Simulate the Gamma arrival and service times
        service_shape_paras = jnp.full((1, total_samples), service_shape_parameter)
        arrival_shape_paras = jnp.full((1, total_samples), arrival_shape_parameter)
        
        key, subkey = random.split(key)
        service_times = random.gamma(key, service_shape_paras, None)
        arrival_times = random.gamma(subkey, arrival_shape_paras, None)

        return service_times, arrival_times
    
    def sample(self, theta, constant_seed=None):
        # theta: (service_rate, arrival_rate)
        x = self.generator(theta, constant_seed=constant_seed)
        return x
    
    def generator_single(self, theta, uvals, service_shape_parameter=1, arrival_shape_parameter=1,
                  sample_period=20, burn_in_period=10, constant_seed=None):
        service_rate, arrival_rate = theta
        if self.shape is not None:
            service_shape_parameter = self.shape
        if constant_seed is not None:
            key = random.PRNGKey(constant_seed)
        else:
            key = random.PRNGKey(np.random.randint(2**32))

        recurrsionvar = jnp.zeros(1)
        totaltime = jnp.zeros(1)

        service_times, arrival_times = self.pull_simulation_drivers_single(service_shape_parameter, arrival_shape_parameter,
                                                                    sample_period + burn_in_period, key)

        service_ratio = service_shape_parameter / service_rate
        arrival_ratio = arrival_shape_parameter / arrival_rate

        for i in range(sample_period + burn_in_period):
            # Simulate from G/G/1
            recurrsionvar = jnp.maximum(recurrsionvar + service_ratio * service_times[:, i]
                                        - arrival_ratio * arrival_times[:, i], 0.0)

            if i >= burn_in_period:
                totaltime += recurrsionvar

        return totaltime / sample_period
    
    def grad_generator(self, theta):
        gradient = grad(self.generator_single, argnums=0)
        grad_ = vmap(jit(gradient), in_axes=(None,0), out_axes=1)(theta)
        return jnp.reshape(grad_, (1,len(theta),self.m))
    
class QueueModel_1d():
    def __init__(self, m, shape=None):
        self.m = m  # number of points sampled from P_\theta at each optim. iteration
        self.shape = shape
    
    def generator(self, theta, service_shape_parameter=1, arrival_shape_parameter=0.5,
                  sample_period=20, burn_in_period=10, constant_seed=None):
        service_rate = theta
        arrival_rate = 1
        if self.shape is not None:
            service_shape_parameter = self.shape
        if constant_seed is not None:
            key = random.PRNGKey(constant_seed)
        else:
            key = random.PRNGKey(np.random.randint(2**32))

        recurrsionvar = jnp.zeros(self.m)
        totaltime = jnp.zeros(self.m)

        service_times, arrival_times = self.pull_simulation_drivers(service_shape_parameter, arrival_shape_parameter,
                                                                    sample_period + burn_in_period, key)

        service_ratio = service_shape_parameter / service_rate
        arrival_ratio = arrival_shape_parameter / arrival_rate

        for i in range(sample_period + burn_in_period):
            # Simulate from G/G/1
            recurrsionvar = jnp.maximum(recurrsionvar + service_ratio * service_times[:, i]
                                        - arrival_ratio * arrival_times[:, i], 0.0)

            if i >= burn_in_period:
                totaltime += recurrsionvar

        return totaltime / sample_period

    def pull_simulation_drivers(self, service_shape_parameter, arrival_shape_parameter, total_samples, key=None):
        # Simulate the Gamma arrival and service times
        service_shape_paras = jnp.full((self.m, total_samples), service_shape_parameter)
        arrival_shape_paras = jnp.full((self.m, total_samples), arrival_shape_parameter)
        
        key, subkey = random.split(key)
        service_times = random.gamma(key, service_shape_paras, None)
        arrival_times = random.gamma(subkey, arrival_shape_paras, None)

        return service_times, arrival_times
    
    def pull_simulation_drivers_single(self, service_shape_parameter, arrival_shape_parameter, total_samples, key=None):
        # Simulate the Gamma arrival and service times
        service_shape_paras = jnp.full((1, total_samples), service_shape_parameter)
        arrival_shape_paras = jnp.full((1, total_samples), arrival_shape_parameter)
        
        key, subkey = random.split(key)
        service_times = random.gamma(key, service_shape_paras, None)
        arrival_times = random.gamma(subkey, arrival_shape_paras, None)

        return service_times, arrival_times
    
    def sample(self, theta, constant_seed=None):
        # theta: (service_rate, arrival_rate)
        x = self.generator(theta, constant_seed=constant_seed)
        return x
    
    def generator_single(self, theta, uvals, service_shape_parameter=1, arrival_shape_parameter=1,
                  sample_period=20, burn_in_period=10, constant_seed=None):
        service_rate = theta
        arrival_rate = 1
        if self.shape is not None:
            service_shape_parameter = self.shape
        if constant_seed is not None:
            key = random.PRNGKey(constant_seed)
        else:
            key = random.PRNGKey(np.random.randint(2**32))

        recurrsionvar = jnp.zeros(1)
        totaltime = jnp.zeros(1)

        service_times, arrival_times = self.pull_simulation_drivers_single(service_shape_parameter, arrival_shape_parameter,
                                                                    sample_period + burn_in_period, key)

        service_ratio = service_shape_parameter / service_rate
        arrival_ratio = arrival_shape_parameter / arrival_rate

        for i in range(sample_period + burn_in_period):
            # Simulate from G/G/1
            recurrsionvar = jnp.maximum(recurrsionvar + service_ratio * service_times[:, i]
                                        - arrival_ratio * arrival_times[:, i], 0.0)

            if i >= burn_in_period:
                totaltime += recurrsionvar

        return totaltime / sample_period
    
    def grad_generator(self, theta):
        gradient = grad(self.generator_single, argnums=0)
        grad_ = vmap(jit(gradient), in_axes=(None,0), out_axes=1)(theta)
        return jnp.reshape(grad_, (1,len(theta),self.m))
