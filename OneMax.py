# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

"""
Created on Sun Oct 10 18:06:55 2021

@author: Lingxun
"""
import six
import sys
sys.modules['sklearn.externals.six'] = six
import mlrose as ml
import numpy as np
import matplotlib.pyplot as plt
import time

seed =1

def plot_fitness_curve(data,title):
    plt.figure()
    plt.title(title)
    plt.xlabel("iterations")
    plt.ylabel("fitness")
    legend = ["Randomize hill climbing","Simulated annealing","Genetic algorithm","MIMIC"]
    for i in range(len(data)):
        plt.semilogx(data[i])
    plt.legend(legend)
    






'''
===============================================================================
Four peaks problem
===============================================================================
'''


fitness_OneMax = ml.OneMax()

wall_clock_rhc = {}
wall_clock_sa = {}
wall_clock_ga = {}
wall_clock_mimic = {}

def solve_opt(input_size):
    
    problem_OneMax = ml.DiscreteOpt(input_size,fitness_OneMax)

    '''
    Randomize hill climbing
    '''
    start = time.time()
    best_state_rhc, best_fitness_rhc,fitness_curve_rhc = ml.random_hill_climb(problem_OneMax, 
    max_attempts = 10, max_iters =10000,curve=True,random_state=seed,restarts=5)
    end = time.time()
    
    print("\nRandomized hill climbing")
    # print('best state = ',best_state_rhc)
    print('Number of iterations = ',fitness_curve_rhc.size)
    print('best fitness =',best_fitness_rhc)
    print('wall clock time (sec) =',end-start)
    wall_clock_rhc[input_size] = end-start
    
    """
    Simulated annealing
    """
    start = time.time()
    # Define decay schedule
    schedule = ml.ExpDecay(init_temp=100,exp_const=0.005)
    best_state_sa, best_fitness_sa,fitness_curve_sa = ml.simulated_annealing(problem_OneMax, schedule = schedule,
    max_attempts = 20, max_iters =10000,curve=True,random_state=seed)
    end = time.time()
    
    print("\nSimulated annealing")
    # print('best state = ',best_state_sa)
    print('Number of iterations = ',fitness_curve_sa.size)
    print('best fitness =',best_fitness_sa)
    print('wall clock time (sec) =',end-start)
    
    wall_clock_sa[input_size] = end-start
    """
    Genetic Algorithm
    """
    start = time.time()
    best_state_ga, best_fitness_ga,fitness_curve_ga = ml.genetic_alg(problem_OneMax,  pop_size=200, mutation_prob=0.1,
    max_attempts = 10, max_iters =1000,curve=True,random_state=seed)
    end = time.time()
    
    print("\nGenetic Algorithm")
    # print('best state = ',best_state_ga)
    print('Number of iterations = ',fitness_curve_ga.size)
    print('best fitness =',best_fitness_ga)
    print('wall clock time (sec) =',end-start)
    wall_clock_ga[input_size] = end-start
    
    
    """
    MIMIC
    """
    start = time.time()
    best_state_mimic, best_fitness_mimic,fitness_curve_mimic = ml.mimic(problem_OneMax,  pop_size=200, keep_pct=0.2,
    max_attempts = 10, max_iters =1000,curve=True,random_state=seed)
    end = time.time()
    
    print("\nMIMIC")
    # print('best state = ',best_state_mimic)
    print('Number of iterations = ',fitness_curve_mimic.size)
    print('best fitness =',best_fitness_mimic)
    print('wall clock time (sec) =',end-start)
    
    wall_clock_mimic[input_size] = end-start


    data = [fitness_curve_rhc,fitness_curve_sa,fitness_curve_ga,fitness_curve_mimic ]
    
    plot_fitness_curve(data,'One Max problem with '+str(input_size)+' inputs')

inputs = [50,75,100]

for i in inputs:
    print("\nInput size = ",i)
    solve_opt(i)
    
plt.figure
plt.semilogy(wall_clock_rhc.keys(),wall_clock_rhc.values(),'o-')
plt.semilogy(wall_clock_rhc.keys(),wall_clock_sa.values(),'x-')
plt.semilogy(wall_clock_rhc.keys(),wall_clock_ga.values(),'+-')
plt.semilogy(wall_clock_rhc.keys(),wall_clock_mimic.values(),'s-')
plt.xlabel('Input size')
plt.ylabel('Wall clock time (sec)')
plt.legend(['RHC','SA','GA','MIMIC'])
plt.title('Computational time')





#SA is the best (one of the best objective, just slightly lower than MIMIC but much faster.Objective is much better than GA)

