# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 19:17:36 2021

@author: Lingxun
"""

import numpy as np
import math
import pandas as pd

num_init_sample = 5000

#Underlying function
def feasibility(x):
    if sum(x) > 3.2:
        return False
    elif x[0]*x[1] < 0.2:
        return False
    elif math.exp(x[0]*x[1]*x[2]) > 2*x[3]:
        return False
    elif x[2]*x[3] > x[1]:
        return False
    else:
        return True

np.random.seed(2)
data = {}
label = {}
truelabel = 0
falselabel = 0
#Sample the dataset
for i in range(num_init_sample):
    x = np.random.random_sample(size = (4))
    fea = feasibility(x)
    if fea == True: 
        data[i] =x
        label[i] = 1
        truelabel+=1
    else:
        if np.random.random() > 0.85:
            data[i] = x
            label[i] = 0
            falselabel += 1
            
print("# of true labels =",truelabel)
print("# of false labels =",falselabel)


#Create features (list of edges) and lables
df_data = pd.DataFrame.from_dict(data,orient='index')
df_label = pd.DataFrame.from_dict(label,orient='index')


import warnings
import time
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import six
import sys
sys.modules['sklearn.externals.six'] = six
import mlrose as ml
# from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

y =df_label.values.ravel()
X = df_data.values

# X_train, X_test, y_train, y_test = train_test_split(X, y, \
# test_size = 0.2, random_state = 1)
    
    
# scaler = MinMaxScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)
# X = scaler.transform(X)
    
def buildNN(algorithm,learning_rate,max_attempts,iterations): 
    print(algorithm)
    NN_model = ml.NeuralNetwork(hidden_nodes = [32,16,8], activation = 'relu', \
    algorithm = algorithm, max_iters = iterations,bias = True, is_classifier = True, learning_rate =learning_rate, \
    mutation_prob = 0.1, pop_size = 500,
    clip_max = 1,
    early_stopping = True, max_attempts =max_attempts, random_state = 1,curve=True)
        
    NN_model.fit(X_train,y_train)
        
    y_train_pred = NN_model.predict(X_train)
    y_train_accuracy = accuracy_score(y_train, y_train_pred)
    # print('train accuracy = ',y_train_accuracy)
        
    y_test_pred = NN_model.predict(X_test)
    y_test_accuracy = accuracy_score(y_test, y_test_pred)
    # print('test accuracy = ',y_test_accuracy)
    
    global kf_train_acc,kf_test_acc
    kf_train_acc += y_train_accuracy
    kf_test_acc += y_test_accuracy
    
    return NN_model
    
    
algos = ["gradient_descent","random_hill_climb","simulated_annealing","genetic_alg"]
lr = [0.00001,0.5,0.5,0.001]
ma = [10,100,10,10]
it = [10000,10000,5000,5000]

for i in range(4):
    kf = KFold(n_splits=4)
    kf_train_acc = 0
    kf_test_acc = 0
    kf_loss = 0
    start = time.time()
    for train_index, test_index in kf.split(X):
        
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        model = buildNN(algos[i],lr[i],ma[i],it[i])
        kf_loss += model.loss
        
    end = time.time()
    print('solution time = ',end-start)
    print('cross-val train accuracy',kf_train_acc/4)
    print('cross-val test accuracy',kf_test_acc/4)
    print('cross-val loss',kf_loss/4)
    plt.semilogx(-1*model.fitness_curve)
    plt.legend(algos)
    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.title("Training loss vs. iterations")
    


# """
# Randomized Hill Climbing Tunning
# """

# NN_model = ml.NeuralNetwork(hidden_nodes = [32,16,8], activation = 'relu', \
#     algorithm = "random_hill_climb", max_iters = 10000,bias = True, is_classifier = True, learning_rate =0.9, \
#     mutation_prob = 0.5, pop_size = 500,
#     clip_max = 1,
#     early_stopping = True, max_attempts =10, random_state = 1,curve=True)
        
# NN_model.fit(X_train,y_train)
        
# y_train_pred = NN_model.predict(X_train)
# y_train_accuracy = accuracy_score(y_train, y_train_pred)
# print('train accuracy = ',y_train_accuracy)
        
# y_test_pred = NN_model.predict(X_test)
# y_test_accuracy = accuracy_score(y_test, y_test_pred)
# print('test accuracy = ',y_test_accuracy)



# NN_model = ml.NeuralNetwork(hidden_nodes = [32,16,8], activation = 'relu', \
#     algorithm = "random_hill_climb", max_iters = 10000,bias = True, is_classifier = True, learning_rate =0.9, \
#     mutation_prob = 0.5, pop_size = 500,
#     clip_max = 1,
#     early_stopping = True, max_attempts =50, random_state = 1,curve=True)
        
# NN_model.fit(X_train,y_train)
        
# y_train_pred = NN_model.predict(X_train)
# y_train_accuracy = accuracy_score(y_train, y_train_pred)
# print('train accuracy = ',y_train_accuracy)
        
# y_test_pred = NN_model.predict(X_test)
# y_test_accuracy = accuracy_score(y_test, y_test_pred)
# print('test accuracy = ',y_test_accuracy)




# """
# Genetic Algorithm Tunning
# """


# NN_model = ml.NeuralNetwork(hidden_nodes = [32,16,8], activation = 'relu', \
#     algorithm = "genetic_alg", max_iters = 5000,bias = True, is_classifier = True, learning_rate =0.9, \
#     mutation_prob = 0.5, pop_size = 500,
#     clip_max = 1,
#     early_stopping = True, max_attempts =100, random_state = 2,curve=True)
        
# NN_model.fit(X_train,y_train)
        
# y_train_pred = NN_model.predict(X_train)
# y_train_accuracy = accuracy_score(y_train, y_train_pred)
# print('train accuracy = ',y_train_accuracy)
        
# y_test_pred = NN_model.predict(X_test)
# y_test_accuracy = accuracy_score(y_test, y_test_pred)
# print('test accuracy = ',y_test_accuracy)



# NN_model = ml.NeuralNetwork(hidden_nodes = [32,16,8], activation = 'relu', \
#     algorithm = "genetic_alg", max_iters = 5000,bias = True, is_classifier = True, learning_rate =0.9, \
#     mutation_prob = 0.9, pop_size = 5000,
#     clip_max = 1,
#     early_stopping = True, max_attempts =10, random_state = 1,curve=True)
        
# NN_model.fit(X_train,y_train)
        
# y_train_pred = NN_model.predict(X_train)
# y_train_accuracy = accuracy_score(y_train, y_train_pred)
# print('train accuracy = ',y_train_accuracy)
        
# y_test_pred = NN_model.predict(X_test)
# y_test_accuracy = accuracy_score(y_test, y_test_pred)
# print('test accuracy = ',y_test_accuracy)





# NN_model = ml.NeuralNetwork(hidden_nodes = [32,16,8], activation = 'relu', \
#     algorithm = "simulated_annealing", max_iters = 5000,bias = True, is_classifier = True, learning_rate =0.5, \
#     mutation_prob = 0.1, pop_size = 2000,
#     clip_max = 1,
#     early_stopping = True, max_attempts =100, random_state = 1,curve=True)
        
# NN_model.fit(X_train,y_train)
        
# y_train_pred = NN_model.predict(X_train)
# y_train_accuracy = accuracy_score(y_train, y_train_pred)
# print('train accuracy = ',y_train_accuracy)
        
# y_test_pred = NN_model.predict(X_test)
# y_test_accuracy = accuracy_score(y_test, y_test_pred)
# print('test accuracy = ',y_test_accuracy)

# print(NN_model.fitness_curve)




  