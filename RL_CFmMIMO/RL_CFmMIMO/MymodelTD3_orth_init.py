# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 14:39:30 2024

@author: user
"""
###### initialised relu weights with kaiming he and tanh with xavier
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F 
import ipdb
from torch.distributions import Normal

    ## weights by default initialised from uniform distribution of U(-sqrt(k),sqrt(k))
    #where k =1/(input feature size)
    ## but above problematic for relu, tanh and sigmoid because gradients can vanish or exlode so use
    ## xavier and kaiming initialization respectively
def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

class Actor(nn.Module):
    def __init__(self,number_of_states,number_of_actions,hidden1=256,hidden2=256, init_wt=3e-3):
        super(Actor,self).__init__()
        
        self.fc1=nn.Linear(number_of_states, hidden1)
        self.fc2=nn.Linear(hidden1, hidden2)
        self.fc3=nn.Linear(hidden2,number_of_actions)
        #self.fc4=nn.Linear(hidden2,number_of_actions)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        #self.initialise_weights(init_wt)
        
        self.LOG_STD_MAX=2 
        self.LOG_STD_MIN=-20
        
        
        
    def initialise_weights(self,init_wt):
        
        nn.init.orthogonal_(self.fc1.weight.data)
        nn.init.orthogonal_(self.fc2.weight.data)
        nn.init.orthogonal_(self.fc3.weight.data)
        #nn.init.orthogonal_(self.fc4.weight.data)
        
    def forward(self,xs):
        
        out=self.fc1(xs)
        out=self.relu(out)
        out=self.fc2(out)
        out=self.relu(out)
        out=self.fc3(out)
        out=self.tanh(out)
       
        return out
		
		
        
class Critic(nn.Module):
    def __init__(self, number_of_states, number_of_actions, hidden1=256, hidden2=256, init_wt=3e-3):
        super(Critic, self).__init__()
        
        #Critic 1
        self.fc1 = nn.Linear(number_of_states+number_of_actions, hidden1)
        self.fc2 = nn.Linear(hidden1,hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        self.relu = nn.ReLU()
        
        #Critic 2
        self.fc4 = nn.Linear(number_of_states+number_of_actions, hidden1)
        self.fc5 = nn.Linear(hidden1,hidden2)
        self.fc6 = nn.Linear(hidden2, 1)
        self.relu = nn.ReLU()
        self.init_weights(init_wt)
        
    def init_weights(self, init_wt):
        
        nn.init.orthogonal_(self.fc1.weight.data)
        nn.init.orthogonal_(self.fc2.weight.data)
        nn.init.orthogonal_(self.fc3.weight.data)
        nn.init.orthogonal_(self.fc4.weight.data)
        nn.init.orthogonal_(self.fc5.weight.data)
        nn.init.orthogonal_(self.fc6.weight.data)
        
       
    
    def forward(self, xc):
        #x1,x2, a = xc
        x1,a = xc
        #out = self.fc1(torch.cat([x1,x2],1))
        q1 = self.fc1(torch.cat([x1,a],1))
        q1 = self.relu(q1)
        q1 = self.fc2(q1)
        q1 = self.relu(q1)
        q1 = self.fc3(q1)
        
        
        q2 = self.fc4(torch.cat([x1,a],1))
        q2 = self.relu(q2)
        q2 = self.fc5(q2)
        q2 = self.relu(q2)
        q2 = self.fc6(q2)
        
        return q1,q2
     
    def Q1(self,xc):
        x1,a = xc
        
        q1 = self.fc1(torch.cat([x1,a],1))
        q1 = self.relu(q1)
        q1 = self.fc2(q1)
        q1 = self.relu(q1)
        q1 = self.fc3(q1)
        return q1
    
