# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 16:47:21 2024

@author: user
"""
#https://github.com/sfujim/TD3/blob/master/main.py

import numpy as np
import ipdb
import numba
import copy
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import Normal
import torch.nn.functional as F 
#import copy
from MymodelTD3_orth_init import (Actor, Critic)
#from Mymemory import SequentialMemory
#from random_process import OrnsteinUhlenbeckProcess
from util import *
#import action_space
import random
#import matplotlib.pyplot as plt
#import time
criterion = nn.MSELoss()
class WolpertingerAgent(object):
    def __init__(self,nb_states,nb_actions,args):#protoaction, no. ues per AP, number of neighbors to search 
        if args.seed > 0:
            self.seed(args.seed) 
            set_numba_seed(args.seed)
        self.noise_clip=args.noise_clip
        self.sigma=args.sigma
        self.sigma_target=args.sigma_target
        self.K=args.UEnumber
        self.M=args.APnumber
        self.tau_p=args.tau_p
        
        self.nb_states = nb_states
        self.nb_actions= nb_actions
        
        self.update_iter=0
        self.update_freq=args.update_freq
        self.adaptive_alpha=args.adaptive_alpha
        self.alpha=args.alpha
        
        self.dvc=args.dvc
        self.rmsize=args.rmsize
       
        
               #self.env=args.env
        #check action space items-- 
        #print(self.action_space._Space__space)
        # Create Actor and Critic Network
        net_cfg = {
            'hidden1':args.hidden1, 
            'hidden2':args.hidden2, 
            'init_wt':args.init_wt,
            
        }
        self.actor = Actor(self.nb_states, self.nb_actions, **net_cfg).to(self.dvc)
        self.actor_target = Actor(self.nb_states, self.nb_actions, **net_cfg).to(self.dvc)
        self.actor_optim  = Adam(self.actor.parameters(), lr=args.prate)

        self.critic = Critic(self.nb_states, self.nb_actions, **net_cfg).to(self.dvc)
        self.critic_target = Critic(self.nb_states, self.nb_actions, **net_cfg).to(self.dvc)
        self.critic_optim  = Adam(self.critic.parameters(), lr=args.rate)
        
        # Make sure target is with the same weight
        hard_update(self.critic_target, self.critic)
        hard_update(self.actor_target, self.actor)
        #Create replay buffer
        
        self.replay_buffer = Replay_Buffer(self.nb_states, self.nb_actions, max_size=self.rmsize, dvc=self.dvc)
       
        
        
        # Hyper-parameters
        self.batch_size = args.bsize
        self.tau = args.tau
        self.discount = args.discount
        self.depsilon = 1.0 / args.epsilon

        # 
        self.epsilon = 1.0
        self.s_t = None # Most recent state
        self.a_t = None # Most recent action
        self.is_training = True
        
        
        
        
        # 
        if USE_CUDA: self.cuda()
        
    
  
    
                


    def update_policy(self,gainovernoise,steps,APMask,pilotIndex,CandidateMask,pre_train=False):
        # Sample batch
        self.update_iter+=1
        
       
        
            
        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = self.replay_buffer.sample(self.batch_size)
         
            
        
        
        
        # Prepare for the target q batch
        with torch.no_grad(): 
            ## we compute targets with next actions from policy. no target actor here
            #next_actions,next_log_pi_as=self.actor(to_tensor(next_state_batch[:,:,0]),deterministic=False,with_logprob=True)
            target_noise=np.clip((np.random.randn(self.batch_size,self.nb_actions)*self.sigma_target),-self.noise_clip,self.noise_clip)
            next_actions=(self.actor_target(next_state_batch).cpu()+\
                                 target_noise).clamp(-1.,1.).type(torch.FloatTensor)
            #next_discrete_actions=self.get_target_actions(next_state_batch,next_actions,SA_mode=self.SA_mode)
            next_discrete_actions=self.get_target_actions(next_state_batch.cpu(),next_actions.cpu(),gainovernoise,APMask,pilotIndex,CandidateMask)
           
            #normalized_next_discrete_actions=self.import_point(next_discrete_actions)
            next_q1_values,next_q2_values = self.critic_target([next_state_batch,
            next_discrete_actions],
                )
            
            ##min of q1 and q2
            next_q_values=torch.min(next_q1_values,next_q2_values)
            
            # target_q_batch = to_tensor(reward_batch) + \
            # self.discount*to_tensor(terminal_batch.astype(np.float32))*(next_q_values-self.alpha*next_log_pi_as)
            
            target_q_batch = reward_batch + \
            self.discount*terminal_batch*(next_q_values)
                

        ############### Critic update######################
        self.critic.zero_grad()
        ## get current Q values
        
        q1_batch,q2_batch = self.critic([ state_batch,action_batch ])
        
        
        ########
        value_loss = criterion(q1_batch, target_q_batch)+criterion(q2_batch,target_q_batch)
        
        value_loss.backward()
        
        self.critic_optim.step()
        
       
        for params in self.critic.parameters(): params.requires_grad=False
        ###Update Actor ##########################
        
        #Do delayed updates 
        if self.update_iter%self.update_freq==0:
            
            # Actor update
            
            a=self.actor(state_batch)
            q_policy=self.critic.Q1([ state_batch,a])
            
            
            
            policy_loss=-q_policy.mean()
           
            
            
            self.actor.zero_grad()
            policy_loss.backward()
            self.actor_optim.step()
            
            
        
            # Target update
            soft_update(self.actor_target, self.actor, self.tau)
            soft_update(self.critic_target, self.critic, self.tau)
            
        ### unfreeze critic 
        for params in self.critic.parameters():params.requires_grad=True
        
        
       
       
    def eval(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()

    def cuda(self):
        self.actor.cuda()
        #self.actor_target.cuda()
        self.critic.cuda()
        self.critic_target.cuda()

    def observe(self,r_t, s_t1, done,terminated):
        if self.is_training:
            self.memory.append(self.s_t, np.array(self.a_t,dtype=np.float32), r_t, done,terminated)
            #print(self.s_t)
            #print(done)
            #ipdb.set_trace()
            self.s_t = s_t1
            
            
   
## double check if need to use normalised actions
    def random_action(self,K,gainovernoise,APMask,pilotIndex,CandidateMask):
       
        action=np.random.randint(0,2,self.nb_actions)
       
        full_action=(action*np.array(CandidateMask)+np.array(APMask)).clip(0,1)
        bin_action=torch.from_numpy(full_action.reshape(self.M,K))
       
        
        
        self.a_t = full_action
        return full_action
 
    
    def get_target_actions(self,s_t1,next_protoactions,gainovernoise,APMask,pilotIndex,CandidateMask):
        
       
        next_base_actions=copy.deepcopy(next_protoactions)
        next_base_actions[next_base_actions<0]=0
        next_base_actions[next_base_actions>0]=1
        next_base_actions=next_base_actions.int()
        next_discrete_actions_np=np.zeros_like(next_base_actions)
        ###compute valid base action
        #apply action mask first. since its (-1,1),simply add and clip to (0,1)
        
        next_base_actions_full=torch.zeros((self.batch_size,self.M*self.K),dtype=torch.int32)
        #next_base_actions_full[:,ActionIndices]=next_base_actions
        next_discrete_actions_np_full=np.zeros_like(next_base_actions_full)
        next_base_actions_full=(next_base_actions*CandidateMask+APMask).clamp(min=0,max=1)
        bin_actions=next_base_actions_full.reshape(-1,self.M,self.K)
        
        
        
        
       
        return (next_base_actions_full.to(torch.int64)).to(self.dvc)
       
       
    
    
       
       
       
    
    def get_base_action_value(self,s_t,base_action):
        
        #s_tnew=np.expand_dims(np.array(s_t).transpose(),axis=0)
        s_tnew=np.array(s_t)
        #critic_input = [to_tensor(s_tnew[:,:,0]), to_tensor(self.import_point(base_action))]
        critic_input = [to_tensor(s_tnew), to_tensor(base_action)]
        baseQ_value=self.critic.Q1(critic_input)
        
        return(baseQ_value)
        
 
    def select_action(self, s_t, gainovernoise,APMask,pilotIndex,CandidateMask):
        
        
        proto_action= self.TD3_select_action(s_t)
        ###map to 0 if <0 and 1 if >0
        base_action=copy.deepcopy(proto_action)
        base_action[base_action>0]=1
        base_action[base_action<0]=0
        base_action=base_action.astype(np.int32)
        
        #apply action mask first. since its (-1,1),simply add and clip to (0,1)
        #full_action=np.zeros(self.M*self.K,).astype(int)
        #full_action[ActionIndices]=base_action
        
        full_action=(base_action*np.array(CandidateMask)+np.array(APMask)).clip(0,1)
        ###map to valid action 
        bin_action=torch.from_numpy(full_action.reshape(self.M,self.K))
       
        
        
        self.a_t=full_action
        self.action_val=0. ##not necessary. just didnt want to change other files
        
         
        return self.a_t.astype(int),self.action_val,proto_action
       

    def TD3_select_action(self,s_t):
        # only used when interact with the env
        #s_tnew=np.expand_dims(np.array(s_t).transpose(),axis=0)
        s_tnew=np.array(s_t)
        with torch.no_grad():
            #action,_=self.actor(to_tensor(s_tnew[:,:,0]),deterministic,with_logprob=False)
            action=self.actor(to_tensor(s_tnew)).cpu().detach().numpy()
            #####use gaussian clipped noise
            
            noise=self.sigma*np.random.randn(action.size)
            
            action += self.is_training*np.clip(noise,-self.noise_clip,self.noise_clip)
            action = np.clip(action, -1., 1.) # clip between 0&1 because use tanh
        
        #action=action.detach().numpy() ## convert to np array
        return action
        
       
		

    def reset(self, obs):
        self.s_t = obs
        #self.random_process.reset_states()

    def load_weights(self, output):
        if output is None: return

        self.actor.load_state_dict(
            torch.load('{}/actor.pkl'.format(output),map_location=torch.device('cpu'))
        )

        self.critic.load_state_dict(
            torch.load('{}/critic.pkl'.format(output),map_location=torch.device('cpu'))
        )


    def save_model(self,output):
        torch.save(
            self.actor.state_dict(),
            '{}/actor.pkl'.format(output)
        )
        torch.save(
            self.critic.state_dict(),
            '{}/critic.pkl'.format(output)
        )

    def seed(self,s):
        np.random.seed(s)
        random.seed(s)
        torch.manual_seed(s)
        if USE_CUDA:
            torch.cuda.manual_seed(s)
    
    
    
class Replay_Buffer():
        def __init__(self,state_dim,action_dim,max_size,dvc):
            self.max_size=max_size
            self.dvc=dvc
            self.ptr=0
            self.size=0
            
            self.s=torch.zeros((max_size,state_dim),dtype=torch.float,device=self.dvc)
            self.a=torch.zeros((max_size,action_dim),dtype=torch.float,device=self.dvc)
            self.r=torch.zeros((max_size,1),dtype=torch.float,device=self.dvc)
            self.s_next=torch.zeros((max_size,state_dim),dtype=torch.float,device=self.dvc)
            self.done=torch.zeros((max_size,1),dtype=torch.bool,device=self.dvc)
            self.terminated=torch.zeros((max_size,1),dtype=torch.float,device=self.dvc)
            
        def add(self,s,a,r,s_next,done,terminated):
            
            self.s[self.ptr]=torch.from_numpy(s).to(self.dvc)
            self.a[self.ptr]=torch.from_numpy(a).to(self.dvc) #a numpy array?
            self.r[self.ptr]=torch.from_numpy(r)
            self.s_next[self.ptr]=torch.from_numpy(s_next).to(self.dvc)
            
            self.done[self.ptr]=torch.tensor(done)
            self.terminated[self.ptr]=(torch.tensor(0.) if terminated else torch.tensor(1.)).to(self.dvc)
            
            self.ptr=(self.ptr+1)%self.max_size
            self.size=min(self.size+1,self.max_size)
            
        def sample(self,batch_size):
            ind=torch.randint(0,self.size,device=self.dvc,size=(batch_size,))
            return self.s[ind], self.a[ind], self.r[ind],self.s_next[ind], self.terminated[ind]




@numba.jit
def set_numba_seed(seed):
    np.random.seed(seed)    
    

    
    
