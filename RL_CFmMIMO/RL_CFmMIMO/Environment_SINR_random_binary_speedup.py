# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 16:47:22 2024

@author: user
"""
import ipdb
from scipy import linalg
from typing import Optional
import random
import numpy as np
import torch
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import BoundedTensorSpec, CompositeSpec, UnboundedContinuousTensorSpec
from torchrl.envs import (
    CatTensors,
    EnvBase,
    Transform,
    TransformedEnv,
    UnsqueezeTransform,
)
import numba
#import copy
import time

#### environment was constructed based on the tutorial at https://pytorch.org/tutorials/advanced/pendulum.html
## default inputs


#place APs and UEs


## generate random setup with AP and UE locations, and compute spatial corrleation
#matrices and LSF gain, assign pilots and return initial allocation as best APs
    
def setup(M,K,dcorr,delta,sigma,antennas,PNlog,ASD_azimuth,tau_p,hap,hue):
    
    #place APs
    AP=np.empty([M,9],dtype=np.complex128)
    AP0=np.empty(M,dtype=np.complex128)
    AP0.real=np.random.uniform(-0.5,0.5,M)
    AP0.imag=np.random.uniform(-0.5,0.5,M) # initial positions
    AP[:,0]=AP0
    AP[:,1]=AP0+1
    AP[:,2]=AP0-1
    AP[:,3]=AP0+1j
    AP[:,4]=AP0-1j
    AP[:,5]=AP0+1+1j
    AP[:,6]=AP0+1-1j
    AP[:,7]=AP0-1-1j
    AP[:,8]=AP0-1+1j
    #compute AP-AP distance
    APdistance=np.zeros([M,M],dtype=np.float64)
    for i in range(M):
        distAP=np.abs(AP[i,0]-AP)
        APdistance[i,:]=np.min(distAP,axis=1)
    APcorrelation=2**(-APdistance/dcorr)
    
    #place UEs
    UE=np.empty([K,9],dtype=np.complex128)
    UE0=np.empty(K,dtype=np.complex128)
    UE0.real=np.random.uniform(-0.5,0.5,K)
    UE0.imag=np.random.uniform(-0.5,0.5,K) # initial positions
    UE[:,0]=UE0
    UE[:,1]=UE0+1
    UE[:,2]=UE0-1
    UE[:,3]=UE0+1j
    UE[:,4]=UE0-1j
    UE[:,5]=UE0+1+1j
    UE[:,6]=UE0+1-1j
    UE[:,7]=UE0-1-1j
    UE[:,8]=UE0-1+1j
    
    #compute UE-UEdistance
    UEdistance=np.zeros([K,K],dtype=np.float64)
    for i in range(K):
        distUE=np.abs(UE[i,0]-UE)
        UEdistance[i,:]=np.min(distUE,axis=1)
    UEcorrelation=2**(-UEdistance/dcorr)
  #  for i in range(9):
      #  plt.scatter(AP[:,i].real, AP[:,i].imag)
    #compute shadow fading
    APshadow=np.sqrt(delta)*sigma*np.matmul(linalg.sqrtm(APcorrelation),np.random.normal(0,1,M))
    UEshadow=np.sqrt(1-delta)*sigma*np.matmul(linalg.sqrtm(UEcorrelation),np.random.normal(0,1,K))
    Z_shadow=np.zeros([M,K])
    for m in range(M):
        for k in range(K):
            Z_shadow[m,k]=APshadow[m]+UEshadow[k]
    # compute large scale fading coefficient beta
    distance=np.zeros([M,K]) #AP-UE distance
    pos=np.zeros([M,K]) # arg of minimum distance AP for spatial correlation
    
    for k in range(K):
            dist=np.abs(UE[k,0]-AP)
            distance[:,k]=np.min(dist,axis=1)
            pos[:,k]=np.argmin(dist,axis=1)
    #ipdb.set_trace()
    distance=np.sqrt((distance*1000)**2+np.array((hap-hue)**2))
    BETAdB=-30.5-36.7*np.log10(distance)+Z_shadow
    gainovernoisedB=BETAdB-np.array(PNlog) ## cnat subract np array and tensor
    gainovernoise=10**(gainovernoisedB/10).astype(np.float64) #gain over noise  linear
    #uncorrelated rayleigh 
    #for k in range(K):
        #for m in range(M):
            #R[:,:,m,k]=beta[m,k]*np.eye(antennas)
    
    # apply spatial correlation
    R=np.zeros((antennas,antennas,M,K),dtype='complex')
    for k in range(K):
        for m in range(M):
            #compute nominal angle between ue k and AP m
            
            #azimuth_angle=np.angle(UE[k,0]-AP[m,int(pos[m,k])])
            #elevation_angle=np.arcsin((hap-hue)/distance[int(m),int(k)])
            #import functionRlocalscattering
            #R[:,:,m,k]=gainovernoise[m,k]*functionRlocalscattering.R(antennas,azimuth_angle,ASD_azimuth)
            R[:,:,m,k]=gainovernoise[m,k]*np.eye(antennas)

    #compute DCC serving APs
    #get master AP
    ServingAPsDCC=np.zeros((M,K),dtype=np.int64)
    masterAPs=np.zeros(K,dtype=int)
    pilotIndex=np.zeros(K,dtype=int)
    
    for k in range(K):
        master=np.argmax(gainovernoisedB[:,k])
        #print(master)
        masterAPs[k]=master
        ServingAPsDCC[master,k]=1
        #assign orthogonalpilots to first tau_p Ues
        if k<tau_p:
            pilotIndex[k]=k
        else:
            pilotinterference=np.zeros(tau_p)
            for t in range(tau_p):
                pilotinterference[t]=np.sum(gainovernoise[master,np.where\
                                                          (pilotIndex[0:k]==t)])
            
            bestpilot=np.argmin(pilotinterference)
            pilotIndex[k]=bestpilot
            #print(pilotinterference)
            #print(bestpilot)
    
    
    return(UE,AP,R,distance,gainovernoisedB,gainovernoise,pilotIndex,ServingAPsDCC)

### compute channel estimates
def ChannelEstimates(M,antennas,channels,K,p,tau_p,pilotIndex,R):
        H=np.random.normal(0,1,(M*antennas,channels,K))+1j*np.random.normal\
            (0,1,(M*antennas,channels,K))
        # aply corelation matrices to channels
        for m in range(M):
            for k in range(K):
                Rsqrt=linalg.sqrtm(R[:,:,m,k])
                H[antennas*m:antennas*(m+1),:,k]=np.sqrt(0.5)*np.matmul\
                    (Rsqrt,H[antennas*m:antennas*(m+1),:,k])
        eyeN=np.eye(antennas)
        #generate noise
        Np=np.sqrt(0.5)*(np.random.normal(0,1,(antennas,channels,M,tau_p))\
                         +1j*np.random.normal(0,1,(antennas,channels,M,tau_p)))
        Hhat=np.zeros((M*antennas,channels,K),dtype=np.complex128)
        B=np.zeros(R.shape,dtype='complex')
        C=np.zeros(R.shape,dtype='complex')
        for m in range(M):
            for t in range(tau_p):
                #compute processed signal for all UEs using pilot t
                yp=np.sqrt(p)*tau_p*np.sum(H[antennas*m:antennas*(m+1),:,t==pilotIndex],\
                                           axis=2)+np.sqrt(tau_p)*Np[:,:,m,t]
                PsiInv=(p*tau_p*np.sum(R[:,:,m,t==pilotIndex],axis=2)+eyeN)
                Pilot_t_UEs=np.where(pilotIndex==t)[0] #uses which use pilot t
                for k in Pilot_t_UEs:
                    RPsi=R[:,:,m,k]/PsiInv
                    #MMSE estimate
                    Hhat[antennas*m:antennas*(m+1),:,k]=np.sqrt(p)*np.matmul(RPsi,yp)
                    #spatial correlation matrix of estimate
                    B[:,:,m,k]=p*tau_p*np.matmul(RPsi,R[:,:,m,k])
                    #spatial correlation matrix of the estimation error
                    C[:,:,m,k]=R[:,:,m,k]-B[:,:,m,k]
        return(H,Hhat,B,C)
### convert to tensor

### Generate AP and UE channels and beta, matrices

def compute_rates(M,K,APs,gainovernoise,Pd,tau_p,tau_c,B,R,pilotIndex):
    
    Power=np.zeros((M,K),dtype=float)
    for m in range(M):
        servedUEs=np.where(APs[m,:]==1)[0]
        denominator=np.sum(np.sqrt(gainovernoise[m,servedUEs]))
        for k in servedUEs:
            Power[m,k]=Pd*np.sqrt(gainovernoise[m,k])/denominator  
#compute SEs
    signal_MR=np.zeros(K,dtype='float')
    interf_MR=np.zeros(K,dtype='float')
    cont_MR=np.zeros((K,K),dtype='float')
    pre_logfactor=1-tau_p/tau_c
    for m in range(M):
        servedUEs=np.where(APs[m,:]==1)[0]
        for k in servedUEs:
            signal_MR[k]=signal_MR[k]+np.sqrt(Power[m,k]*np.real(np.trace(B[:,:,m,k])))
            for i in range(K):
                interf_MR[i]=interf_MR[i]+Power[m,k]*\
                    np.real(np.trace(np.matmul(B[:,:,m,k],R[:,:,m,i])))/np.real(np.trace(B[:,:,m,k]))
                if pilotIndex[k]==pilotIndex[i]:
                    cont_MR[i,k]=cont_MR[i,k]+np.sqrt(Power[m,k])*\
                        np.real(np.trace(np.matmul((np.matmul(B[:,:,m,k],np.linalg.inv(R[:,:,m,k]))),R[:,:,m,i])))\
                            /np.sqrt(np.real(np.trace(B[:,:,m,k])))
    SINR=(np.abs(signal_MR)**2)/(interf_MR+np.sum(np.abs(cont_MR)**2,axis=1)-np.abs(signal_MR)**2+1)
    SE_MR=pre_logfactor*np.real(np.log2(1+SINR)) 
    return(SE_MR,SINR,Power)

@numba.jit
def compute_rates1(M,K,APs,gainovernoise,Pd,tau_p,tau_c,B,R,pilotIndex):
    
    Power=np.zeros((M,K),dtype=np.float32)
    for m in range(M):
        servedUEs=np.where(APs[m,:]==1)[0]
        denominator=np.sum(np.sqrt(gainovernoise[m,servedUEs]))
        for k in servedUEs:
            Power[m,k]=Pd*np.sqrt(gainovernoise[m,k])/denominator  
#compute SEs
    signal_MR=np.zeros(K,dtype=np.float32)
    interf_MR=np.zeros(K,dtype=np.float32)
    cont_MR=np.zeros((K,K),dtype=np.float32)
    pre_logfactor=1-tau_p/tau_c
    for m in range(M):
        servedUEs=np.where(APs[m,:]==1)[0]
        for k in servedUEs:
            signal_MR[k]=signal_MR[k]+np.sqrt(Power[m,k]*np.real(np.trace(B[:,:,m,k])))
            for i in range(K):
                interf_MR[i]=interf_MR[i]+Power[m,k]*\
                    np.real(np.trace(np.dot(B[:,:,m,k],R[:,:,m,i])))/np.real(np.trace(B[:,:,m,k]))
                if pilotIndex[k]==pilotIndex[i]:
                    cont_MR[i,k]=cont_MR[i,k]+np.sqrt(Power[m,k])*\
                        np.real(np.trace(np.dot((np.dot(B[:,:,m,k],np.linalg.inv(R[:,:,m,k]))),R[:,:,m,i])))\
                            /np.sqrt(np.real(np.trace(B[:,:,m,k])))
    SINR=(np.abs(signal_MR)**2)/(interf_MR+np.sum(np.abs(cont_MR)**2,axis=1)-np.abs(signal_MR)**2+1)
    SE_MR=pre_logfactor*np.real(np.log2(1+SINR)) 
    return(SE_MR,SINR,Power)
@numba.jit
def ComputeEE1(ServingAPs,UE_SEs,UE_power,M,Bandwidth,antennas):
    Pbackhaul=np.zeros(M,dtype=np.float32)
    PAPs=np.zeros(M,dtype=np.float32)
    for m in range(M):
        
        EEservedUEs=np.argwhere(ServingAPs[m,:]==1)
        
        
        Pbackhaul[m]=0.825+(Bandwidth/1e9)*np.sum(UE_SEs[EEservedUEs[:,0]])*0.25
        PAPs[m]=(1/0.4)*np.sum(UE_power[m,EEservedUEs[:,0]])/1000+0.2*antennas
        
    TotalPower=np.sum(PAPs)+np.sum(Pbackhaul)
    EE=((Bandwidth*np.sum(UE_SEs))/TotalPower)/1e6
    return EE
def BetaAPs(M,K,gainovernoise): 
    ServingAPsBeta=np.zeros((M,K)) 
    for k in range(K): 
        
        gain=gainovernoise[:,k]
        sorted_gain=np.sort(gain)[::-1]## sort in descending order
        current_gain=np.zeros(M)
        for m in range(M): 
            current_gain[m]=np.sum(sorted_gain[:m+1])/np.sum(sorted_gain)
        #ipdb.set_trace()
        stopindex=np.where(current_gain>=0.95)[0][0]
        #ipdb.set_trace()
        for id in range(stopindex+1):
            AP=np.where(gain==sorted_gain[id])[0]
            ServingAPsBeta[AP,k]=1
    
    return(ServingAPsBeta)

def BetaMatching(APquota,UEquota,gainovernoise,M,K):
    UEpref=np.argsort(-gainovernoise,0)
    APpref=np.argsort(-gainovernoise,1)
    ServingAPsBetascalable=np.zeros((M,K),dtype=np.int32)
    proposal_round=0
    while np.sum(UEpref)>M*K:
        
        availaps=UEpref>-1
        
        for i in range(K):
           
            
            #ue k proposes to ap 
            if UEpref[availaps[:,i],i].size==0:
                continue
            
            ap=UEpref[availaps[:,i],i][0]
            
            # if ap.numel()==0:
            #     continue
            if np.sum(ServingAPsBetascalable[ap,:])<APquota:
                ServingAPsBetascalable[ap,i]=1 
                UEpref[proposal_round,i]=-1
            
            else:
                servedues=np.argwhere(ServingAPsBetascalable[ap,:]==1)
                prefk=np.argwhere(APpref[ap,:]==i)
                prefother=np.argwhere(APpref[ap,:]==servedues)[:,1]
                prefswap=np.max(prefother)
                
                if prefk<prefswap:
                    swapue=servedues[np.argmax(prefother)]
                    ServingAPsBetascalable[ap,i]=1 
                    ServingAPsBetascalable[ap,swapue]=0
                    UEpref[proposal_round,i]=-1
                    #swap 
                else:
                    #dontswap and clear from pref list
                    UEpref[proposal_round,i]=-1
        #clear pref list of full ues
        current_aps=np.sum(ServingAPsBetascalable,0)
        full_ues=np.argwhere(current_aps==UEquota)
        UEpref[:,full_ues]=-1
        proposal_round+=1
       
    return(ServingAPsBetascalable)
##ServingAPsDCC
#compute DCC serving APs
#get master AP
def DCCAPs(M,K,tau_p,gainovernoise,pilotIndex): 
   
    ServingAPsDCC=np.zeros((M,K),dtype=int)
    pilots=pilotIndex
    for k in range(K): 
        master=np.argmax(gainovernoise[:,k])
        ServingAPsDCC[master,k]=1
#each AP serves UE with strongest channel on each pilot
    for m in range(M): 
        for t in range(tau_p): 
            pilotUEs=np.where(pilots==t) #this is a tuple
            pilotUEs1=pilotUEs[0] #array of indices
        #print(pilotUEs1)
            UEindex=np.argmax(gainovernoise[m,pilotUEs1])
        #print(pilotUEs)
            ServingAPsDCC[m,pilotUEs1[UEindex]]=1
    return(ServingAPsDCC)

def ServingAPsRow_to_real(self,ServingAPs):
    #returns o/p of sum_(from i=1 to m) sum_(from j=1 to n)*a_ij*2^((i-1)n+j-1)
    #m= rows, n= columns
    #note indices 0 is the last element as, thats why flipped
    #converting flattened o/p to binary
    M,K=ServingAPs.shape
    indices=np.flip(np.arange(K))
    real=np.sum(ServingAPs*2**indices,axis=1)
    return(real)
def ComputeEE(ServingAPs,UE_SEs,UE_power,M,Bandwidth,antennas):
    Pbackhaul=np.zeros(M)
    PAPs=np.zeros(M)
    for m in range(M):
        
        EEservedUEs=np.argwhere(ServingAPs[m,:]==1)
        if not torch.is_tensor(EEservedUEs):
            EEservedUEs=torch.from_numpy(EEservedUEs)
        Pbackhaul[m]=0.825+(Bandwidth/1e9)*torch.sum(UE_SEs[EEservedUEs[:,0]])*0.25
        PAPs[m]=(1/0.4)*torch.sum(UE_power[m,EEservedUEs])/1000+0.2*antennas
        
    TotalPower=torch.sum(PAPs)+torch.sum(Pbackhaul)
    EE=((Bandwidth*torch.sum(UE_SEs))/TotalPower)/1e6
    return EE
    
trackaction=[]
## moved below to reset function otw doesnt reset to 0 at end of episode so will rarely terminate
#trackreward=np.zeros(20)
#steps=0
#maxreward=0 # we want to keep track og max reward seen so far in episode
### define environment step . takes i/p state = serving APs , action, and beta matrix and outputs
#reward, done, next serving APs
def _step(self,tensordict):
        done=False
        ServingAPs=tensordict["ServingAPs"]
        SINR=tensordict["SINR"]
        PreviousAPs=ServingAPs
        gainovernoise=tensordict["gainovernoise"]
        K=tensordict["params","K"] #UE number
        M=tensordict["params","M"] #AP number
        tau_p=tensordict["params","tau_p"] #number of pilots
        tau_c=tensordict["params","tau_c"] #coherence length
        B=tensordict["B"] #correlation matrix of channel estimate
        R=tensordict["R"] #correlation matrix 
        pilotIndex=tensordict["pilotIndex"]
        Bandwidth=tensordict["params","Bandwidth"]
        antennas=tensordict["params","antennas"]
        new_grouping=tensordict["action"].squeeze(-1)
        new_grouping.clamp(tensordict["params","min_grouping"],\
                           tensordict["params","max_grouping"])
        Pd=tensordict["params","Pd"]
       
        CandidateAPs=tensordict["CandidateAPs"]
        APMask=tensordict["APMask"]
        
        NumberofActions=tensordict["NumberofActions"]
        ActionIndices=tensordict["ActionIndices"]
        
        ServingAPsNew=torch.zeros((M*K,),dtype=torch.int32)
        ServingAPsNew=new_grouping
        ServingAPsNew=ServingAPsNew.reshape(M,K)
    #     ### compute power
    #     aa=time.time()
    #     Power1=torch.zeros(M,K,dtype=torch.float)
    #     for m in range(M):
    #         servedUEs=torch.where(ServingAPsNew[m,:]==1)[0]
    #         denominator=torch.sum(torch.sqrt(gainovernoise[m,servedUEs]))
    #         for k in servedUEs:
    #             Power1[m,k]=Pd*torch.sqrt(gainovernoise[m,k])/denominator
        
    # #compute SEs
    #     signal_MR=np.zeros(K,dtype='float')
    #     interf_MR=np.zeros(K,dtype='float')
    #     cont_MR=np.zeros((K,K),dtype='float')
    #     pre_logfactor=1-tau_p/tau_c
    #     for m in range(M):
    #         servedUEs=np.where(ServingAPsNew[m,:]==1)[0]
    #         for k in servedUEs:
    #             signal_MR[k]=signal_MR[k]+np.sqrt(Power1[m,k]*np.real(np.trace(B[:,:,m,k])))
    #             for i in range(K):
    #                 interf_MR[i]=interf_MR[i]+Power1[m,k]*\
    #                     np.real(np.trace(np.matmul(B[:,:,m,k],R[:,:,m,i])))/np.real(np.trace(B[:,:,m,k]))
    #                 if pilotIndex[k]==pilotIndex[i]:
                        
    #                     cont_MR[i,k]=cont_MR[i,k]+np.sqrt(Power1[m,k])*\
    #                         np.real(np.trace(np.matmul((np.matmul(B[:,:,m,k],np.linalg.inv(R[:,:,m,k]))),R[:,:,m,i])))\
    #                             /np.sqrt(np.real(np.trace(B[:,:,m,k])))
                      
    #     SINR1=(np.abs(signal_MR)**2)/(interf_MR+np.sum(np.abs(cont_MR)**2,axis=1)-np.abs(signal_MR)**2+1)
    #     SE_MR1=pre_logfactor*np.real(np.log2(1+SINR1))    
    #     print(time.time()-aa)
        #ipdb.set_trace()
        #aa=time.time()
        SE_MR,SINR,Power=compute_rates1(M.item(),K.item(),np.array(ServingAPsNew),np.array(gainovernoise),Pd.item(),tau_p.item(),tau_c.item(),np.array(B),np.array(R),np.array(pilotIndex))
        
        #aa=time.time()
        EE=ComputeEE1(np.array(ServingAPsNew),SE_MR,Power,M.item(),Bandwidth.item(),antennas.item())
        #print(time.time()-aa)
        
        
        
        # aa=time.time()
        # #####compute EE
        # Pbackhaul=np.zeros(M)
        # PAPs=np.zeros(M)
        # for m in range(M):
            
        #     EEservedUEs=np.argwhere(ServingAPsNew[m,:]==1)
            
        #     Pbackhaul[m]=0.825+(Bandwidth/1e9)*np.sum(SE_MR[EEservedUEs])*0.25
            
        #     PAPs[m]=(1/0.4)*np.sum(Power[m,EEservedUEs])/1000+0.2*antennas
        
        # TotalPower=np.sum(PAPs)+np.sum(Pbackhaul)
        # EE=((Bandwidth*np.sum(SE_MR))/TotalPower)/1e6
        
        # print(time.time()-aa)
        # ipdb.set_trace()
        ## check for Penalty. appl separate penalties so can learn better
        APsperUE=torch.sum(ServingAPsNew,axis=0)
        UEsperAP=torch.sum(ServingAPsNew,axis=1)
       
        
        Penalty=0
        APPenalty=0
        UEPenalty=0
        SharedPilotPenalty=0
        # if any(APsperUE==0):
        #     UEPenalty=-10 
        # elif any(UEsperAP>tau_p):
        #     APPenalty=-10
        #elif any(SharedPilotUEsperAP)>0:
        #     SharedPilotPenalty=-10
        # else:
        #     Penalty=0
        
        
        
        reward=sum(SE_MR) + Penalty+APPenalty+UEPenalty+SharedPilotPenalty
        #print(self.maxreward)
        
        # ######set reward to zero if receive invalid action and terminate episode
        # if any(APsperUE==0) or any(UEsperAP>tau_p) :
        #     # print(APsperUE)
        #     # print(UEsperAP)
        #     # ipdb.set_trace()    
        #     reward=0 
        #     done=True
        if reward>self.maxreward:
          self.maxreward=reward
        
        self.trackreward[self.steps%20]=reward #should be mod thr+1
        self.steps+=1 
        
        
        threshold=max(sum(self.BetaRate),sum(self.BetaRate))
        
        
            
                
        out = TensorDict(
        {
            "ServingAPs": ServingAPsNew,
            "APsperUE":APsperUE,
            "UEsperAP":UEsperAP,
            #"SharedPilotUEsperAP":SharedPilotUEsperAP,
            "SINR":SINR,
            "EE":EE,
            "gainovernoise": gainovernoise,
            "R": R,
            "B": B,
            "pilotIndex": pilotIndex,
            "params": tensordict["params"],
            "reward": reward,
            "done": done,
            "CandidateAPs": CandidateAPs,
            "APMask":APMask,
            "NumberofActions":NumberofActions,
            "ActionIndices":ActionIndices,
        },
        tensordict.shape,
        )
        return out
    
def _reset(self, tensordict):
    
    self.trackreward=np.zeros(20)
    self.steps=0
    self.maxreward=0 # we want to keep track og max reward seen so far in episode
    self.meet_threshold=0
    if tensordict is None or tensordict.is_empty():
        # if no ``tensordict`` is passed, we generate a single set of hyperparameters
        # Otherwise, we assume that the input ``tensordict`` contains all the relevant
        # parameters to get started.
        tensordict = self.gen_params(self.K,self.M,self.tau_p,batch_size=self.batch_size)
    # for non batch-locked environments, the input ``tensordict`` shape dictates the number
    # of simulators run simultaneously. In other contexts, the initial
    # random state's shape will depend upon the environment batch-size instead.
        M=tensordict["params","M"]
        K=tensordict["params","K"]
        tau_p=tensordict["params","tau_p"]
        Pd=tensordict["params","Pd"]
        tau_c=tensordict["params","tau_c"]
        Bandwidth=tensordict["params","Bandwidth"]
        antennas=tensordict["params","antennas"]
        gainovernoise=self.gainovernoise
        pilotIndex=self.pilotIndex
        APMask=InitMask(gainovernoise, pilotIndex)
        #APMask=torch.zeros((1,M*K),dtype=torch.int32)
        ##### get candidate APs. those which contribute >0.001 of total to UE
        
        CandidateAPs=np.zeros((M,K),dtype=np.int32)
        AP_contr=gainovernoise/np.sum(gainovernoise,axis=0)
        CandidateAPs[AP_contr>self.threshold]=1
        CandidateAPs=CandidateAPs.reshape(1,-1)
        ###assign init aps for each ue between 10-20% 
        ServingAPs=InitialAPs1(self.gainovernoisedB, self.M, self.K,self.threshold)
        ###apply mask to init APs
        
        ServingAPs=(ServingAPs*CandidateAPs.reshape(self.M,self.K)+np.array(APMask.reshape(self.M,self.K))).clip(0,1)
        
        ActionIndices=torch.argwhere(torch.from_numpy(CandidateAPs==1)[0]).squeeze()
        NumberofActions=ActionIndices.shape[0]
        #print(ServingAPs)
        #ipdb.set_trace()
        B=self.B
        R=self.R
        ServingAPsBeta=BetaAPs(M, K,gainovernoise)
        ServingAPsDCC=DCCAPs(M, K,tau_p,gainovernoise,pilotIndex)
        ServingAPsBetascalable=BetaMatching(self.tau_p, np.sum(ServingAPsBeta,0), gainovernoise, self.M, self.K)
        #ipdb.set_trace()
        SE_MR,SINR,Power=self.compute_rates(M,K,ServingAPs,gainovernoise,Pd,tau_p,tau_c,B,R,pilotIndex)
        
        
        #####compute EE
        Pbackhaul=torch.zeros(M)
        PAPs=torch.zeros(M)
        Power=torch.from_numpy(Power)
        for m in range(M):
            
            EEservedUEs=np.argwhere(ServingAPs[m,:]==1)
            if not torch.is_tensor(EEservedUEs):
                EEservedUEs=torch.from_numpy(EEservedUEs)
            
            Pbackhaul[m]=0.825+(Bandwidth/1e9)*torch.sum(SE_MR[EEservedUEs])*0.25
            PAPs[m]=(1/0.4)*torch.sum(Power[m,EEservedUEs])/1000+0.2*antennas
            
        TotalPower=torch.sum(PAPs)+torch.sum(Pbackhaul)
        EE=((Bandwidth*torch.sum(SE_MR))/TotalPower)/1e6
        #print('betaaction',ServingAPsRow_to_real(self, ServingAPsBeta))
        #print('dccaction',ServingAPsRow_to_real(self, ServingAPsDCC))
        #ipdb.set_trace()
        self.BetaRate,_,_=self.compute_rates(M,K,ServingAPsBeta,gainovernoise,Pd,tau_p,tau_c,B,R,pilotIndex)
        self.DCCRate,_,_=self.compute_rates(M,K,ServingAPsDCC,gainovernoise,Pd,tau_p,tau_c,B,R,pilotIndex)
        self.BetaMatchingRate,_,_=self.compute_rates(M,K,ServingAPsBetascalable,gainovernoise,Pd,tau_p,tau_c,B,R,pilotIndex)
        
        APsperUE=torch.sum(torch.from_numpy(ServingAPs),axis=0)
        UEsperAP=torch.sum(torch.from_numpy(ServingAPs),axis=1)
        
        
    out = TensorDict(
        {
            "R": R,
            "B": B,
            "ServingAPs": ServingAPs,
            "APsperUE":APsperUE,
            #"SharedPilotUEsperAP":SharedPilotUEsperAP,
            "UEsperAP":UEsperAP,
            "SINR":SINR,
            "EE":EE,
            "gainovernoise":gainovernoise,
            "pilotIndex":pilotIndex,
            "params": tensordict["params"],
            "CandidateAPs": CandidateAPs,
            "APMask":APMask,
            "NumberofActions":NumberofActions,
            "ActionIndices":ActionIndices,
        },
        batch_size=tensordict.shape,
    )
    return out
def _make_spec(self, td_params):
    # Under the hood, this will populate self.output_spec["observation"]
    M=td_params["params","M"]
    K=td_params["params","K"]
    antennas=td_params["params","antennas"]
    tau_p=td_params["params","tau_p"]
    self.observation_spec = CompositeSpec(
        ServingAPs=BoundedTensorSpec(#not sure this is correct? 
            low=0,
            high=1,
            shape=(M,K),
            dtype=torch.int64,
        ),
        APsperUE=BoundedTensorSpec(#not sure this is correct? 
            low=0,
            high=M,
            shape=(K,),
            dtype=torch.float64,
        ),
        UEsperAP=BoundedTensorSpec(#not sure this is correct? 
            low=0,
            high=K,
            shape=(M,),
            dtype=torch.float64,
        ),
        # SharedPilotUEsperAP=BoundedTensorSpec(#not sure this is correct? 
        #     low=0,
        #     high=tau_p,
        #     shape=(M,),
        #     dtype=torch.float64,
        # ),
        SINR=UnboundedContinuousTensorSpec(#not sure this is correct? 
            shape=(K,),
            dtype=torch.float64,
        ),
        R=UnboundedContinuousTensorSpec(
            shape=(antennas,antennas,M,K),
            dtype=torch.complex128,
        ),
        B=UnboundedContinuousTensorSpec(
            shape=(antennas,antennas,M,K),
            dtype=torch.complex128,
        ),
        gainovernoise=UnboundedContinuousTensorSpec(
            shape=(M,K),
            dtype=torch.float64,
        ),
        pilotIndex=BoundedTensorSpec(
            low=0,
            high=tau_p-1,
            shape=(K,),
            dtype=torch.int32,
        ),
        # we need to add the ``params`` to the observation specs, as we want
        # to pass it at each step during a rollout
        params=make_composite_from_td(td_params["params"]),
        shape=(),
        CandidateAPs=BoundedTensorSpec(#not sure this is correct? 
            low=0,
            high=1,
            shape=(1,M*K),
            dtype=torch.int32,
        ),
        APMask=BoundedTensorSpec(
            low=-1,
            high=1,
            shape=(1,M*K),
            dtype=torch.int32,
        ),
        
        NumberofActions=BoundedTensorSpec(
            low=0,
            high=M*K,
            shape=(),
            dtype=torch.int32,
        ),
        ActionIndices=BoundedTensorSpec(
            low=0,
            high=M*K,
            shape=(M*K,),
            dtype=torch.int32,
        ),
    )
    # since the environment is stateless, we expect the previous output as input.
    # For this, ``EnvBase`` expects some state_spec to be available
    self.state_spec = self.observation_spec.clone()
    # action-spec will be automatically wrapped in input_spec when
    # `self.action_spec = spec` will be called supported
    self.action_spec = BoundedTensorSpec(
        low=0,
        high=1,
        shape=(M*K,),
        dtype=torch.int32,
    )
    self.reward_spec = UnboundedContinuousTensorSpec(shape=(*td_params.shape, 1),dtype=torch.float32)


def make_composite_from_td(td):
    # custom function to convert a ``tensordict`` in a similar spec structure
    # of unbounded values.
    composite = CompositeSpec(
        {
            key: make_composite_from_td(tensor)
            if isinstance(tensor, TensorDictBase)
            else UnboundedContinuousTensorSpec(
                dtype=tensor.dtype, device=tensor.device, shape=tensor.shape
            )
            for key, tensor in td.items()
        },
        shape=td.shape,
    )
    return composite

def gen_params(UEnumber, APnumber, tau_p,batch_size=None) -> TensorDictBase:
    """Returns a ``tensordict`` containing the physical parameters ."""
    if batch_size is None:
        batch_size = []

    BW=20*10**6
    Noisefigure=9
    td = TensorDict(
        {
            "params": TensorDict(
                {
                    "K": UEnumber, #number of UEs
                    "M": APnumber, #number of APs
                    "tau_p":tau_p, #pilot length
                    "tau_c": 200, # coherence length
                    "min_grouping": 0, #decimal for serving APs= 0 for all K
                    "max_grouping": 2**(UEnumber), #decimal for serving APs= 1 for all K
                    "dcorr":0.009, #decorellation distance
                    "delta":0.5, #for shadowing corelation
                    "sigma":4, # sd of shadow fading
                    "NF":9,# noise figure
                    "Bandwidth":BW, #bandwidth 
                    "PNlog":(-203.975+10*np.log10(BW)+Noisefigure), #noise in dB at 290K
                    "hap":15, #ap height in meters
                    "hue":1.65, #ue height in meters
                    "ASD_azimuth":np.deg2rad(15),
                    "ASD_elevation":np.deg2rad(15),
                    "antennaspacing":0.5,
                    "D":1, # in km square length
                    "antennas":1, #number of antennas
                    "channels":20, # number of channel realizations
                    "p":100, #pilot power in mW
                    "Pd":200, # downlink power in mw
                    #"B":B,
                    #"R":R,
                    #"pilotIndex":pilotIndex,
   
                },
                [],
            )
        },
        [],
    )
    if batch_size:
        td = td.expand(batch_size).contiguous()
    return td
def _set_seed(self, seed: Optional[int]):
    rng = torch.manual_seed(seed)
    self.rng = rng
    rng1=np.random.seed(seed)
    self.rng1=rng1
    rng2=random.seed(seed)
    self.rng2=rng2

def InitialAPs(gainovernoisedB,numberofAPs,M,K,threshold):
    InitAPs=np.zeros((M,K),dtype=np.int32)
    CandidateAPs=np.zeros((M,K),dtype=np.int32)
    AP_no=np.round(threshold*M).astype(int)
    for k in range(K):
        sorted=np.argsort(gainovernoisedB[:,k])
        InitAPs[sorted[-numberofAPs:],k]=1
        CandidateAPs[sorted[-AP_no:],k]=1
    return(InitAPs,CandidateAPs)

def InitialAPs1(gainovernoisedB,M,K,threshold): # assign every UE between 10 to 20% of APs
    minAPs=np.round(0.1*M)
    maxAPs=np.round(0.2*M)
    numberAPsperUE=np.random.randint(minAPs,maxAPs+1,K)
    InitAPs=np.zeros((M,K),dtype=np.int32)
    #CandidateAPs=np.zeros((M,K),dtype=np.int32)
    AP_no=np.round(threshold*M).astype(int)
    for k in range(K):
        sorted=np.argsort(gainovernoisedB[:,k])
        InitAPs[sorted[-numberAPsperUE[k]:],k]=1
        #CandidateAPs[sorted[-AP_no:],k]=1
    return(InitAPs)
def InitMask(gainovernoise,pilotIndex):
    M,K=gainovernoise.shape
    MasterAPs=torch.zeros((M,K),dtype=torch.int32)
   
    #sort UEs in order of weakest gain and assign them master AP. Ensure no shared pilots
    #If shared pilot, assign next AP and so on until each UE has a master AP
    _,sortedues=torch.sort(torch.from_numpy(np.sum(gainovernoise,axis=0)))
    sorted_uegain,sorted_apindices=torch.sort(torch.from_numpy(gainovernoise),axis=0,descending=True)
    for k in range(K):
        count=0
        ue=sortedues[k]
        while count<M:
            masterap=sorted_apindices[count,ue]
            servedues=torch.where(MasterAPs[masterap,:]==1)[0]
            
            #assign ap if its free or doesnt serve any ues with its pilot
            if servedues.numel()==0  or (pilotIndex[servedues]==pilotIndex[ue]).any()==False:
                MasterAPs[masterap,ue]=1
                ##update barred ues
                sharedpilotues=np.argwhere(pilotIndex==pilotIndex[ue])
                MasterAPs[masterap,sharedpilotues[sharedpilotues!=ue.item()]]=-1
                count=M
            else:
                count+=1
   
    #MasterAPs=torch.zeros((M,K),dtype=torch.int32)
    return MasterAPs.reshape(1,-1)
    
    
class CfMIMOEnv(EnvBase):
    batch_locked = False
    
    # def __init__(self, td_params=None, seed=None, device="cpu"):
    #     if td_params is None:
    #         td_params = self.gen_params()
    def __init__(self, K,M,tau_p,threshold,td_params=None, seed=None, device="cpu"):
            self.K=K
            self.M=M 
            self.tau_p=tau_p
            
            self.threshold=threshold
            if td_params is None:
                td_params = self.gen_params(self.K,self.M,self.tau_p)
            super().__init__(device=device, batch_size=[])
            self._make_spec(td_params)
            if seed is None or seed<0:
                seed = torch.empty((), dtype=torch.int64).random_().item()
            self._set_seed(seed)
            self.seed=seed
            
            #ipdb.set_trace()
            dcorr=td_params["params","dcorr"]
            delta=td_params["params","delta"]
            sigma=td_params["params","sigma"]
            antennas=td_params["params","antennas"]
            PNlog=td_params["params","PNlog"]
            ASD_azimuth=td_params["params","ASD_azimuth"]
            tau_p=td_params["params","tau_p"]
            hap=td_params["params","hap"]
            hue=td_params["params","hue"]
            channels=td_params["params","channels"]
            p=td_params["params","p"]
            UE,AP,self.R,distance,self.gainovernoisedB,self.gainovernoise,self.pilotIndex,ServingAPs=\
                self.setup(M,K,dcorr,delta,sigma,antennas,PNlog,ASD_azimuth,tau_p,hap,hue)
            #self.ServingAPs=InitialAPs(self.gainovernoisedB,5,self.M,self.K)
            self.ServingAPs=np.random.randint(0,2,(self.M,self.K))
            
            self.H,self.Hhat,self.B,self.C=self.ChannelEstimates(M,antennas,channels,K,p,tau_p,self.pilotIndex,self.R)
            
        

    # Helpers: _make_step and gen_params
    gen_params = staticmethod(gen_params)
    _make_spec = _make_spec
    setup=staticmethod(setup) # make static method otw assumes .self as anothe rinput and counts 12 instead of 11
    ChannelEstimates=staticmethod(ChannelEstimates)
    compute_rates=staticmethod(compute_rates)
    InitialAPs=staticmethod(InitialAPs)

    # Mandatory methods: _step, _reset and _set_seed
    _reset = _reset #_ keeps it private to the class. If want to call it outside the class make public
    _step=_step
    #_step = staticmethod(_step)
    
    _set_seed = _set_seed
    
    
#     #####test
#env=CfMIMOEnv()
#check_env_specs(env)
# print("observation_spec:",env.observation_spec)    
# print("state_spec:",env.state_spec)    
# print("reward_spec:",env.reward_spec)   
# ## test reset and view some values 
# td=env.reset()
# print("reset tensordict",td)
# print(td["ServingAPs"])
# print(td["gainovernoise"])
# ## test random rollout and view some values 
# td=env.rand_step(td)
# print("random step tensordit",td)
# print(td["action"])
# print(td["next","reward"])
### test normalization
