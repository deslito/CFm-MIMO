
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
import torch
import ipdb
from util import *
from sklearn.decomposition import PCA
#import functools
class Evaluator(object):

    def __init__(self, num_episodes, interval, UEnumber,save_path='',SA_mode=None,max_episode_length=None):
        self.num_episodes = num_episodes
        self.max_episode_length = max_episode_length
        self.interval = interval
        self.UEnumber=UEnumber
        self.save_path = save_path
        self.results = np.array([]).reshape(num_episodes,0)
        self.sumBeta=np.array([]).reshape(num_episodes,0)
        self.sumBetascalable=np.array([]).reshape(num_episodes,0)
        self.sumDCC=np.array([]).reshape(num_episodes,0)
        self.sumML=np.array([]).reshape(num_episodes,0)
        self.EEML=np.array([]).reshape(num_episodes,0)
        self.perUEDCC=np.array([]).reshape(0,num_episodes,UEnumber)
        self.perUEBeta=np.array([]).reshape(0,num_episodes,UEnumber)
        self.perUEBetascalable=np.array([]).reshape(0,num_episodes,UEnumber)
        self.perUEML=np.array([]).reshape(0,num_episodes,UEnumber)

    def __call__(self, env, policy,args, APMask,ActionIndices,debug=False, visualize=False, save=True):

        self.is_training = False
        observation = None
        result = []
        M=args.APnumber
        K=args.UEnumber
        tau_p=args.tau_p
        
        SEs_DCC_all=[]
        SEs_ML_all=[]
        EEs_ML_all=[]
        SEs_Beta_all=[]
        SEs_Betascalable_all=[]
        eval_reward=[]
        ML_actionsall=[]
        DCC_actionsall=[]
        Beta_actionsall=[]
        self.SumDCC_step=[]
        self.SumBeta_step=[]
        self.SumBetascalable_step=[]
        self.SumML_step=[]
        self.EEML_step=[]
        
        

        for episode in range(self.num_episodes):
            
            
            SEs_ML=[]
            EEs_ML=[]
            ML_actions=[]
            
            # reset at the start of episode
            #observation = env.reset()
            td_eval=env.reset()
            obs_eval=td_eval.select("normalized_gainovernoise","ServingAPs") #state
            s1_t_eval=obs_eval["normalized_gainovernoise"].view(args.UEnumber*args.APnumber)
            s2_t_eval=obs_eval["ServingAPs"].view(args.UEnumber*args.APnumber).to(torch.float)
            #returns MK as row vector packed as row1,row2 etc
            #s_t_eval=[np.array(s1_t_eval),np.array(s2_t_eval)]
            #s_t_eval=[np.array(s1_t_eval)*np.array(s2_t_eval)]
            s_t_eval=[np.array(td_eval["normalized_gainovernoise"].view(args.UEnumber*args.APnumber))]
            observation=s_t_eval# tensordict with environment i/ps and o/ps
            episode_steps = 0
            episode_reward = 0.
            
            ##################### compute DCC and Beta values once because constant #################################
            if episode==0:
                ServingAPsBeta=np.zeros((M,K)) 
                for k in range(K): 
                    gain=td_eval["gainovernoise"][:,k]
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
                self.beta_act=ServingAPsBeta.reshape(-1,M*K)
                
                ####scalable beta
                
                APquota=td_eval["params","tau_p"]
                UEquota=np.sum(ServingAPsBeta,0)
                
                UEpref=np.argsort(-np.array(td_eval["gainovernoise"]),0)
                APpref=np.argsort(-np.array(td_eval["gainovernoise"]),1)
                ServingAPsBetascalable=np.zeros((M,K),dtype=np.int32)
                proposal_round=0
                while np.sum(UEpref)>M*K:
                    
                    availaps=UEpref>-1
                    
                    for i in range(K):
                        
                        if UEpref[availaps[:,i],i].size==0:
                            continue
                        #ue k proposes to ap 
                        ap=UEpref[availaps[:,i],i][0]
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
                self.betascalable_act=ServingAPsBetascalable.reshape(-1,M*K)
                       
                    
                ##ServingAPsDCC
                #compute DCC serving APs
                #get master AP
                ServingAPsDCC=np.zeros((M,K),dtype=int)
                pilots=td_eval["pilotIndex"]
                for k in range(K): 
                    master=np.argmax(td_eval["gainovernoise"][:,k])
                    ServingAPsDCC[master,k]=1
                #each AP serves UE with strongest channel on each pilot
                for m in range(M): 
                    for t in range(tau_p): 
                        pilotUEs=np.where(pilots==t) #this is a tuple
                        pilotUEs1=pilotUEs[0] #array of indices
                        #print(pilotUEs1)
                        UEindex=np.argmax(td_eval["gainovernoise"][m,pilotUEs1])
                        #print(pilotUEs)
                        ServingAPsDCC[m,pilotUEs1[UEindex]]=1
                self.DCC_act=ServingAPsDCC.reshape(-1,M*K)
                
                self.SE_DCC,Power_DCC=self.compute_rates(args, td_eval, ServingAPsDCC)
                
                self.SE_Beta,Power_Beta=self.compute_rates(args, td_eval, ServingAPsBeta)
                self.SE_BetaScalable,Power_BetaScalable=self.compute_rates(args, td_eval, ServingAPsBetascalable)
                
                self.EE_DCC=self.ComputeEE(ServingAPsDCC, self.SE_DCC, Power_DCC, M, td_eval["params","Bandwidth"], td_eval["params","antennas"])
                self.EE_Beta=self.ComputeEE(ServingAPsBeta, self.SE_Beta, Power_Beta, M, td_eval["params","Bandwidth"], td_eval["params","antennas"])
                self.EE_Betascalable=self.ComputeEE(ServingAPsBetascalable, self.SE_BetaScalable, Power_BetaScalable, M, td_eval["params","Bandwidth"], td_eval["params","antennas"])
            ########################################################################################################
            
            assert observation is not None

            # start episode
            done = False
            terminated=False
            truncated=False
            while not done:
                # basic operation, action ,reward, blablabla ...
                action,_,_= policy(observation)
               
                full_action=np.zeros(M*K,)+np.array(APMask.squeeze()).clip(0,1)
                full_action[ActionIndices]=action
                ML_actions.append(full_action)
                td_eval["action"]=action
                #observation, reward, done, info = env.step(action)
                td_eval=env.step(td_eval)
                ## get new state after action
                obs2=td_eval.select("normalized_gainovernoise",("next","ServingAPs")) #state
                s1_t2_eval=obs2["normalized_gainovernoise"].view(args.UEnumber*args.APnumber)
                s2_t2_eval=obs2["next","ServingAPs"].view(args.UEnumber*args.APnumber).to(torch.float)
                #returns MK as row vector packed as row1,row2 etc
                #s_t2_eval=[np.array(s1_t2_eval),np.array(s2_t2_eval)]
                #s_t2_eval=[np.array(s1_t2_eval)*np.array(s2_t2_eval)]
                s_t2_eval=[np.array(td_eval["next","normalized_gainovernoise"].view(args.UEnumber*args.APnumber))]
                observation,reward,done=s_t2_eval,np.array(td_eval["next","reward"]),np.array(td_eval["next","done"])
                terminated=done
                if self.max_episode_length and episode_steps >= self.max_episode_length -1:
                    done = True
                    truncated=True
                
                #if self.is_training==False: # compute these values once because AP,UE locations fixed per episode
                    ## plot comparison values for DCC and Beta
                    ##ServingAPsBeta
                    ## only compute once since constant over episode
                    
                
                SE_ML,Power_ML=self.compute_rates(args,td_eval,td_eval["next", "ServingAPs"])
                
                EE_ML=self.ComputeEE(td_eval["next", "ServingAPs"], SE_ML, Power_ML, args.APnumber, td_eval["params","Bandwidth"], td_eval["params","antennas"])
                #########################
                #check
                # DCC_dec=self.ServingAPsRow_to_real(ServingAPsDCC)
                # Beta_dec=self.ServingAPsRow_to_real(ServingAPsBeta)
                # print('DCC',DCC_dec)
                # print('Beta',Beta_dec)
                #print('ML',action)
                #print('Beta',beta_act)
                #print('DCC',DCC_act)
                 
                # print('SEDCC',SE_DCC)
                # print('SEBeta',SE_Beta)
                #print('SEML',torch.sum(SE_ML))
                # print(torch.sum(SE_DCC),torch.sum(SE_Beta),torch.sum(SE_ML))
                # ipdb.set_trace()
                #########################
                
                
                SEs_ML.append(np.array(SE_ML))
                EEs_ML.append(np.array(EE_ML)) ###EE per step

                # update
                episode_reward += reward
                episode_steps += 1
                eval_reward.append(reward)
                #print(episode_steps)
            if debug: prYellow('[Evaluate] #Episode{}: episode_reward:{}'.format(episode,episode_reward))
            result.append(episode_reward)
            
            ### save all episode results
            #if self.is_training == False:
            ML_actionsall.append(np.array(ML_actions))
            SEs_ML_all.append(np.mean(np.array(SEs_ML),axis=0)) ##mean over episode
            EEs_ML_all.append(np.mean(np.array(EEs_ML),axis=0)) ## mean over episode
            self.SumML_step.append(np.sum(np.array(SEs_ML),axis=1).reshape(-1,1))
           
            self.EEML_step.append(np.array(EEs_ML).reshape(-1,1))
            
            # aa=np.array(SumML_step).reshape(-1,1)
            # bb=np.array(SumDCC_step).reshape(-1,1)
            # cc=np.array(SumBeta_step).reshape(-1,1)
            # plt.plot(aa)
            # plt.plot(bb)
            # plt.plot(cc)
            # plt.legend(['ML','DCC','Beta'])
            # plt.show()
            
            # ipdb.set_trace()  
                #print(SEs_ML_all)
                #print(SEs_DCC_all)
                #ipdb.set_trace()
        ### visualize testing actions 
        #if self.is_training == False:
        # Apply PCA to reduce to 2 dimensions
        actionsML=np.array(ML_actionsall).reshape(-1,M*K)
        
        # pca = PCA(n_components=2)
        # reduced_ML = pca.fit_transform(actionsML)
        # ### use same pca as above
        # reduced_beta=pca.transform(actionsBeta)
        # reduced_dcc=pca.transform(actionsDCC)
        # # Scatter plot in 2D
        # plt.scatter(reduced_ML[:, 0], reduced_ML[:, 1],label='visited_actions')
        # plt.scatter(reduced_beta[:, 0], reduced_beta[:, 1],label='betaactions')
        # plt.scatter(reduced_dcc[:, 0], reduced_dcc[:, 1],label='dccactions')
        # plt.xlabel("PC1")
        # plt.ylabel("PC2")
        # plt.title("PCA: 2D Visualization")
        # plt.legend()
        # plt.show()
            #ipdb.set_trace()
        result = np.array(result).reshape(-1,1)
        self.results = np.hstack([self.results, result])
        #if self.is_training == False:
        
        self.SEs_ML_all=SEs_ML_all
        self.EEs_ML_all=EEs_ML_all
        #np.mean(np.array(SEs_ML_all),axis=1)
        
       
        self.ML_actionsall=np.array(ML_actionsall).squeeze()
        
       
        self.DCC_actionsall=np.tile(self.DCC_act,(self.num_episodes,self.max_episode_length,1))
        self.Beta_actionsall=np.tile(self.beta_act,(self.num_episodes,self.max_episode_length,1))
        self.Betascalable_actionsall=np.tile(self.betascalable_act,(self.num_episodes,self.max_episode_length,1))
        self.sumML=np.hstack([self.sumML,np.sum(self.SEs_ML_all,axis=1).reshape(-1,1)])
        self.EEML=np.hstack([self.EEML,np.array(self.EEs_ML_all).reshape(-1,1)])
        self.EEDCC=np.tile(self.EE_DCC,(self.num_episodes,1)) 
        self.EEBeta=np.tile(self.EE_Beta,(self.num_episodes,1)) 
        self.EEBetascalable=np.tile(self.EE_Betascalable,(self.num_episodes,1)) 
        self.sumDCC= np.tile(np.array(torch.sum(self.SE_DCC)),self.num_episodes).reshape(-1,1)
        self.sumBeta= np.tile(np.array(torch.sum(self.SE_Beta)),self.num_episodes).reshape(-1,1)
        self.sumBetascalable= np.tile(np.array(torch.sum(self.SE_BetaScalable)),self.num_episodes).reshape(-1,1)
        
        if episode==0:
            
            self.perUEML=np.expand_dims(self.SEs_ML_all,axis=0)
        else:
            #not really a good measure for ML case because its a mean over whole episode which could be diffe/
            #different for different steps depending on allocation
            ## actually its ok because converges to one value
            
            self.perUEML=np.concatenate((self.perUEML,np.expand_dims(self.SEs_ML_all,axis=0)), axis=0)
        
        self.perUEBeta=np.tile(np.expand_dims(np.array(self.SE_Beta),axis=(0,1)),(self.num_episodes,1))
        self.perUEBetascalable=np.tile(np.expand_dims(np.array(self.SE_BetaScalable),axis=(0,1)),(self.num_episodes,1))
        self.perUEDCC=np.tile(np.expand_dims(np.array(self.SE_DCC),axis=(0,1)),(self.num_episodes,1))
        
        self.SumML_step=np.array(self.SumML_step)
        self.EEML_step=np.array(self.EEML_step)
        self.SumDCC_step= np.tile(np.expand_dims(self.sumDCC,1),(self.max_episode_length,1))
        self.SumBeta_step= np.tile(np.expand_dims(self.sumBeta,1),(self.max_episode_length,1))
        self.SumBetascalable_step= np.tile(np.expand_dims(self.sumBetascalable,1),(self.max_episode_length,1))
        
        self.EEDCC_step= np.tile(np.expand_dims(self.EE_DCC,axis=0),(self.num_episodes,self.max_episode_length,1))
        self.EEBeta_step= np.tile(np.expand_dims(self.EE_Beta,axis=0),(self.num_episodes,self.max_episode_length,1))
        self.EEBetascalable_step= np.tile(np.expand_dims(self.EE_Betascalable,axis=0),(self.num_episodes,self.max_episode_length,1))
        
        #ipdb.set_trace()
        self.visualize=visualize
        
        
        
        if save:
            self.save_results('{}/validate_reward'.format(self.save_path))
        return np.mean(result)#,eval_reward #mean cumulative episode reward, reward per step

    def save_results(self, fn):
       
        y = np.mean(self.results, axis=0)
        error=np.std(self.results, axis=0)
                    
        x = range(0,self.results.shape[1]*self.interval,self.interval)
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        plt.xlabel('Timestep')
        plt.ylabel('Average Reward')
        ax.errorbar(x, y, yerr=error, fmt='-o')
        plt.savefig(fn+'.png')
        savemat(fn+'.mat', {'reward':self.results})
        plt.close()
        #ipdb.set_trace()
        # if self.is_training==False:
            
        savemat(fn+'1.mat',{'SumSE_DCC':self.sumDCC,'SumSE_Beta':self.sumBeta,'SumSE_Betascalable':self.sumBetascalable,'SumSE_ML':self.sumML,'PerUE_DCC':self.perUEDCC,'perUE_Beta':self.perUEBeta,'perUE_Betascalable':self.perUEBetascalable,'perUE_ML':self.perUEML,\
                            'SumDCC_step':self.SumDCC_step,'SumBeta_step':self.SumBeta_step,'SumBetascalable_step':self.SumBetascalable_step,'SumML_step':self.SumML_step,'DCCActions':self.DCC_actionsall,'BetaActions':self.Beta_actionsall,\
                                'BetaScalableActions':self.Betascalable_actionsall,'MLActions':self.ML_actionsall,'EEML_step':self.EEML_step,'EEDCC_step':self.EEDCC_step,'EEBeta_step':self.EEBeta_step,\
                                   'EEBetascalable_step':self.EEBetascalable_step, 'AvgEE_ML':self.EEML,'AvgEE_DCC':self.EEDCC,'AvgEE_Beta':self.EEBeta,'AvgEE_Betascalable':self.EEBetascalable})
            
            # #ipdb.set_trace()
            # ysum=np.linspace(0,1,num=sumDCC.size)
            # yue=np.linspace(0,1,num=perUEDCC.size)
            # plt.figure(1)
            # ### in final plot remove duplicate values
            # plt.plot(sumDCC,ysum,'g',
            # label='DCC')
            # plt.plot(sumBeta,ysum,'b',
            # label='Beta')
            # plt.plot(sumML,ysum,'r',
            # label='ML')
            # plt.xlabel('sumSE')
            # plt.ylabel('CDF')
            # plt.legend()
            # plt.savefig(fn+'1.png')
            
            # plt.figure(2)
            # ### in final plot remove duplicate values
            # plt.plot(perUEDCC,yue,'g',
            # label='DCC')
            # plt.plot(perUEBeta,yue,'b',
            # label='Beta')
            # plt.plot(perUEML,yue,'r',
            # label='ML')
            # plt.xlabel('per UE SE')
            # plt.ylabel('CDF')
            # plt.savefig(fn+'2.png')

            
            
    #@functools.lru_cache(maxsize=10000)    
    def compute_rates(self,args,td,APs):
        self.M=args.APnumber
        self.K=args.UEnumber
        self.ServingAPs=APs
        self.gainovernoise=td["gainovernoise"]
        self.Pd=td["params","Pd"]
        self.tau_p=td["params","tau_p"]
        self.tau_c=td["params","tau_c"]
        self.B=td["B"]
        self.R=td["R"]
        self.pilotIndex=td["pilotIndex"]
        Power=torch.zeros(self.M,self.K,dtype=torch.float)
        for m in range(self.M):
            servedUEs=np.where(self.ServingAPs[m,:]==1)[0]
            denominator=torch.sum(torch.sqrt(self.gainovernoise[m,servedUEs]))
            for k in servedUEs:
                Power[m,k]=self.Pd*torch.sqrt(self.gainovernoise[m,k])/denominator  
    #compute SEs
        signal_MR=np.zeros(self.K,dtype='float')
        interf_MR=np.zeros(self.K,dtype='float')
        cont_MR=np.zeros((self.K,self.K),dtype='float')
        pre_logfactor=1-self.tau_p/self.tau_c
        for m in range(self.M):
            servedUEs=np.where(self.ServingAPs[m,:]==1)[0]
            for k in servedUEs:
                signal_MR[k]=signal_MR[k]+np.sqrt(Power[m,k]*np.real(np.trace(self.B[:,:,m,k])))
                for i in range(self.K):
                    interf_MR[i]=interf_MR[i]+Power[m,k]*\
                        np.real(np.trace(np.matmul(self.B[:,:,m,k],self.R[:,:,m,i])))/np.real(np.trace(self.B[:,:,m,k]))
                    if self.pilotIndex[k]==self.pilotIndex[i]:
                        cont_MR[i,k]=cont_MR[i,k]+np.sqrt(Power[m,k])*\
                            np.real(np.trace(np.matmul((np.matmul(self.B[:,:,m,k],np.linalg.inv(self.R[:,:,m,k]))),self.R[:,:,m,i])))\
                                /np.sqrt(np.real(np.trace(self.B[:,:,m,k])))
        SINR=(np.abs(signal_MR)**2)/(interf_MR+np.sum(np.abs(cont_MR)**2,axis=1)-np.abs(signal_MR)**2+1)
        SE_MR=pre_logfactor*np.real(np.log2(1+SINR)) 
        return(SE_MR,Power)
    ##### convert row to decimal

    def ServingAPsRow_to_real(self,ServingAPs):
        #returns o/p of sum_(from i=1 to m) sum_(from j=1 to n)*a_ij*2^((i-1)n+j-1)
        #m= rows, n= columns
        #note indices 0 is the last element as, thats why flipped
        #converting flattened o/p to binary
        M,K=ServingAPs.shape
        indices=np.flip(np.arange(K))
        real=np.sum(ServingAPs*2**indices,axis=1)
        return(real)
    
    def ComputeEE(self,ServingAPs,UE_SEs,UE_power,M,Bandwidth,antennas):
        
        Pbackhaul=torch.zeros(M)
        PAPs=torch.zeros(M)
        
        for m in range(M):
            
            EEservedUEs=np.argwhere(ServingAPs[m,:]==1)
            if not torch.is_tensor(EEservedUEs):
                EEservedUEs=torch.from_numpy(EEservedUEs)
            
            Pbackhaul[m]=0.825+(Bandwidth/1e9)*torch.sum(UE_SEs[EEservedUEs])*0.25
            PAPs[m]=(1/0.4)*torch.sum(UE_power[m,EEservedUEs])/1000+0.2*antennas
                
        TotalPower=torch.sum(PAPs)+torch.sum(Pbackhaul)
        EE=((Bandwidth*torch.sum(UE_SEs))/TotalPower)/1e6
        return EE
