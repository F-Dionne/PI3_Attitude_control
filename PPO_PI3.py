#Entrainement d'un réseau neuronal pour le contrôle d'attitude d'un drone
#Écrit par François Dionne
#https://medium.com/swlh/coding-ppo-from-scratch-with-pytorch-part-2-4-f9d8b8aa938a
#packages installés à C:\Users\Fanch\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.9_qbz5n2kfra8p0\LocalCache\local-packages\Python39\site-packages

import sys
sys.path.append('../')
import gym
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from torch.optim import Adam
from gym_pybullet_drones.envs.single_agent_rl.HoverAviary import HoverAviary


#Définition de la classe qui permettra de faire l'initialisation des réseaux neuronaux.
class FeedForwardNN(nn.Module):

    def __init__(self,N_input,N_output,Largeur):
        super(FeedForwardNN,self).__init__()
        #Définition du NN
        self.layer1 = nn.Linear(N_input,Largeur)
        self.layer2 = nn.Linear(Largeur,Largeur)
        self.layer3 = nn.Linear(Largeur,N_output)


    def forward(self,obs):
        #Définition d'une évalutation du NN
        #Transformation des observations en torch.tensor
        if isinstance(obs,np.ndarray):
            obs = torch.tensor(obs,dtype=torch.float)

        #Évaluation
        activ1 = F.tanh(self.layer1(obs))
        activ2 = F.tanh(self.layer2(activ1))
        output = self.layer3(activ2)

        return output



#Initialisation de l'environnement
env = gym.make('hover-aviary-v0')

class PPO:
    def __init__(self,env):
        #Initialiser les paramètres de la PPO
        self._init_hyperparameters()

        #Définir l'environnement
        self.env = env
        self.obs_dim = 3 #Nombre d'entrées du RN (erreurs de vitesses angulaires)
        self.action_dim = 4 #Nombre de sorties du RN (Commandes au moteurs)

        # Initialiser l'acteur à entraîner et le critique et leurs algorithmes d'optimisation (ADAM)
        self.acteur = FeedForwardNN(self.obs_dim, self.action_dim,32)
        self.critique = FeedForwardNN(self.obs_dim, 1, 5)
        self.actor_optim = Adam(self.acteur.parameters(),lr=self.lr)
        self.critique_optim = Adam(self.critique.parameters(),lr=self.lr)

        #Initialisation des matrices pour la normalisation des actions
        self.cov_var = torch.full(size=(self.action_dim,),fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)

        #Initialisation de paramètres pour savoir où on est rendu dans le training
        self.nb_batch = 0
        self.batch_rew_tracker = []

    def learn(self, max_batch):

        while self.nb_batch < max_batch:

            #Faire une batch de simulations et enregistrer les observations
            batch_obs, batch_act, batch_logprob, batch_rtg, batch_lens, batch_avg_rew = self.rollout()

            #Enregistrement de l'avancement
            self.nb_batch += 1
            self.batch_rew_tracker.append(batch_avg_rew)

            #Calcul de l'avantage et normalisation
            V, log_probs = self.evaluate(batch_obs, batch_act)
            A_k = batch_rtg - V.detach()
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            #Print l'avancement de l'entraînement et enregistrement
            self.update(batch_avg_rew)

            for _ in range(self.n_updates_per_iteration):
                #Probabilité des actions sachant les états
                V, curr_log_probs = self.evaluate(batch_obs,batch_act)
                #Calcul du ratio (equations PPO)
                ratios = torch.exp(curr_log_probs - batch_logprob)

                #Calcul surrogate losses
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

                #Calcul de la perte pour l'entrainement de l'acteur
                actor_loss = (-torch.min(surr1,surr2)).mean()
                critic_loss = nn.MSELoss()(V,batch_rtg)
                #Faire la backprop sur l'acteur
                self.actor_optim.zero_grad()
                actor_loss.backward()
                self.actor_optim.step()

                #Faire le backprop sur le critique
                self.critique_optim.zero_grad()
                critic_loss.backward(retain_graph=True)
                self.critique_optim.step()


    def evaluate(self, batch_obs,batch_act):
        mean = self.acteur(batch_obs)
        dist = MultivariateNormal(mean,self.cov_mat)
        log_probs = dist.log_prob(batch_act)
        V = self.critique(batch_obs).squeeze()
        return V, log_probs

    def rollout(self):
        #Données que l'on enregistre
        batch_obs = []          #Observations
        batch_act = []          #Actions prises
        batch_logprob = []      #Probrabilité logarithmique des actions
        batch_rew = []          #Récompenses amassées
        batch_rtg = []          #Récompenses à venir
        batch_lens = []         #Longueur des épisodes
        batch_avg_rew = []      #Pour calculer la récompense moyenne de la batch

        t = 0
        while t < self.step_par_batch:
            ep_rews = []
            obs, setpoint = self.env.reset()
            done = False

            for ep_t in range(self.max_step_par_episode):

                ROLL = obs[3]
                PITCH = obs[4]
                YAW = obs[5]
                VEL_ROLL = obs[9]
                VEL_PITCH = obs[10]
                VEL_YAW = obs[11]
                vecteur_angvel = np.array([VEL_ROLL, VEL_PITCH, VEL_YAW])
                Transformation_matrix = np.array([[1, np.sin(ROLL) * np.tan(PITCH), np.cos(ROLL) * np.tan(PITCH)],
                                                  [0, np.cos(ROLL), -np.sin(ROLL)],
                                                  [0, np.sin(ROLL) / np.cos(PITCH), np.cos(ROLL) / np.cos(PITCH)]])
                mat_inv = np.linalg.inv(Transformation_matrix)
                vitesses_ang_body = np.matmul(mat_inv, vecteur_angvel)
                err = setpoint - vitesses_ang_body
                obs = err#np.hstack([err,obs[12:16]]).reshape(7,)

                batch_obs.append(obs)
                action, log_prob = self.get_action(obs)
                obs, rew, done, _ = self.env.step(action,setpoint)

                #Enregistrer
                ep_rews.append(rew)
                batch_act.append(action)
                batch_logprob.append(log_prob)
                t += 1
                if done:
                    break
            batch_lens.append(ep_t + 1) #Nb de timestep effectué
            batch_rew.append(ep_rews)
            batch_avg_rew.append(ep_rews[-1])
        #Passer les array sous forme de tenseurs
        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_act = torch.tensor(batch_act, dtype=torch.float)
        batch_logprob = torch.tensor(batch_logprob, dtype=torch.float)
        batch_rtg = self.compute_rtgs(batch_rew)
        batch_avg_rew = np.mean(batch_avg_rew)
        return batch_obs, batch_act, batch_logprob, batch_rtg, batch_lens, batch_avg_rew

    def get_action(self,obs):
        #Obtenir l'action calculée par RN
        mean = self.acteur(obs)
        #Créer la distribution normale pour avoir un peu d'actions aléatoires (exploration)
        dist = MultivariateNormal(mean,self.cov_mat)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.cpu().detach().numpy(), log_prob.detach()

    def compute_rtgs(self,batch_rew):
        batch_rtgs = []
        for ep_rews in reversed(batch_rew):
            discounted_reward = 0
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0,discounted_reward)
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)
        return batch_rtgs

    def update(self, batch_avg_rew):
        print("Reward : %f \n Batch : %d" %(batch_avg_rew, self.nb_batch))
        Path = "F:\PI3\gym-pybullet-drones-0.5.2\examples\checkpoints\Batch_%s_Rew_%s.tar"  %(self.nb_batch,batch_avg_rew)

        if self.nb_batch%self.checkpoint_frequency == 0:
            torch.save({
                'epoch': self.nb_batch,
                'model_state_dict': self.acteur.state_dict(),
                'optimizer_state_dict': self.critique.state_dict(),
                'batch_rew_hist': self.batch_rew_tracker,
                'step_par_batch': self.step_par_batch,
                'max_step_par_episode': self.max_step_par_episode,
                'gamma': self.clip,
                'n_updates_per_iteration': self.n_updates_per_iteration,
                'lr': self.lr,
                'checkpoint_frequency': self.checkpoint_frequency
            }, Path)


    def load(self,fichier):
        Path = "F:\PI3\gym-pybullet-drones-0.5.2\examples\checkpoints\%s" %(fichier)
        checkpoint = torch.load(Path)
        self.acteur.load_state_dict(checkpoint["model_state_dict"])
        self.critique.load_state_dict(checkpoint["optimizer_state_dict"])
        self.nb_batch = checkpoint['epoch']
        self.batch_rew_tracker = checkpoint['batch_rew_hist']


    def _init_hyperparameters(self):
        #À modifier pour améliorer le résultat de l'entrainement
        self.step_par_batch = 3200
        self.max_step_par_episode = 50
        self.gamma = 0.99               #discount factor
        self.clip = 0.25
        self.n_updates_per_iteration = 2
        self.lr = 0.0003
        self.checkpoint_frequency = 25      #Checkpoint à chaque cb de batch




#Entrainement du contrôleur avec la classe PPO définie plus haut

controleur = PPO(env)

#controleur.load('Batch_750_Rew_-0.06393787184505353.tar')
controleur.learn(15000) #Faire de l'entrainement pour X episodes

