
import sys
sys.path.append('../')
import time
import argparse
import gym
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import time
from matplotlib import pyplot as plt

from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.single_agent_rl.HoverAviary import HoverAviary
from gym_pybullet_drones.utils.utils import sync, str2bool

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

acteur = FeedForwardNN(3, 4,32)

fichier = 'Batch_7850_Rew_-0.0249390885649675.tar'
Path = "F:\PI3\gym-pybullet-drones-0.5.2\examples\checkpoints\%s" %(fichier)
checkpoint = torch.load(Path)
acteur.load_state_dict(checkpoint["model_state_dict"])
batch_rew_tracker = checkpoint['batch_rew_hist']
plt.figure(1)
plt.plot(batch_rew_tracker)


env = gym.make('hover-aviary-v0')
obs, setpoint = env.reset()

t = []
vitesses_angulaires = [[],[],[],[],[],[]]
actions_prises = [[],[],[],[]]

for j in range(1):
    setpoint = np.array([0,0,1])
    for i in range(300):
        if 40 > i > 20:
            setpoint = np.array([-0, -0, -1])
        elif i > 100:
            setpoint = np.array([0,0,0])

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
        obs = setpoint - vitesses_ang_body
        #obs = np.hstack([err, obs[12:16]]).reshape(7, )
        action = acteur(obs)
        print(obs)
        obs, reward, done, info = env.step(action.detach().numpy(), setpoint)
        vitesses_angulaires[0].append(vitesses_ang_body[0])
        vitesses_angulaires[1].append(vitesses_ang_body[1])
        vitesses_angulaires[2].append(vitesses_ang_body[2])
        vitesses_angulaires[3].append(setpoint[0])
        vitesses_angulaires[4].append(setpoint[1])
        vitesses_angulaires[5].append(setpoint[2])
        t.append((i+1)/240)
        actions_prises[0].append(action.detach().numpy()[0])
        actions_prises[1].append(action.detach().numpy()[1])
        actions_prises[2].append(action.detach().numpy()[2])
        actions_prises[3].append(action.detach().numpy()[3])

        #print(obs)
        time.sleep(0.004)
    # plots
    env.close()
    fig, axs = plt.subplots(3)
    fig.suptitle('Solicitation de tous les axes')
    axs[0].plot(t, vitesses_angulaires[0],t, vitesses_angulaires[3])
    axs[0].legend(['Vitesse de Roll','Roll setpoint'])
    axs[0].set_xlabel('Temps (s)')
    axs[0].set_ylabel('Vitesse angulaire (rad/s)')
    axs[1].plot(t, vitesses_angulaires[1], t, vitesses_angulaires[4])
    axs[1].legend(['Vitesse de Pitch','Pitch setpoint'])
    axs[1].set_xlabel('Temps (s)')
    axs[1].set_ylabel('Vitesse angulaire (rad/s)')
    axs[2].plot(t, vitesses_angulaires[2],t, vitesses_angulaires[5])
    axs[2].legend(['Vitesse de Yaw','Yaw setpoint'])
    axs[2].set_xlabel('Temps (s)')
    axs[2].set_ylabel('Vitesse angulaire (rad/s)')



    #plot actions
    plt.figure(3)
    plt.plot(t, actions_prises[0],t, actions_prises[1],t, actions_prises[2],t, actions_prises[3])
    plt.title('Actions prises')
    plt.show()




    #obs, setpoint = env.reset()



