from sklearn.preprocessing import KBinsDiscretizer
import numpy as np 
import math
from typing import Tuple
import gym
from tqdm import tqdm
import torch
from torch import nn
import torch.optim as optim
import matplotlib.pyplot as plt


env = gym.make('CartPole-v0')

n_bins = ( 6 , 12 )
lower_bounds = [ env.observation_space.low[2], -math.radians(50) ]
upper_bounds = [ env.observation_space.high[2], math.radians(50) ]

def discretizer( _ , __ , angle, pole_velocity ) -> Tuple[int,...]:
    
    est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
    est.fit([lower_bounds, upper_bounds ])
    return tuple(map(int,est.transform([[angle, pole_velocity]])[0]))

Q_table = np.zeros(n_bins + (env.action_space.n,))

def policy( state : tuple ):
    
    return np.argmax(Q_table[state])

def new_Q_value( reward : float ,  new_state : tuple , discount_factor=1 ) -> float:
    
    future_optimal_value = np.max(Q_table[new_state])
    learned_value = reward + discount_factor * future_optimal_value
    return learned_value

# Adaptive learning of Learning Rate
def learning_rate(n : int , min_rate=0.01 ) -> float  :
    
    return max(min_rate, min(1.0, 1.0 - math.log10((n + 1) / 25)))

def exploration_rate(n : int, min_rate= 0.1 ) -> float :
    
    return max(min_rate, min(1, 1.0 - math.log10((n  + 1) / 25)))

n_episodes = 2000 
for e in tqdm(range(n_episodes)):
    
    # Siscretize state into buckets
    current_state, done = discretizer(*env.reset()), False
    
    while done==False:
        
        # policy action 
        action = policy(current_state) # exploit
        
        # insert random action
        if np.random.random() < exploration_rate(e) : 
            action = env.action_space.sample() # explore 
         
        # increment enviroment
        obs, reward, done, _ = env.step(action)
        new_state = discretizer(*obs)
        
        # Update Q-Table
        lr = learning_rate(e)
        learnt_value = new_Q_value(reward , new_state )
        old_value = Q_table[current_state][action]
        Q_table[current_state][action] = (1-lr)*old_value + lr*learnt_value
        
        current_state = new_state

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

#generative adversarial network
z_dim = (1, 4, 6, 6) #noise vector dimension
class Generator(nn.Module):
    def __init__(self, z_dim=4, hidden_dim=64):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.model = nn.Sequential(            
            self.make_gen_block(z_dim, hidden_dim * 4),
            self.make_gen_block(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=1),
            self.make_gen_block(hidden_dim * 2, hidden_dim),
            self.make_gen_block(hidden_dim, 4, kernel_size=4, final_layer=True)
            )
    def make_gen_block(self, input_channels, output_channels, kernel_size=3, stride=2, padding=3, final_layer=False):
        
        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True)
            )
        else: # Final Layer
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, 1, padding=0),
                nn.Sigmoid()
            )

    def forward(self, noise):
        return self.model(noise)

class Discriminator(nn.Module):
    def __init__(self, in_chan=4, hidden_dim=64):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            self.make_disc_block(in_chan, hidden_dim),
            self.make_disc_block(hidden_dim, hidden_dim * 2),
            self.make_disc_block(hidden_dim * 2, 1, final_layer=True),
            #nn.Sigmoid()
            )
        
    def make_disc_block(self, input_channels, output_channels, kernel_size=2, stride=1, final_layer=False):
        if not final_layer:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride),
                nn.BatchNorm2d(output_channels),
                nn.LeakyReLU(0.2, inplace=True)
            )
        else: # Final Layer
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride),
                nn.Linear(3, 1),
                nn.Sigmoid()
            )
    def forward(self, inp):
        disc_pred = self.model(inp)
        return disc_pred.view(len(disc_pred), -1)
    
def get_noise(mean, std_dev):
    return torch.from_numpy(np.random.normal(mean, std_dev, size=z_dim)).to(device)

disc = Discriminator().cuda()
gen = Generator().cuda()
gen_opt = optim.Adam(gen.parameters(), lr=0.0001)
disc_opt = optim.Adam(disc.parameters(), lr=0.001)
criterion = nn.BCELoss()


running_Gloss = []
running_Dloss = []
fake_noise = get_noise(0.0,0.5)
def get_disc_loss(real):
    #fake_noise = get_noise()
    fake = gen(fake_noise.float())
    disc_fake_pred = disc(fake.detach())
    disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
    disc_real_pred = disc(real.float())
    disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
    disc_loss = (disc_fake_loss + disc_real_loss) / 2
    running_Dloss.append(disc_loss)
    return disc_loss

def get_gen_loss():
    #fake_noise = get_noise() 
    fake = gen(fake_noise.float())
    disc_fake_pred = disc(fake)
    gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
    running_Gloss.append(gen_loss)
    return gen_loss

real = Q_table/np.linalg.norm(Q_table)
real = torch.reshape(torch.from_numpy(real), (1, 4, 6, 6)).to(device)

rt = 2000
for epoch in tqdm(range(rt)):
    
    # Update discriminator
    # Zero out the gradients before backpropagation
    disc_opt.zero_grad()

    # Calculate discriminator loss
    disc_loss = get_disc_loss(real)

    # Update gradients
    disc_loss.backward(retain_graph=True)

    # Update optimizer
    disc_opt.step()

    gen_opt.zero_grad()
    gen_loss = get_gen_loss()
    gen_loss.backward()
    gen_opt.step()
        


x = range(rt)
y = running_Dloss
plt.plot(x,y)
plt.title('Performance through epochs')
plt.xlabel('Epochs')
plt.ylabel('disc_loss')
plt.show()

x = range(rt)
y = running_Gloss
plt.plot(x,y)
plt.title('Performance through epochs')
plt.xlabel('Epochs')
plt.ylabel('gen_loss')
plt.show()


# to be looped for optimal qtable selection

best_performer = 400 # best performing agent steps (this value can be tweaked around)

from Desktop.SCAAI.CustomEnv.custom_cartpole.env.cp2 import CartPoleEnv2

while True:
    
    step_cntr = 0
    env = CartPoleEnv2()
    gen_fake_noise = get_noise(0, np.random.randint(0,100))
    gen_qtable = gen(gen_fake_noise.float()).detach().cpu().numpy()
    gen_qtable = np.reshape(gen_qtable, Q_table.shape)
    n_ep = 1
        
    current_state = discretizer(*env.reset())
    done = False

    while done==False:
        action = np.argmax(gen_qtable[current_state])
        obs, reward, done, _ = env.step(action)
        new_state = discretizer(*obs)
        current_state = new_state
        
        step_cntr += 1
        #env.render()
    if (step_cntr>best_performer): 
        print("found! steps taken:", step_cntr)
        best_qtable = gen_qtable        
        break
    else:
        continue 
    print("\nsteps taken by the generator agent: ", step_cntr)


# generator as an agent 1

# plot performance graphs for different noise vectors!!!

 #just a counter, to note the total steps taken in the 200 episodes

from Desktop.SCAAI.CustomEnv.custom_cartpole.env.cp2 import CartPoleEnv2
mean_arr = [0,0,1,1,-1,-1,2,2,-2,-2]
std_dev_arr = [0.25,1,1.5,2,0.5,1,1.5,2,3,3]
total_ep_steps = []

fig, axs = plt.subplots(2, 5, figsize=(15, 6))
fig.subplots_adjust(hspace = .5, wspace=.001)
axs=axs.ravel()

for i in range(10):
    total_ep_steps=[]
    env = CartPoleEnv2()
    gen_fake_noise = get_noise(mean_arr[i], std_dev_arr[i])
    gen_qtable = gen(gen_fake_noise.float()).detach().cpu().numpy()
    gen_qtable = np.reshape(gen_qtable, Q_table.shape)
    n_ep = 10
    for e in range(n_ep):
        step_cntr = 0
        current_state = discretizer(*env.reset())
        done = False
    
        while done==False:
            action = np.argmax(gen_qtable[current_state])
            obs, reward, done, _ = env.step(action)
            new_state = discretizer(*obs)
            current_state = new_state
            
            step_cntr += 1
            #env.render()
        total_ep_steps.append(step_cntr)
    print("\nsteps taken by the generator agent: ", step_cntr)
    #env.close()
    
    
    
    y = total_ep_steps
    x = range(len(y))
    axs[i].bar(x,y, color='mediumseagreen')
    axs[i].axis(ymin=0,ymax=500)


# random actions agent [for comparison]
from Desktop.SCAAI.CustomEnv.custom_cartpole.env.cp2 import CartPoleEnv2
 

fig, axs = plt.subplots(2, 5, figsize=(15, 6))
fig.subplots_adjust(hspace = .5, wspace=.001)
axs=axs.ravel()
expl_rt = 0

for i in range(10):
    env = CartPoleEnv2()
    
    n_ep=10
    total_ep_steps_wsome_rnd = []
    for e in range(n_ep):
        step_cntr = 0
        obs = env.reset()
        rewards = 0
        done = False
        
        while not done:
            action = np.argmax(Q_table[current_state])
            # insert random action
            obs, reward, done, _ = env.step(action)
            new_state = discretizer(*obs)
            current_state = new_state
            
            env.render()
            step_cntr += 1
        total_ep_steps_wsome_rnd.append(step_cntr)
    env.close()
    print("steps taken: ", step_cntr)
    expl_rt += 0.1      
    
    y = total_ep_steps_wsome_rnd
    x = range(10)
    axs[i].bar(x,y, color='mediumseagreen')
    axs[i].axis(ymin=0,ymax=500)
    plt.title('Performance from original qtable')
    plt.xlabel('Episodes')
    plt.ylabel('Step_taken')


