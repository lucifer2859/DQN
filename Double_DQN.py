import math, random, os

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

from IPython.display import clear_output
import matplotlib.pyplot as plt

### Use Cuda ###
device = "cuda:3"

### Replay Buffer ###
from collections import deque

class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
            
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done
    
    def __len__(self):
        return len(self.buffer)


### Cart Pole Environment ###
env_id = "CartPole-v0"
env = gym.make(env_id)

### Epsilon greedy exploration ###
epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 500

epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)

# plt.plot([epsilon_by_frame(i) for i in range(10000)])

### Double Deep Q Network ###
class DQN(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(DQN, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, env.action_space.n)
        )
        
    def forward(self, x):
        return self.layers(x)
    
    def act(self, state, epsilon):
        if random.random() > epsilon:
            with torch.no_grad():
                state = Variable(torch.FloatTensor(state).unsqueeze(0)).to(device)
            q_value = self.forward(state)
            action = int(q_value.max(1)[1].data[0].cpu().int().numpy())
        else:
            action = random.randrange(env.action_space.n)
        return action

current_model = DQN(env.observation_space.shape[0], env.action_space.n).to(device)
target_model  = DQN(env.observation_space.shape[0], env.action_space.n).to(device)
    
optimizer = optim.Adam(current_model.parameters())

replay_buffer = ReplayBuffer(1000)

### Synchronize current policy net and target net ###
def update_target(current_model, target_model):
    target_model.load_state_dict(current_model.state_dict())

update_target(current_model, target_model)

### Computing Temporal Difference Loss ###
def compute_td_loss(batch_size):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state      = Variable(torch.FloatTensor(np.float32(state))).to(device)
    next_state = Variable(torch.FloatTensor(np.float32(next_state))).to(device)
    action     = Variable(torch.LongTensor(action)).to(device)
    reward     = Variable(torch.FloatTensor(reward)).to(device)
    done       = Variable(torch.FloatTensor(done)).to(device)

    q_values = current_model(state)
    next_q_values = current_model(next_state)
    next_q_state_values = target_model(next_state) 

    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1) 
    next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
    expected_q_value = reward + gamma * next_q_value * (1 - done)
    
    loss = (q_value - expected_q_value.detach()).pow(2).mean()
        
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss

def CartPole_plot(frame_idx, rewards, losses):
    clear_output(True)
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, np.mean(rewards[-10:])))
    plt.plot(rewards)
    plt.subplot(132)
    plt.title('loss')
    plt.plot(losses)
    plt.savefig('img/Double_DQN_CartPole_%s.png' % (frame_idx))
    plt.cla()
    plt.close("all")


### Training CartPole ###
num_frames = 40000
batch_size = 32
gamma = 0.99

losses = []
all_rewards = []
episode_reward = 0

state = env.reset()
for frame_idx in range(1, num_frames + 1):
    epsilon = epsilon_by_frame(frame_idx)
    action = current_model.act(state, epsilon)
    
    next_state, reward, done, _ = env.step(action)
    replay_buffer.push(state, action, reward, next_state, done)
    
    state = next_state
    episode_reward += reward
    
    if done:
        state = env.reset()
        all_rewards.append(episode_reward)
        episode_reward = 0
        
    if len(replay_buffer) > batch_size:
        loss = compute_td_loss(batch_size)
        losses.append(loss.item())
        
    if frame_idx % 1000 == 0:
        CartPole_plot(frame_idx, all_rewards, losses)
        if frame_idx > 1000:
            os.system('rm img/Double_DQN_CartPole_%s.png' % (frame_idx - 1000))
        
    if frame_idx % 100 == 0:
        update_target(current_model, target_model)


### Atari Environment ###
from common.wrappers import make_atari, wrap_deepmind, wrap_pytorch

env_id = "PongNoFrameskip-v4"
env = make_atari(env_id)
env = wrap_deepmind(env)
env = wrap_pytorch(env)

class CnnDQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(CnnDQN, self).__init__()
        
        self.input_shape = input_shape
        self.num_actions = num_actions
        
        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def feature_size(self):
        return self.features(Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)
    
    def act(self, state, epsilon):
        if random.random() > epsilon:
            with torch.no_grad():
                state = Variable(torch.FloatTensor(np.float32(state)).unsqueeze(0)).to(device)
            q_value = self.forward(state)
            action = int(q_value.max(1)[1].data[0].cpu().int().numpy())
        else:
            action = random.randrange(env.action_space.n)
        return action

def Atari_plot(frame_idx, rewards, losses):
    clear_output(True)
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, np.mean(rewards[-10:])))
    plt.plot(rewards)
    plt.subplot(132)
    plt.title('loss')
    plt.plot(losses)
    plt.savefig('img/Double_DQN_Atari_%s.png' % (frame_idx))
    plt.cla()
    plt.close("all")

current_model = CnnDQN(env.observation_space.shape, env.action_space.n).to(device)
target_model  = CnnDQN(env.observation_space.shape, env.action_space.n).to(device)
    
optimizer = optim.Adam(current_model.parameters(), lr=0.0001)

replay_initial = 10000
replay_buffer = ReplayBuffer(100000)

update_target(current_model, target_model)

epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 30000

epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)

# plt.plot([epsilon_by_frame(i) for i in range(1000000)])


### Training Atari ###
num_frames = 2000000
batch_size = 32
gamma = 0.99

losses = []
all_rewards = []
episode_reward = 0

state = env.reset()
for frame_idx in range(1, num_frames + 1):
    epsilon = epsilon_by_frame(frame_idx)
    action = current_model.act(state, epsilon)
    
    next_state, reward, done, _ = env.step(action)
    replay_buffer.push(state, action, reward, next_state, done)
    
    state = next_state
    episode_reward += reward
    
    if done:
        state = env.reset()
        all_rewards.append(episode_reward)
        episode_reward = 0
        
    if len(replay_buffer) > replay_initial:
        loss = compute_td_loss(batch_size)
        losses.append(loss.item())
        
    if frame_idx % 10000 == 0:
        Atari_plot(frame_idx, all_rewards, losses)
        if frame_idx > 10000:
            os.system('rm img/Double_DQN_Atari_%s.png' % (frame_idx - 10000))
        
    if frame_idx % 1000 == 0:
        update_target(current_model, target_model)