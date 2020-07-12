# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 14:31:36 2020

@author: lenovo
"""

import pandas as pd
from dqn import *
import numpy as np
from collections import deque
from keras.layers import Dense, Conv1D, Flatten
from keras.models import Sequential
from keras.preprocessing.sequence import TimeseriesGenerator



class stockBroker():
    
    def __init__(self, balance, stockPrices, scale):
        self.initBalance = balance
        self.balance = balance
        self.stockPrices = stockPrices
        self.currIndex = 0
        self.currNStock = 0
        self.scale = scale
        
        self.look_back = 5
        
        self.X = self.stockPrices/self.scale
        self.X = np.array(self.X).reshape(len(self.X),1)
        
    def getStockPrices(self):
        return self.stockPrices
    
    def getCurrPrice(self):
        price = self.stockPrices[self.currIndex]
        return price
    
    def getScaledPastPrice(self):
        prices = self.stockPrices[self.currIndex:self.currIndex+self.look_back]
        prices = prices / self.scale
        return prices.reshape(1,self.look_back,1)
    
    def buyStock(self):
        n = self.balance//self.getCurrPrice()
        self.balance -= n * self.getCurrPrice()
        self.currNStock += n
    
    def sellStock(self):
        self.balance += self.currNStock * self.getCurrPrice()
        self.currNStock = 0
    
    def reset(self):
        self.currIndex = 0
        self.currNStock = 0
        self.balance = self.initBalance
        state = ([self.getCurrPrice() / self.scale,\
                  self.initBalance / self.scale, 0, 0])
        return state
    
    def train_model(self, epochs = 100):
    
        look_back = 5
        
        self.model = Sequential()
        self.model.add(Conv1D(32,input_shape=(look_back,1),
                         kernel_size=5,strides=1,padding="valid"))
        self.model.add(Flatten())
        self.model.add(Dense(1))
        
        self.model.compile(optimizer="adam", loss="mse",metrics=["mse"])
        
        generator = TimeseriesGenerator(self.X, self.X,
                                       length=look_back, sampling_rate=1,
                                       batch_size=1)
        
        self.model.fit_generator(generator,epochs=epochs)
        
        self.model.save("prediction.h5")
    
    
    def step(self, action):
        
        #hold sell by (0 1 2)
        if action == 1:
            self.buyStock()
        elif action == 2:
            self.sellStock()
        
        done = True if self.currIndex == (len(self.stockPrices) - 1) else False
        
        equity = (self.currNStock * self.getCurrPrice()) / self.scale
        reward = (self.balance + equity) / self.scale
        
        prediction = 0
        try:
            past_price = self.getScaledPastPrice()
            prediction = self.model.predict(past_price)[0][0]
        except:
            pass

        state = ([self.getCurrPrice() / self.scale,\
                  self.balance / self.scale, equity, prediction], reward, done)
        self.currIndex += 1

        
        return state


    


win_trials = 100

state_size = 4
action_size = 3

agent = DQNAgent(state_size, action_size)

# should be solved in this number of episodes
episode_count = 2000
batch_size = 64

scale = 10000

df = pd.read_csv("AAPL.csv")

close_price = df["Close"]


scores = deque(maxlen=win_trials)

env = stockBroker(5000, close_price, scale)
env.train_model(10)

# Q-Learning sampling and fitting
for episode in range(episode_count):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    done = False
    total_reward = 0
    while not done:
        # in CartPole-v0, action=0 is left and action=1 is right
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        # state = [pos, vel, theta, angular speed]
        next_state = np.reshape(next_state, [1, state_size])
        # store every experience unit in replay buffer
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward = reward * scale

    # call experience relay
    if len(agent.memory) >= batch_size:
        agent.replay(batch_size)

    scores.append(total_reward)
    mean_score = np.mean(scores)
    if mean_score >= 10000 \
            and episode >= win_trials:
        print("Mean assets = %0.2lf in %d episodes"
              % (episode, mean_score, win_trials))
        print("Epsilon: ", agent.epsilon)
        agent.save_weights()
        break
    if (episode + 1) % win_trials == 0:
        print("Episode %d: Mean assets = \
               %0.2lf in %d episodes" %
              ((episode + 1), mean_score, win_trials))


