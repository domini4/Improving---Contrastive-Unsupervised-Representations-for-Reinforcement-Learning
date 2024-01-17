# -*- coding: utf-8 -*-
# MIT License
#
# Copyright (c) 2017 Kai Arulkumaran
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# ==============================================================================
from collections import deque
import random
#import atari_py
import cv2
import torch
import gym
from skimage import color
from skimage import io
import os
from PIL import Image


class Env():
  def __init__(self, args):
    self.device = args.device
    #self.ale = atari_py.ALEInterface()
    #self.ale.setInt('random_seed', args.seed)
    #self.ale.setInt('max_num_frames_per_episode', args.max_episode_length)
    #self.ale.setFloat('repeat_action_probability', 0)  # Disable sticky actions
    #self.ale.setInt('frame_skip', 0)
    #self.ale.setBool('color_averaging', False)
    #self.ale.loadROM(atari_py.get_game_path(args.game))  # ROM loading must be done after setting options
    #actions = self.ale.getMinimalActionSet()
    # setting environment varibales for each environment
    if args.game == "CartPole-v1":
        actions_new = [0, 1]
        self.env_name = "CartPole-v1"
    else:
        actions_new = [0, 1, 2]
        self.env_name = "MountainCar-v0"
    #self.actions = dict([i, e] for i, e in zip(range(len(actions)), actions))
    self.actions_new = dict([i, e] for i, e in zip(range(len(actions_new)), actions_new))
#    self.lives = 0  # Life counter (used in DeepMind training)
#    self.life_termination = False  # Used to check if resetting only from loss of life
    self.window = args.history_length  # Number of frames to concatenate
    self.state_buffer = deque([], maxlen=args.history_length)
    self.training = True  # Consistent with model training mode
    self.env1 = gym.make(self.env_name)
    self.env1.reset()

  def _get_state(self):
    #state = cv2.resize(self.ale.getScreenGrayscale(), (84, 84), interpolation=cv2.INTER_LINEAR)
    img = self.env1.render(mode='rgb_array')
    state = cv2.resize(color.rgb2gray(img), (126, 84), interpolation = cv2.INTER_CUBIC)
    return torch.tensor(state, dtype=torch.float32, device=self.device)
    #return torch.tensor(state, dtype=torch.float32, device=self.device).div_(255)

  def _get_state_evaluate(self, results_dir, T, _, step_count, t):
    #state = cv2.resize(self.ale.getScreenGrayscale(), (84, 84), interpolation=cv2.INTER_LINEAR)
    img = self.env1.render(mode='rgb_array')
    im = Image.fromarray(img)
    im.save(os.path.join(results_dir, "test" + str(T) + "_" + str(_) + str(step_count) + "_" + str(t)) + ".jpeg")
    state = cv2.resize(color.rgb2gray(img), (126, 84), interpolation = cv2.INTER_CUBIC)
    return torch.tensor(state, dtype=torch.float32, device=self.device)
    #return torch.tensor(state, dtype=torch.float32, device=self.device).div_(255)

  def _reset_buffer(self):
    for _ in range(self.window):
      self.state_buffer.append(torch.zeros(84, 126, device=self.device))
      #self.state_buffer.append(torch.zeros(84, 84, device=self.device))

  def reset(self):
#    if self.life_termination:
#      self.life_termination = False  # Reset flag
#      self.ale.act(0)  # Use a no-op after loss of life
#    else:
      # Reset internals
#      self._reset_buffer()
#      self.ale.reset_game()
      # Perform up to 30 random no-ops before starting
#      for _ in range(random.randrange(30)):
#        self.ale.act(0)  # Assumes raw action 0 is always no-op
#        if self.ale.game_over():
#          self.ale.reset_game()
    # Process and return "initial" state
    #new code for classic control
    self._reset_buffer()
    self.env1.reset()
    #end new code for classic control
    observation = self._get_state()
    #torch.set_printoptions(threshold=10_000)
    self.state_buffer.append(observation)
#    self.lives = self.ale.lives() # do not need for gym
    return torch.stack(list(self.state_buffer), 0)

  def evaluate(self, action, results_dir, T, _, step_count):
    # Repeat action 4 times, max pool over last 2 frames
    frame_buffer = torch.zeros(2, 84, 126, device=self.device)
    reward, done = 0, False
    for t in range(4):
      new_state, reward_s, is_terminal, info = self.env1.step(action)
      reward += reward_s
      if t == 2:
        frame_buffer[0] = self._get_state_evaluate(results_dir, T, _, step_count, t)
      elif t == 3:
        frame_buffer[1] = self._get_state_evaluate(results_dir, T, _, step_count, t)
      done = is_terminal
      if done:
        break
    observation = frame_buffer.max(0)[0]
    self.state_buffer.append(observation)
    return torch.stack(list(self.state_buffer), 0), reward, done

  def step(self, action):
    # Repeat action 4 times, max pool over last 2 frames
    frame_buffer = torch.zeros(2, 84, 126, device=self.device)
    #frame_buffer = torch.zeros(2, 84, 84, device=self.device)
    reward, done = 0, False
#    for t in range(4):
#      reward += self.ale.act(self.actions.get(action)) # pass in the action to get reward
#      if t == 2:
#        frame_buffer[0] = self._get_state()
#      elif t == 3:
#        frame_buffer[1] = self._get_state()
#      done = self.ale.game_over() # done or not
#      if done:
#        break
    for t in range(4):
      new_state, reward_s, is_terminal, info = self.env1.step(action)
      reward += reward_s
      #reward += self.ale.act(self.actions.get(action)) # pass in the action to get reward
      if t == 2:
        frame_buffer[0] = self._get_state()
      elif t == 3:
        frame_buffer[1] = self._get_state()
      done = is_terminal
      #done = self.ale.game_over() # done or not
      if done:
        break
    observation = frame_buffer.max(0)[0]
    self.state_buffer.append(observation)
    # Detect loss of life as terminal in training mode
    # no need to set lives in classic control
#    if self.training:
#      lives = self.ale.lives() # no need in gym
#      if lives < self.lives and lives > 0:  # Lives > 0 for Q*bert # only for DMControl
#        self.life_termination = not done  # Only set flag when not truly done
#        done = True
#      self.lives = lives
    # Return state, reward, done
    return torch.stack(list(self.state_buffer), 0), reward, done

  # Uses loss of life as terminal signal
  def train(self):
    self.training = True

  # Uses standard terminal signal
  def eval(self):
    self.training = False

  def action_space(self):
    return len(self.actions_new)
    #return len(self.actions)

  def get_sample(self):
    return self.env1.action_space.sample()

  def render(self):
    cv2.imshow('screen', self.ale.getScreenRGB()[:, :, ::-1])
    cv2.waitKey(1)

  def close(self):
    cv2.destroyAllWindows()
