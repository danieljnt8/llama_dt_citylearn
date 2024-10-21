"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the CC BY-NC license found in the
LICENSE.md file in the root directory of this source tree.
"""

from torch.utils.tensorboard import SummaryWriter
import argparse
import pickle
import random
import time
import gym

import torch
import numpy as np

from datasets import load_from_disk
import datasets
from datasets import Dataset
import utils
from lamb import Lamb
#from stable_baselines3.common.vec_env import SubprocVecEnv
from pathlib import Path
from data import create_dataloader
from llama.decision_transformer_llama import DecisionTransformerLlama
#from evaluation import create_vec_eval_episodes_fn, vec_evaluate_episode_rtg
#from evaluation_citylearn import create_eval_episodes_fn_ori,evaluate_episode_rtg_ori,create_test_episodes_fn_ori,test_episode_rtg_ori
from trainer import SequenceTrainerLlama
from logger import Logger
import pandas as pd

from utils_.helpers import *


from utils_.variant_dict import variant

MAX_EPISODE_LEN = 8760


def update_loss_csv(iter_value, loss, filename='loss_per_epoch.csv',type_name="Epoch"):
    # Try to read the existing CSV file
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        # If file does not exist, create a new DataFrame with headers
        df = pd.DataFrame(columns=[type_name, 'Loss'])
    
    # Append the new data to the DataFrame
    new_row = {type_name: iter_value, 'Loss': loss}
    df = df.append(new_row, ignore_index=True)
    
    # Write the updated DataFrame back to the CSV file
    df.to_csv(filename, index=False)
    
class Experiment:
    def __init__(self, variant,dataset_path):
        self.variant = variant
        self.device = variant["device"]
         
         
        self.state_dim = 32
        self.act_dim = 1
        self.action_range= [-1,1]
        
        self.initial_trajectories = self._get_initial_trajectories(dataset_path=dataset_path)



        self.offline_trajs, self.state_mean, self.state_std = self._load_dataset(
            self.initial_trajectories
        )

        print("state_mean "+str(self.state_mean))
        print("state_std "+ str(self.state_std ))
        
        self.model = DecisionTransformerLlama(
            state_dim=self.state_dim,
            act_dim=self.act_dim,
            action_range=self.action_range,
            max_length=variant["K"],
            eval_context_length=variant["eval_context_length"],
            max_ep_len=MAX_EPISODE_LEN,
            hidden_size=variant["embed_dim"],
            n_layer=variant["n_layer"],
            n_head=variant["n_head"],
        ).to(device=self.device)
        
        self.optimizer = torch.optim.Adam(
        self.model.parameters(), 
        weight_decay=1e-4, 
        eps=1e-8, 
        lr=1e-4
        )
        
        self.pretrain_iter = 0
        
        self.reward_scale = 1.0 
        self.logger = Logger(variant)
        self.train()
        
    def _get_initial_trajectories(self,dataset_path):
        dataset = load_from_disk(dataset_path)
        dataset,_ = segment_v2(dataset["observations"],dataset["actions"],dataset["rewards"],dataset["dones"])
        trajectories = datasets.Dataset.from_dict({k: [s[k] for s in dataset] for k in dataset[0].keys()})

        return trajectories
   
    def _get_env_spec(self,env):
        state_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        action_range = [
                -1,#float(env.action_space.low.min()) ,
                1#float(env.action_space.high.max()) ,
            ]
        return state_dim,act_dim, action_range
    
    def _load_dataset(self,trajectories):
        states, traj_lens, returns = [], [], []
        for path in trajectories:
            states.append(path["observations"])
            traj_lens.append(len(path["observations"]))
            returns.append(np.array(path["rewards"]).sum())
        traj_lens, returns = np.array(traj_lens), np.array(returns)

            # used for input normalization
        states = np.concatenate(states, axis=0)
        state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
        num_timesteps = sum(traj_lens)

        print("=" * 50)
        print(f"Starting new experiment: city_learn")
        print(f"{len(traj_lens)} trajectories, {num_timesteps} timesteps found")
        print(f"Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}")
        print(f"Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}")
        print(f"Average length: {np.mean(traj_lens):.2f}, std: {np.std(traj_lens):.2f}")
        print(f"Max length: {np.max(traj_lens):.2f}, min: {np.min(traj_lens):.2f}")
        print("=" * 50)

        sorted_inds = np.argsort(returns)  # lowest to highest
        num_trajectories = 1
        timesteps = traj_lens[sorted_inds[-1]]
        ind = len(trajectories) - 2
        while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] <= num_timesteps:
            timesteps += traj_lens[sorted_inds[ind]]
            num_trajectories += 1
            ind -= 1
        sorted_inds = sorted_inds[-num_trajectories:]
        print(sorted_inds)
        #print(trajectories[1])
        for ii in sorted_inds:
            print(ii)
        #print(trajectories[0].keys())
        trajectories = [trajectories[int(ii)] for ii in sorted_inds]

        for trajectory in trajectories:
            for key in trajectory.keys():
                trajectory[key] = np.array(trajectory[key])


        return trajectories, state_mean, state_std
    
    def _save_model(self, path_prefix, is_pretrain_model=False,iteration=0):
        to_save = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "pretrain_iter": self.pretrain_iter,
            "args": self.variant,
            "np": np.random.get_state(),
            "python": random.getstate(),
            "pytorch": torch.get_rng_state(),
        }

        if is_pretrain_model:
            with open(f"{path_prefix}/pretrain_model.pt", "wb") as f:
                torch.save(to_save, f)
            print(f"Model saved at {path_prefix}/pretrain_model.pt")
        else:
            with open(f"{path_prefix}/model_{iteration}.pt", "wb") as f:
                torch.save(to_save, f)
            print(f"\nModel saved at {path_prefix}/model_{iteration}.pt")

    
    def train(self):
        print("\n\n\n*** Pretrain ***")
        
        trainer = SequenceTrainerLlama(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=None,
            device=self.device,
        )
        
        writer = (
            SummaryWriter(self.logger.log_path) if self.variant["log_to_tb"] else None
        )
        
        while self.pretrain_iter < self.variant["max_pretrain_iters"]:
            # in every iteration, prepare the data loader
            dataloader = create_dataloader(
                trajectories=self.offline_trajs,
                num_iters=self.variant["num_updates_per_pretrain_iter"],
                batch_size=self.variant["batch_size"],
                max_len=self.variant["K"],
                state_dim=self.state_dim,
                act_dim=self.act_dim,
                state_mean=self.state_mean,
                state_std=self.state_std,
                reward_scale=self.reward_scale,
                action_range=self.action_range,
            )
            
            train_outputs = trainer.train_iteration(
                dataloader=dataloader,
            )

            update_loss_csv(self.pretrain_iter,train_outputs["training/train_loss_mean"],filename=self.logger.log_path+'/loss_per_epoch_pretrain.csv'
                            ,type_name="Iteration")
            
            if self.pretrain_iter % 10 == 0 : 
                self._save_model(
                    path_prefix=self.logger.log_path,
                    is_pretrain_model=False,
                    iteration=self.pretrain_iter
                )

            if self.pretrain_iter - 1 == self.variant["max_pretrain_iters"]:
                self._save_model(
                    path_prefix=self.logger.log_path,
                    is_pretrain_model=False,
                    iteration = self.pretrain_iter
                )
                
            self.pretrain_iter += 1
        
        
        

        
def run_experiment(seed):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=seed)
    #parser.add_argument("--env", type=str, default="hopper-medium-v2")

    # model options
    parser.add_argument("--K", type=int, default=24)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--eval_context_length", type=int, default=20)
    

    # shared evaluation options
    parser.add_argument("--eval_rtg", type=int, default=-6000)
    parser.add_argument("--num_eval_episodes", type=int, default=1)

    # shared training options
    parser.add_argument("--init_temperature", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", "-wd", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=1)

    # pretraining options
    parser.add_argument("--max_pretrain_iters", type=int, default=41)
    parser.add_argument("--num_updates_per_pretrain_iter", type=int, default=100) #30

    # environment options
    parser.add_argument("--device", type=str, default="cuda") ##cuda 
    parser.add_argument("--log_to_tb", "-w", type=bool, default=True)
    parser.add_argument("--save_dir", type=str, default="./exp_4")
    parser.add_argument("--exp_name", type=str, default="default")

    args = parser.parse_args()

    utils.set_seed_everywhere(args.seed)
    experiment = Experiment(vars(args), dataset_path="data_interactions/winner_dataset_phase_1.pkl")
   
    print("=" * 50)
    

if __name__ == "__main__":
    seeds = [53728, 12345, 67890, 54321, 98765]
    for seed in seeds:
        run_experiment(seed)