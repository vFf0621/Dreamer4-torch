import torch
import numpy as np
import cv2
import gymnasium as gym
import gym_multi_car_racing
from models import Dreamer4
from buffer import ReplayBuffer # Ensure your updated buffer is in buffer.py
from tqdm import tqdm
import wandb
import os
import argparse

# --- Setup ---
wandb.init(project="Dreamer4", entity="fguan", name="test")
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
# Training parameters
import os
os.makedirs("./ckpts", exist_ok=True)
os.makedirs("./eval_imgs", exist_ok=True)

def simulate(env, num_warmups, num_interaction_episodes,num_agents, ch, h, w, patch , Nr, latent_tokens, z_dim, action_dim, latent_dim, 
                 rep_depth , rep_d_model, dyn_d_model, num_heads, dropout, k_max, mtp, task_id,  kmax_prob,
                 policy_bins , reward_bins , pretrain, reward_clamp,level_vocab , level_embed_dim,mode,num_tasks, Sa,
                 batch_lens, batch_size, accum, max_imag_len, buffer_limit, train, ckpt, rep_lr=1e-4, rep_decay=1e-3,eval_context_len=15,
                 dyn_lr=1e-4, dyn_decay=1e-3, ac_lr = 1e-4, ac_decay=1e-3, policy_lr=1e-4, policy_decay=1e-3 , save_every=500):
    agents = [Dreamer4(agent_id=i, ch=ch, h=h,
                w=w, 
                patch = patch, 
                latent_tokens=latent_tokens, 
                z_dim=z_dim,
                action_dim=action_dim, 
                latent_dim=latent_dim, 
                rep_depth = rep_depth, 
                rep_d_model=rep_d_model, 
                dyn_d_model=dyn_d_model, 
                num_heads=num_heads, 
                dropout=dropout, 
                k_max=k_max, 
                Sa = Sa, 
                Nr = Nr, 
                kmax_prob=kmax_prob,
                eval_context_len=eval_context_len,
                mtp=mtp, 
                num_tasks=num_tasks,
                policy_bins = policy_bins, 
                reward_bins = reward_bins, 
                pretrain=pretrain, 
                reward_clamp=reward_clamp,
                level_vocab = level_vocab, 
                level_embed_dim = level_embed_dim,
                batch_lens = batch_lens, 
                batch_size=batch_size, 
                accum=accum, 
                max_imag_len=max_imag_len, 
                ckpt=ckpt, 
                dyn_lr=dyn_lr,
                task_id = task_id,
                rep_lr=rep_lr, 
                rep_decay=rep_decay,
                dyn_decay=dyn_decay,
                ac_lr=ac_lr,
                ac_decay=ac_decay,
                policy_lr=policy_lr, 
                policy_decay=policy_decay) for i in range(num_agents)]
    # 1. Initialize Buffer OUTSIDE the loop so data persists
    buffer = ReplayBuffer(buffer_limit=buffer_limit, obs_size=(ch, h,w), action_size=action_dim )
    total_steps = 0
    writer = {}
    for epi in range(num_warmups + num_interaction_episodes):
        # --- Episode Initialization ---
        observation, _ = env.reset()
        done = np.zeros(1, dtype=bool)
        score = 0
        
        # Temporary lists to store the FULL episode
        episode_obs = []
        if mode == "pretrain":
            train_reward=False
            train_model =False#(epi > 0) and (epi <= 1200) 
            train_policy = False#(epi > 900)
        elif mode=="dyn":
            train_reward=True
            train_model =True#(epi > 0) and (epi <= 1200) 
            train_policy = False#(epi > 900)
  
        elif mode=="policy":
            train_reward=False
            train_model =False#(epi > 0) and (epi <= 1200) 
            train_policy = True#(epi > 900)
        elif mode=="inference":
            train_reward=False
            train_model =False#(epi > 0) and (epi <= 1200) 
            train_policy = False#(epi > 900)

        else:
            raise AssertionError("Unknown training mode. Acceptable modes are: {pretrain, dyn, finetune_bc, policy, inference}")
        step = 0
        # --- Collection Phase ---

        if not buffer.full:
            while (not done.all()):
                # Handle Actions
                act = [None] # Single agent
                for i in range(len(agents)):
                # Warmup: Random Action
                    if not train_policy and epi < num_warmups:
                        act[i] = env.action_space.sample()
                    # Training: Model Action
                    else:
                        agents[i].encoder.eval()
                        # Assuming single agent, extracting index 0
                        state_tensor = torch.from_numpy(observation[0]).to(agents[0].device)
                        act[i] = agents[i].action_step(state_tensor)
                    # Step Environment
                # gym_multi_car_racing usually returns dicts or tuples depending on version
                # ensuring compatibility with your previous unpacking
                next_observation, reward, done, info, _ = env.step(act)
                step += 1
                if train:
                    # Accumulate data for this step (Agent 0)
                    for i in range(len(agents)):
                       if observation[i] is not None:
                            buffer.add(observation[i], act[i], reward[i], next_observation[i], done[i])
                score += reward[0]
                observation = next_observation
            
            # Rendering (optional, can slow down training)
            # env.render() 
        # --- Storage Phase ---
        if train and len(episode_obs) > 0:
            # Convert lists to numpy arrays for the buffer
            # Expected shapes: (T, H, W, C), (T, A), (T, 1), (T, 1)
            total_steps += len(episode_obs)

        # --- Training Phase ---
        # Determine training mode
        is_warmup = epi < num_warmups
        
        # Logic from your script: 
        # Train model only for first 900 eps? Or strictly < warmups?
        # Adjusted based on your snippet:
     
        print(f"Episode {epi} | Score: {score:.2f} | Buffer Size: {len(buffer)} | Steps: {total_steps}")
        writer["episodic_return_0"] = score
        
        # Train on the buffer
        if train and buffer.full: # Ensure min buffer size
            for a in agents:
                # Note: buffer is passed directly; train_one_epoch handles sampling internally
                log_data = a.train_step(writer, buffer, model=train_model, policy=train_policy, train_reward=train_reward)
                wandb.log(data=log_data)
        if epi%save_every==0:
        # --- Checkpointing ---
            print(">>> Saving Parameters <<<")
            for i in range(len(agents)):
                agents[i].save_checkpoint(f"WM_{epi}.pt")
                agents[i].save_rep(f"WM_{epi}.pt")
               # agents[i].model.save_rep(f"WM_{epi}.pt")

                # Ensure evaluate is safe to call
                agents[i].evaluate(buffer)

        # Reset agent internal state (RNN hidden states) for next episode
        if not is_warmup:
             for i in range(1):
                agents[i].reset()
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run simulate() with configurable hyperparameters.")

    # Positional-style inputs (you likely provide these in code, not CLI)
    p.add_argument("--steps", type=int, default=1_000_000, help="Total environment steps.")
    p.add_argument("--seed", type=int, default=0, help="Random seed.")
    p.add_argument("--config", type=str, default="", help="Optional path to a JSON/YAML config (not used unless you load it).")

    # simulate kwargs
    p.add_argument("--num_agents", type=int, default=1)
    p.add_argument("--buffer_limit", type=int, default=12000)

    # Observation / encoding
    p.add_argument("--ch", type=int, default=3, help="Image Channels")
    p.add_argument("--h", type=int, default=96, help="Image Height")
    p.add_argument("--w", type=int, default=96, help="Image Width")
    p.add_argument("--patch", type=int, default=8, help="Patch Size")
    p.add_argument("--latent_tokens", type=int, default=64, help="Nz")
    p.add_argument("--reserved_tokens", type=int, default=4, help="Nr")

    p.add_argument("--z_dim", type=int, default=16, help="Bottleneck")
    p.add_argument("--action_dim", type=int, default=2)
    p.add_argument("--kmax_base_prob", type=int, default=0.5)
    # Model sizes
    p.add_argument("--Sa", type=int, default=4)

    p.add_argument("--pred_dim", type=int, default=256)
    p.add_argument("--rep_depth", type=int, default=4, help="Has to be a multiple of 2")
    p.add_argument("--rep_d_model", type=int, default=256)
    p.add_argument("--dyn_d_model", type=int, default=256)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.01)
    p.add_argument("--k_max", type=int, default=32, help="Has to be a power of 2")
    p.add_argument("--mtp", type=int, default=7)
    p.add_argument("--num_tasks", type=int, default=10)
    p.add_argument("--task_id", type=int, default=0)
    p.add_argument("--eval_context_len", type=int, default=15)

    # Discretization / vocab
    p.add_argument("--policy_bins", type=int, default=100)
    p.add_argument("--reward_bins", type=int, default=100)
    p.add_argument("--reward_clamp_abs", type=float, default=6)
    p.add_argument("--level_vocab", type=int, default=129)
    p.add_argument("--level_embed_dim", type=int, default=256)
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--dyn_lr", type=float, default=1e-4)
    p.add_argument("--dyn_decay", type=float, default=1e-4)
    p.add_argument("--rep_lr", type=float, default=1e-4)
    p.add_argument("--rep_decay", type=float, default=1e-3)
    p.add_argument("--policy_lr", type=float, default=1e-4)
    p.add_argument("--ac_lr", type=float, default=1e-4)
    p.add_argument("--policy_decay", type=float, default=1e-3)
    p.add_argument("--ac_decay", type=float, default=1e-3)
    p.add_argument("--render_mode", type=str, default="rgb_array")
    p.add_argument("--train_mode", type=str, default="pretrain")
    # Training
    p.add_argument(
        "--batch_lens",
        type=int,
        nargs=2,
        default=(45, 65),
        metavar=("MIN_LEN", "MAX_LEN"),
        help="Batch length range, e.g. --batch_lens 45 65",
    )
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--accum", type=int, default=5)
    p.add_argument("--max_imag_len", type=int, default=12008)
    # For memory considerations

    # Flags (default True -> allow toggling off with --no-*)
    p.add_argument("--train", dest="train", action="store_true", default=True)
    p.add_argument("--no-train", dest="train", action="store_false", help="Disable training mode.")
    p.add_argument("--pretrain", dest="pretrain", action="store_true", default=True)
    p.add_argument("--no-pretrain", dest="pretrain", action="store_false", help="Disable pretraining mode.")
    p.add_argument("--save_every", type=int, default=1000)

    return p

def main():
    parser = build_parser()
    args = parser.parse_args()
    # If you actually want to load args.config, do it here and override args.

    # You said simulate(agents, env, 1000000, 0, {}, ...)
    # We'll keep agents/env as variables you set up in code.

    # Ensure environment is created correctly
    env = gym.make("MultiCarRacing-v1", num_agents=args.num_agents, render_mode=args.render_mode) 
    simulate(
        env,
        1000000 if args.train_mode != "inference" else 1,
        args.steps,  # your empty dict argument stays as-is
        num_agents=args.num_agents,
        buffer_limit=args.buffer_limit,
        ch=args.ch,
        h=args.h,
        w=args.w,
        patch=args.patch,
        kmax_prob = args.kmax_base_prob,
        latent_tokens=args.latent_tokens,
        z_dim=args.z_dim,
        eval_context_len = args.eval_context_len,
        action_dim=args.action_dim,
        latent_dim=args.pred_dim,
        rep_depth=args.rep_depth,
        rep_d_model=args.rep_d_model,
        dyn_d_model=args.dyn_d_model,
        num_heads=args.num_heads,
        dropout=args.dropout,
        k_max=args.k_max,
        mtp=args.mtp,
        num_tasks=args.num_tasks,
        policy_bins=args.policy_bins,
        reward_bins=args.reward_bins,
        reward_clamp=args.reward_clamp_abs,
        level_vocab=args.level_vocab,
        level_embed_dim=args.level_embed_dim,
        batch_lens=tuple(args.batch_lens),
        batch_size=args.batch_size,
        Sa=args.Sa,
        Nr=args.reserved_tokens,    
        save_every=args.save_every,

        accum=args.accum,
        max_imag_len=args.max_imag_len,
        train=args.train,
        task_id=args.task_id,
        pretrain=args.pretrain,
        ckpt=args.ckpt,
        dyn_lr=args.dyn_lr,
        rep_lr=args.rep_lr, 
        rep_decay=args.rep_decay,
        dyn_decay=args.dyn_decay,
        ac_lr=args.ac_lr,
        ac_decay=args.ac_decay,
        policy_lr=args.policy_lr, 
        policy_decay=args.policy_decay,
       mode = args.train_mode
    )

if __name__ == "__main__":
    main()