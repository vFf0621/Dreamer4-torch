import torch
import numpy as np
import cv2
import gymnasium as gym
import mani_skill.envs
from models import Dreamer4
from buffer import ReplayBuffer # Ensure your updated buffer is in buffer.py
from tqdm import tqdm
import wandb
import os
import argparse
import multiview_pushcube  # registers PushCubeMultiView-v1
# --- Setup ---
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
# Training parameters
import os
os.makedirs("./ckpts", exist_ok=True)
os.makedirs("./eval_imgs", exist_ok=True)
import os
import numpy as np
def _to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return x

def chw01_to_bgr_uint8(x_chw: np.ndarray) -> np.ndarray:
    x = np.clip(x_chw, 0.0, 1.0)
    x = (x * 255.0).astype(np.uint8)

    x = x.transpose(1, 2, 0)   # CHW -> HWC (RGB)

    x = x[..., ::-1]            # RGB -> BGR
    return x


def extract_input_and_target(obs: dict):
    """
    Adapts for standard PushCube-v1. 
    Input: Base Camera (Stationary)
    Target: Hand Camera (Moving)
    """
    sd = obs["sensor_data"]
    
    # Standard ManiSkill Camera Keys
    base_rgb = _to_numpy(sd["left_view"]["rgb"])

    if base_rgb.ndim == 4: base_rgb = base_rgb[0]

    def to_chw01(img):
        if img.dtype != np.uint8:
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        return img.transpose(2, 0, 1).astype(np.float32) / 255.0

    input_obs = to_chw01(base_rgb)
    return input_obs
def show_windows(input_obs: np.ndarray, wait_ms=1):
    """
    input_obs: (3,H,W) random view
    targ: (6,H,W) where [0:3]=top, [3:6]=left
    Returns key code (press q to stop rendering).
    """
    if cv2 is None:
        raise ImportError("OpenCV not installed. Install: pip install opencv-python")

    rnd = chw01_to_bgr_uint8(input_obs)

    cv2.imshow("random_view (Input)", rnd)

    key = cv2.waitKey(wait_ms) & 0xFF
    return key

def load_replay(
    path: str,
    *,
    seed=None,
    mmap: bool = False,
    allow_pickle: bool = True,
    strict: bool = True,
):
    """
    Load a ReplayBuffer saved by ReplayBuffer.save() from a .npz file.

    Args:
        path: .npz path created by ReplayBuffer.save()
        seed: RNG seed to set on the loaded buffer
        mmap: if True, uses mmap_mode="r" (works best with uncompressed .npz/.npy; with
              savez_compressed it may not meaningfully mmap)
        allow_pickle: needed only if you saved meta (object). If you never save meta,
                      you can set False for extra safety.
        strict: if True, validates shapes/dtypes and raises on mismatch. If False,
                loads best-effort.

    Returns:
        (buf, meta) where meta is the dict saved under "meta" (or None).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    load_kwargs = dict(allow_pickle=allow_pickle)
    if mmap:
        load_kwargs["mmap_mode"] = "r"

    with np.load(path, **load_kwargs) as data:
        required = [
            "input_obs", "target_obs", "action", "reward", "terminal", 
            "next_idx", "write_id", "next_write_id", "size",
        ]
        missing = [k for k in required if k not in data.files]
        if missing:
            raise ValueError(f"Missing keys in npz: {missing}. Found: {data.files}")

        input_obs = data["input_obs"]
        target_obs = data["target_obs"]
        action = data["action"]
        reward = data["reward"]
        terminal = data["terminal"]
        next_idx = data["next_idx"]
        write_id = data["write_id"]
        next_write_id = data["next_write_id"]
        n = int(np.asarray(data["size"]).reshape(-1)[0])

        meta = None
        if "meta" in data.files:
            # saved as np.array([meta], dtype=object)
            meta_arr = data["meta"]
            if meta_arr.size:
                meta = meta_arr.reshape(-1)[0].item() if hasattr(meta_arr.reshape(-1)[0], "item") else meta_arr.reshape(-1)[0]

    # Basic inferred config from arrays
    buf_limit = int(n)
    in_shape = tuple(input_obs.shape[1:])
    tgt_shape = tuple(target_obs.shape[1:])
    act_size = int(action.shape[1])

    # Instantiate a new buffer with exact capacity == saved size (not original limit)
    buf = ReplayBuffer(
        buffer_limit=buf_limit,
        input_obs_size=in_shape,
        target_obs_size=tgt_shape,
        action_size=act_size,
        obs_dtype=input_obs.dtype,
        seed=seed,
    )

    # Fill internal arrays (exactly n)
    if strict:
        def _chk(name, arr, ndim=None, last_dims=None):
            if ndim is not None and arr.ndim != ndim:
                raise ValueError(f"{name}.ndim={arr.ndim} expected {ndim}")
            if last_dims is not None and tuple(arr.shape[-len(last_dims):]) != tuple(last_dims):
                raise ValueError(f"{name}.shape suffix {arr.shape} expected ...{last_dims}")

        _chk("input_obs", input_obs, ndim=4)
        _chk("target_obs", target_obs, ndim=4)
        _chk("action", action, ndim=2)
        _chk("reward", reward, ndim=1)
        _chk("terminal", terminal, ndim=1)
        _chk("next_idx", next_idx, ndim=1)
        _chk("write_id", write_id, ndim=1)
        _chk("next_write_id", next_write_id, ndim=1)

        if not (len(input_obs) == len(target_obs) == len(action) == len(reward) == len(terminal) ==
                len(next_idx) == len(write_id) == len(next_write_id) == n):
            raise ValueError("Saved arrays have inconsistent first dimension lengths.")

    buf.input_obs[:n] = input_obs
    buf.target_obs[:n] = target_obs
    buf.action[:n] = action
    buf.reward[:n] = reward
    buf.terminal[:n] = terminal.astype(np.bool_, copy=False)
    buf.next_idx[:n] = next_idx.astype(np.int64, copy=False)
    buf.write_id[:n] = write_id.astype(np.int64, copy=False)
    buf.next_write_id[:n] = next_write_id.astype(np.int64, copy=False)

    # Set pointers/state
    buf.idx = n
    buf.full = True  # because capacity == n (we created it that way)
    buf._prev = -1

    # Keep global write counter consistent so future adds (if you ever change full behavior)
    # would not collide. Here add() early-returns when full anyway, but keep it sane.
    if n > 0:
        buf._global_write_counter = np.int64(int(buf.write_id[:n].max()) + 1)
    else:
        buf._global_write_counter = np.int64(1)

    return buf
def simulate(env, num_warmups, num_interaction_episodes,num_agent, ch, h, w, patch , Nr, latent_tokens, z_dim, action_dim, latent_dim, 
                 rep_depth , rep_d_model, dyn_d_model, num_heads, dropout, k_max, mtp, task_id, symlog_for_value,
                 symlog_for_reward,  kmax_prob, lambda_, buffer,
                 policy_bins , reward_bins , pretrain, reward_clamp,level_vocab , level_embed_dim,mode,num_tasks, Sa,
                 batch_lens, batch_size, accum, max_imag_len, buffer_limit, train, ckpt, rep_lr=1e-4, rep_decay=1e-3,eval_context_len=15,
                 dyn_lr=1e-4, dyn_decay=1e-3, policy_lr=1e-4, policy_decay=1e-3 , save_every=500):
    wandb.init(project="Dreamer4", entity="fguan", name=mode)

    agent = Dreamer4(agent_id=0, ch=ch, h=h,
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
                lambda_= lambda_,
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
                symlog_for_reward=symlog_for_reward,
                symlog_for_value=symlog_for_value,
                policy_lr=policy_lr, 
                policy_decay=policy_decay) 
    # 1. Initialize Buffer OUTSIDE the loop so data persists
    if buffer:
        buffer = load_replay(buffer)
    else:
        buffer = ReplayBuffer(
        buffer_limit=1,
        input_obs_size=(ch, h ,w),
        target_obs_size=(ch, h, w),
        action_size=action_dim,
    )

    total_steps = 0
    writer = {}
   # agent.save_rep(None)
    for epi in range(num_warmups + num_interaction_episodes):
        # --- Episode Initialization ---
        observation, _ = env.reset()
        observation = extract_input_and_target(observation)
        done = np.zeros(1, dtype=bool)
        score = 0
        
        # Temporary lists to store the FULL episode
        episode_obs = []
        if (mode == "pretrain" or mode=="inference"):
            train_reward=False
            train_model =False#(epi > 0) and (epi <= 1200) 
            train_policy = False#(epi > 900)
        elif mode=="dyn":
            train_reward=False
            train_model =True#(epi > 0) and (epi <= 1200) 
            train_policy = False#(epi > 900)
        elif mode=="finetune":
            train_reward=True
            train_model =True#(epi > 0) and (epi <= 1200) 
            train_policy = False#(epi > 900)
  
        elif mode=="policy":
            train_reward=False
            train_model =False#(epi > 0) and (epi <= 1200) 
            train_policy = True#(epi > 900)

        else:
            raise AssertionError("Unknown training mode. Acceptable modes are: {pretrain, dyn, finetune, policy, inference}")
        step = 0
        # --- Collection Phase ---

        if ( (mode=="inference")):
            while (not done.all()):
                # Handle Actions
                # Warmup: Random Action
                if not train_policy and (mode != "inference"):
                   act = env.action_space.sample()
                    # Training: Model Action
                elif (mode=="inference"):
                    agent.encoder.eval()
                        # Assuming single agent, extracting index 0
                    state_tensor = torch.from_numpy(observation).to(agent.device)
                    act = agent.action_step(state_tensor, )
                    env.render() 

                    # Step Environment
                # gym_multi_car_racing usually returns dicts or tuples depending on version
                # ensuring compatibility with your previous unpacking
                next_observation, reward, done, info, _ = env.step(act)

                next_obs_proc = extract_input_and_target(next_observation)                
                
                step += 1
                if step == 200:
                    done = ~done
                    # Accumulate data for this step (Agent 0)

                buffer.add(observation, observation, act, reward, done)
                score += reward.item()

                show_windows(observation)
                observation = next_obs_proc
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
        log_data=writer
        # Train on the buffer
        if mode != "inference" and train and (buffer.full or (mode=="policy" and epi > 5)): # Ensure min buffer size
                # Note: buffer is passed directly; train_one_epoch handles sampling internally
            log_data = agent.train_step(writer, buffer, model=train_model, policy=train_policy, train_reward=train_reward)
        wandb.log(data=log_data)
        if (epi)%save_every==0:
        # --- Checkpointing ---
            print(">>> Saving Parameters <<<")
            agent.save_checkpoint(f"WM_{epi}.pt")
            agent.save_rep(f"WM_{epi}.pt")
               # agent.model.save_rep(f"WM_{epi}.pt")

        if (epi+1) % 100==0:        # Ensure evaluate is safe to call
            agent.evaluate(buffer)

        # Reset agent internal state (RNN hidden states) for next episode
        agent.reset()
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run simulate() with configurable hyperparameters.")

    # Positional-style inputs (you likely provide these in code, not CLI)
    p.add_argument("--steps", type=int, default=1_000_000, help="Total environment steps.")
    p.add_argument("--seed", type=int, default=0, help="Random seed.")
    p.add_argument("--config", type=str, default="", help="Optional path to a JSON/YAML config (not used unless you load it).")

    # simulate kwargs
    p.add_argument("--num_agent", type=int, default=1)
    p.add_argument("--buffer_limit", type=int, default=100000)

    # Observation / encoding
    p.add_argument("--ch", type=int, default=3, help="Image Channels")
    p.add_argument("--h", type=int, default=256, help="Image Height")
    p.add_argument("--w", type=int, default=256, help="Image Width")
    p.add_argument("--patch", type=int, default=16, help="Patch Size")
    p.add_argument("--latent_tokens", type=int, default=256, help="Nz")
    p.add_argument("--reserved_tokens", type=int, default=4, help="Nr")

    p.add_argument("--z_dim", type=int, default=16, help="Bottleneck")
    p.add_argument("--action_dim", type=int, default=4)
    p.add_argument("--kmax_base_prob", type=int, default=0.5)
    # Model sizes
    p.add_argument("--Sa", type=int, default=4)

    p.add_argument("--pred_dim", type=int, default=512)
    p.add_argument("--rep_depth", type=int, default=8, help="Has to be a multiple of 2")
    p.add_argument("--rep_d_model", type=int, default=256)
    p.add_argument("--dyn_d_model", type=int, default=256)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.01)
    p.add_argument("--k_max", type=int, default=128, help="Has to be a power of 2")
    p.add_argument("--mtp", type=int, default=8)
    p.add_argument("--num_tasks", type=int, default=10)
    p.add_argument("--task_id", type=int, default=0)
    p.add_argument("--eval_context_len", type=int, default=15)
    p.add_argument("--buffer", type=str, default="")

    # Discretization / vocab
    p.add_argument("--policy_bins", type=int, default=101)
    p.add_argument("--reward_bins", type=int, default=101)
    p.add_argument("--reward_clamp_abs", type=float, default=1.1)
    p.add_argument("--level_vocab", type=int, default=129)
    p.add_argument("--level_embed_dim", type=int, default=256)
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--dyn_lr", type=float, default=4e-5)
    p.add_argument("--dyn_decay", type=float, default=1e-4)
    p.add_argument("--rep_lr", type=float, default=4e-5)
    p.add_argument("--lambda_", type=float, default=0.98)
    p.add_argument("--symlog_for_reward", action="store_true", help="Enable symlog for reward", default=False)
    p.add_argument("--symlog_for_value", action="store_true", help="Enable symlog for value", default=False)
    p.add_argument("--rep_decay", type=float, default=1e-3)
    p.add_argument("--policy_lr", type=float, default=1e-4)
    p.add_argument("--policy_decay", type=float, default=1e-4)
    p.add_argument("--render_mode", type=str, default="rgb_array")
    p.add_argument("--train_mode", type=str, default="pretrain")
    # Training
    p.add_argument(
        "--batch_lens",
        type=int,
        nargs=2,
        default=(15, 40),
        metavar=("MIN_LEN", "MAX_LEN"),
        help="Batch length range, e.g. --batch_lens 45 65",
    )
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--accum", type=int, default=2)
    p.add_argument("--max_imag_len", type=int, default=30)
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

    # You said simulate(agent, env, 1000000, 0, {}, ...)
    # We'll keep agent/env as variables you set up in code.

    # Ensure environment is created correctly
    env = gym.make(
        "PushCubeMultiView-v1",
        obs_mode="rgb",

        reward_mode="sparse",
        render_mode = args.render_mode,
        sensor_configs=dict(width=args.w,height=args.h),
        control_mode="pd_ee_delta_pos",
    )
    print(env.action_space)
    simulate(
        env,
        1000000,
        args.steps,  # your empty dict argument stays as-is
        num_agent=args.num_agent,
        buffer_limit=args.buffer_limit,
        ch=args.ch,
        h=args.h,
        w=args.w,
        patch=args.patch,
        buffer=args.buffer,
        kmax_prob = args.kmax_base_prob,
        latent_tokens=args.latent_tokens,
        z_dim=args.z_dim,
        eval_context_len = args.eval_context_len,
        action_dim=args.action_dim,
        latent_dim=args.pred_dim,
        rep_depth=args.rep_depth,
        rep_d_model=args.rep_d_model,
        dyn_d_model=args.dyn_d_model,
        symlog_for_reward=args.symlog_for_reward,
        symlog_for_value=args.symlog_for_value,

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
        lambda_=  args.lambda_,
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
        policy_lr=args.policy_lr, 
        policy_decay=args.policy_decay,
       mode = args.train_mode
    )

if __name__ == "__main__":
    main()
