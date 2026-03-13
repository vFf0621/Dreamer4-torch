import numpy as np
import numpy as np

class ReplayBuffer:
    """
    Stores:
      input_obs  : (C_in,H,W)  e.g. 6xHxW (top+random)
      target_obs : (3,H,W)     top view only
      action, reward, terminal
      cam_pose   : (2,7)       per transition

    sample_seq() follows next_idx pointers (no cross-episode, no padding).
    """

    def __init__(
        self,
        buffer_limit=100000,
        input_obs_size=(3, 128, 128),
        target_obs_size=(3, 128, 128),
        action_size=8,
        obs_dtype=np.float32,
        seed=None
    ):
        print("buffer limit is = ", buffer_limit)

        self.buffer_limit = int(buffer_limit)
        self.input_obs_size = tuple(input_obs_size)
        self.target_obs_size = tuple(target_obs_size)
        self.action_size = int(action_size)

        self.input_obs = np.zeros((self.buffer_limit, *self.input_obs_size), dtype=obs_dtype)
        self.target_obs = np.zeros((self.buffer_limit, *self.target_obs_size), dtype=obs_dtype)

        self.action = np.zeros((self.buffer_limit, self.action_size), dtype=np.float32)
        self.reward = np.zeros((self.buffer_limit,), dtype=np.float32)
        self.terminal = np.zeros((self.buffer_limit,), dtype=np.bool_)

        # (NEW) camera pose: [N, 2, 7] (top + random), each pose is (x,y,z,qw,qx,qy,qz)

        self.idx = 0
        self.full = False
        self.rng = np.random.default_rng(seed)

        # Per-transition next pointer within the same episode
        self.next_idx = np.full((self.buffer_limit,), -1, dtype=np.int64)

        # Protect against stale pointers under ring overwrites
        self.write_id = np.zeros((self.buffer_limit,), dtype=np.int64)
        self.next_write_id = np.zeros((self.buffer_limit,), dtype=np.int64)
        self._global_write_counter = np.int64(1)

        # For single-stream collection: link previous transition to current if not terminal
        self._prev = -1

    def size(self):
        return self.buffer_limit if self.full else self.idx

    def __len__(self):
        return self.size()

    def add(
        self,
        input_obs,
        target_obs,
        action,
        reward,
        done,
        next_input_obs=None,   # accepted but not stored (pointer graph already provides next)
    ):
        if self.full:
            return

        i = self.idx

        # Overwrite-safe reset of outgoing pointer from this slot
        self.next_idx[i] = -1
        self.next_write_id[i] = 0

        # Write transition
        self.input_obs[i] = input_obs
        self.target_obs[i] = target_obs
        self.action[i] = action
        self.reward[i] = reward
        self.terminal[i] = bool(done)

        # Stamp this write
        wid = self._global_write_counter
        self._global_write_counter = wid + 1
        self.write_id[i] = wid

        # Link prev -> i if prev exists and prev was not terminal
        if self._prev != -1 and (not self.terminal[self._prev]):
            self.next_idx[self._prev] = i
            self.next_write_id[self._prev] = wid

        # Update prev (episode resets if done)
        self._prev = -1 if bool(done) else i

        # Advance cursor
        self.idx = (i + 1)
        self.full = (self.idx == self.buffer_limit)

    def sample(self, n):
        max_idx = self.size()
        if max_idx == 0:
            raise ValueError("Buffer is empty.")
        idxes = self.rng.integers(0, max_idx, size=int(n), endpoint=False)
        return (
            self.input_obs[idxes, None],
            self.target_obs[idxes, None],
            self.action[idxes, None],
            self.reward[idxes, None],
            self.terminal[idxes, None].astype(np.float32),
        )

    def save(self, path: str, meta: dict | None = None):
        """
        Save buffer contents to a compressed npz.
        Only saves the filled portion.
        """
        n = self.size()
        payload = dict(
            input_obs=self.input_obs[:n],
            target_obs=self.target_obs[:n],
            action=self.action[:n],
            reward=self.reward[:n],
            terminal=self.terminal[:n],
            next_idx=self.next_idx[:n],
            write_id=self.write_id[:n],
            next_write_id=self.next_write_id[:n],
            size=np.array([n], dtype=np.int64),
        )
        if meta is not None:
            # store meta as np object array (npz supports this)
            payload["meta"] = np.array([meta], dtype=object)
        np.savez_compressed(path, **payload)

    def sample_seq(self, batch_size, chunk_size, min_len=10, return_len=False):
        """
        No cross-episode chunks, no padding.
        Enforces that every sampled start has avail_len >= min_len.
        Returns sequences of length L_eff >= min_len (min across batch), capped by chunk_size.
        """
        N = int(self.size())
        B = int(batch_size)
        L = int(chunk_size)
        min_len = int(min_len)

        if N == 0:
            raise ValueError("Buffer is empty.")
        if L < min_len:
            raise ValueError(f"chunk_size ({L}) must be >= min_len ({min_len})")

        def avail_len(start, Lmax):
            cur = int(start)
            wid_cur = int(self.write_id[cur])
            length = 1
            if bool(self.terminal[cur]) or Lmax == 1:
                return 1

            for _ in range(1, Lmax):
                if bool(self.terminal[cur]):
                    break
                nxt = int(self.next_idx[cur])
                if nxt < 0:
                    break
                if int(self.next_write_id[cur]) != int(self.write_id[nxt]):
                    break
                if int(self.write_id[nxt]) != wid_cur + 1:
                    break
                cur = nxt
                wid_cur = int(self.write_id[cur])
                length += 1
            return length

        all_starts = np.arange(N, dtype=np.int64)
        ok_mask = np.fromiter((avail_len(s, min_len) >= min_len for s in all_starts),
                              dtype=np.bool_, count=N)
        candidates = all_starts[ok_mask]
        if candidates.size < B:
            raise ValueError(f"Not enough starts with avail_len >= {min_len}: need {B}, have {candidates.size}.")

        starts = self.rng.choice(candidates, size=B, replace=(candidates.size < B))

        avails = np.array([avail_len(s, L) for s in starts], dtype=np.int64)
        L_eff = int(avails.min())
        if L_eff < min_len:
            raise RuntimeError(f"L_eff {L_eff} < min_len {min_len}. Bug in candidate filter.")

        idxs = np.empty((B, L_eff), dtype=np.int64)
        for i, s in enumerate(starts):
            cur = int(s)
            idxs[i, 0] = cur
            for t in range(1, L_eff):
                cur = int(self.next_idx[cur])
                idxs[i, t] = cur

        input_obs  = self.input_obs[idxs]
        target_obs = self.target_obs[idxs]
        action     = self.action[idxs]
        reward     = self.reward[idxs]
        done_seq   = self.terminal[idxs]

        if return_len:
            return input_obs, target_obs, action, reward, done_seq , L_eff
        return input_obs, target_obs, action, reward, done_seq, 


    def sample_probe(self, bs, bl ):
        input_obs  = self.input_obs[:bl][None]
        target_obs = self.target_obs[:bl][None]
        action     = self.action[:bl][None]
        reward     = self.reward[:bl][None]
        done_seq   = self.terminal[ :bl][None]
        return input_obs, target_obs, action, reward, done_seq, 
