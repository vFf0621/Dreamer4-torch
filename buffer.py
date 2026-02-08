import numpy as np

class ReplayBuffer:
    """
    Same interface and base outputs as your buffer.
    sample_seq() does NOT ignore short episodes. """
    def __init__(self, buffer_limit=36000, obs_size=(3, 96, 96), action_size=2,
                 obs_dtype=np.float32, seed=None):
        print("buffer limit is = ", buffer_limit)

        self.buffer_limit = int(buffer_limit)
        self.obs_size = tuple(obs_size)
        self.action_size = int(action_size)

        self.observation = np.zeros((self.buffer_limit, *self.obs_size), dtype=obs_dtype)
        self.action = np.zeros((self.buffer_limit, self.action_size), dtype=np.float32)
        self.reward = np.zeros((self.buffer_limit,), dtype=np.float32)
        self.terminal = np.zeros((self.buffer_limit,), dtype=np.bool_)

        self.idx = 0
        self.full = False
        self.rng = np.random.default_rng(seed)

        # Per-transition next pointer within the same episode
        self.next_idx = np.full((self.buffer_limit,), -1, dtype=np.int64)

        # Protect against stale pointers under ring overwrites
        self.write_id = np.zeros((self.buffer_limit,), dtype=np.int64)
        self.next_write_id = np.zeros((self.buffer_limit, ), dtype=np.int64)
        self._global_write_counter = np.int64(1)

        # For single-stream collection: link previous transition to current if not terminal
        self._prev = -1

    def size(self):
        return self.buffer_limit if self.full else self.idx

    def __len__(self):
        return self.size()

    def add(self, state, action, reward, next_state, done):
        if self.full:
            return
        i = self.idx

        # Overwrite-safe reset of outgoing pointer from this slot
        self.next_idx[i] = -1
        self.next_write_id[i] = 0

        # Write transition
        self.observation[i] = state
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
            self.observation[idxes],
            self.action[idxes],
            self.reward[idxes],
            self.terminal[idxes].astype(np.float32),
        )
    def sample_probe(self, batch_size, chunk_size):
        if getattr(self, "_probe_cache", None) is None:
            self._probe_cache = (
                self.observation[:chunk_size][None].copy(),
                self.action[:chunk_size][None].copy(),
                self.reward[:chunk_size][None].copy(),
                self.terminal[:chunk_size][None].copy(),
            )
        return self._probe_cache

    def sample_seq(self, batch_size, chunk_size, min_len=10, return_len=False):
        """
        No cross-episode chunks, no padding.
        Enforces that every sampled start has avail_len >= min_len (default 10).
        Returns sequences of length L_eff >= min_len.
        """
        N = int(self.size())
        B = int(batch_size)
        L = int(chunk_size)
        min_len = int(min_len)

        if N == 0:
            raise ValueError("Buffer is empty.")
        if L < 1:
            raise ValueError("chunk_size must be >= 1")
        if min_len < 1:
            raise ValueError("min_len must be >= 1")
        if L < min_len:
            raise ValueError(f"chunk_size ({L}) must be >= min_len ({min_len})")

        def avail_len(start, Lmax):
            cur = int(start)
            wid_cur = int(self.write_id[cur])
            length = 1
            if bool(self.terminal[cur]) or Lmax == 1:
                return 1

            for _ in range(1, Lmax):
                if bool(self.terminal[cur]):  # episode ends at cur
                    break
                nxt = int(self.next_idx[cur])
                if nxt < 0:
                    break
                # overwrite-safe link check
                if int(self.next_write_id[cur]) != int(self.write_id[nxt]):
                    break
                # strict contiguity for single-stream buffer
                if int(self.write_id[nxt]) != wid_cur + 1:
                    break
                cur = nxt
                wid_cur = int(self.write_id[cur])
                length += 1

            return length

        # --- Build candidate starts that can provide >= min_len steps ---
        # (Uses Lmax=min_len so this filter is cheap-ish.)
        all_starts = np.arange(N, dtype=np.int64)
        ok_mask = np.fromiter((avail_len(s, min_len) >= min_len for s in all_starts),
                            dtype=np.bool_, count=N)
        candidates = all_starts[ok_mask]

        if candidates.size < B:
            raise ValueError(
                f"Not enough starts with avail_len >= {min_len}: need {B}, have {candidates.size}."
            )

        starts = self.rng.choice(candidates, size=B, replace=(candidates.size < B))

        # Now compute the effective length for this batch, capped by chunk_size.
        avails = np.array([avail_len(s, L) for s in starts], dtype=np.int64)
        L_eff = int(avails.min())
        if L_eff < min_len:
            # Should not happen because candidates ensured min_len, but keep as safety.
            raise RuntimeError(f"L_eff {L_eff} < min_len {min_len}. Bug in avail_len/candidates filter.")

        # Gather indices by following next pointers for exactly L_eff steps
        idxs = np.empty((B, L_eff), dtype=np.int64)
        for i, s in enumerate(starts):
            cur = int(s)
            idxs[i, 0] = cur
            for t in range(1, L_eff):
                cur = int(self.next_idx[cur])
                idxs[i, t] = cur

        observation = self.observation[idxs]
        action      = self.action[idxs]
        reward      = self.reward[idxs]
        done_seq    = self.terminal[idxs]

        if return_len:
            return observation, action, reward, done_seq, L_eff
        return observation, action, reward, done_seq
