import numpy as np

class ReplayBuffer:
    """
    Same interface and base outputs as your buffer.
    sample_seq() does NOT ignore short episodes. """
    def __init__(self, buffer_limit=150000, obs_size=(3, 96, 96), action_size=2,
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
        self.next_write_id = np.zeros((self.buffer_limit,), dtype=np.int64)
        self._global_write_counter = np.int64(1)

        # For single-stream collection: link previous transition to current if not terminal
        self._prev = -1

    def size(self):
        return self.buffer_limit if self.full else self.idx

    def __len__(self):
        return self.size()

    def add(self, state, action, reward, next_state, done):

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
        self.idx = (i + 1) % self.buffer_limit
        self.full = self.full or (self.idx == 0)

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

    def sample_seq(self, batch_size, chunk_size):

        last_filled_index = self.idx - chunk_size + 1
        assert self.full or (last_filled_index > batch_size), "too short dataset or too long chunk_size"

        high = self.buffer_limit if self.full else last_filled_index
        sample_index = self.rng.integers(0, high, size=int(batch_size), endpoint=False).reshape(-1, 1)

        chunk_length = np.arange(chunk_size).reshape(1, -1)
        sample_index = (sample_index + chunk_length) % self.buffer_limit

        observation = self.observation[sample_index]
        action = self.action[sample_index]
        reward = self.reward[sample_index]
        done = self.terminal[sample_index]
        return observation, action, reward, done