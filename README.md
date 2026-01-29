# Minimalistic Dreamer4
Compact implementation of Dreamer 4, with the help from ChatGPT.

Please note that Attention Soft Capping is not implemented as no training instability is found. Instead, Scaled Dot-Product Attention is used.

Different from previous implementations, this default implementation uses RoPE1D(as stated in the paper) for time attention and learned PE for space attention, as well as the one directional masking for latent tokens in the encoder and decoder. Also, 
embedding lookup is implemented for continuous action inputs.

Input images are expected to be of shape (CH, H, W) and normalized to [0, 1].


Action inputs are expected to be normalized to [-1, 1]. Discrete action inputs have not been implemented. Instead, one can set the action bin size to the action resolution (for example, 2 for binary actions) and set num_actions accordingly. This is equivalent to discrete embedding lookup as actions are converted into onehot vectors anyways.

This will NOT be any more data efficient than other implementations; it just consists of fewer lines of code.

Action embeddings are interleaved with the latent, not added, as in previous implementations.

As well, there will be no need of curating a dataset(the dataset is generated through random interactions at first).

Training time for MAE is found to be very fast (41 minutes for the 96 x 96 car racing game).

Below are the training artifacts:


<img width="2528" height="1328" alt="W B Chart 1_29_2026, 2_38_21 PM" src="https://github.com/user-attachments/assets/dc894084-0112-4b96-85a1-f797532b97bf" />

<img width="2528" height="1328" alt="W B Chart 1_29_2026, 2_38_27 PM" src="https://github.com/user-attachments/assets/b6de4922-9ccc-46f8-ab54-3d7fe0aa54c9" />

<img width="2528" height="1328" alt="W B Chart 1_29_2026, 2_38_32 PM" src="https://github.com/user-attachments/assets/a39382a6-44b9-44cf-bf16-82580ecf19e1" />

And the reconstructed sequence:

![animation](https://github.com/user-attachments/assets/da93ffd6-1cfd-47c2-b455-19af9ccf7fb8)

Install via 
```python
pip install -r requirements.txt
```
To Install the MultiCarRacing environment, please see: 

https://github.com/vFf0621/Dreamer4-compact/tree/main/multi_car_racing

Some small functions' codes were borrowed from:

```bibtex
@inproceedings{hansen2024tdmpc2,
  title={TD-MPC2: Scalable, Robust World Models for Continuous Control}, 
  author={Nicklas Hansen and Hao Su and Xiaolong Wang},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2024}
}

@misc{ghugare2023simplifyingmodelbasedrllearning,
      title={Simplifying Model-based RL: Learning Representations, Latent-space Models, and Policies with One Objective}, 
      author={Raj Ghugare and Homanga Bharadhwaj and Benjamin Eysenbach and Sergey Levine and Ruslan Salakhutdinov},
      year={2023},
      eprint={2209.08466},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2209.08466}, 
}
```

Original Paper:
```bibtex

@misc{Hafner2025TrainingAgents,
    title={Training Agents Inside of Scalable World Models}, 
    author={Danijar Hafner and Wilson Yan and Timothy Lillicrap},
    year={2025},
    eprint={2509.24527},
    archivePrefix={arXiv},
    primaryClass={cs.AI},
    url={https://arxiv.org/abs/2509.24527}, 
}
```
