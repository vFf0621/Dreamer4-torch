# Unofficial Dreamer4
Compact implementation of Dreamer 4.

Please note that Attention Soft Capping is not implemented as no training instability is found. Instead, Scaled Dot-Product Attention is used

This default implementation uses RoPE1D(as stated in the paper), as well as the one directional masking for latent tokens in the encoder and decoder. Also, 
embedding lookup is implemented for continuous action inputs.

Input images are expected to be of shape (CH, H, W) and normalized to [0, 1].


Action inputs are expected to be normalized to [-1, 1]. Discrete action inputs have not been implemented. Instead, one can set the action bin size to the action resolution (for example, 2 for binary actions) and set num_actions accordingly. This is equivalent to discrete embedding lookup as actions are converted into onehot vectors anyways.

This will NOT be any more data efficient than other implementations; it just consists of fewer lines of code.

Action embeddings are interleaved with the latent, not added, as in previous implementations.

Below are the training artifacts:

<img width="600" height="300" alt="W B Chart 2_28_2026, 6_45_44 PM" src="https://github.com/user-attachments/assets/d67e7c2b-4ab0-4bd5-8370-ade4b840114f" />



And the reconstructed sequence:

![animation](https://github.com/user-attachments/assets/da93ffd6-1cfd-47c2-b455-19af9ccf7fb8)

For the dynamics:

<img width="500" height="300" alt="W B Chart 2_28_2026, 10_55_36 PM" src="https://github.com/user-attachments/assets/714304b4-c737-42f7-9c47-b57f96455ec9" />

Imagined Trajectory:

![output](https://github.com/user-attachments/assets/6a40ab76-da89-4b11-8bf3-fc5d403da0ce)
&nbsp;&nbsp;&nbsp;&nbsp;
![output](https://github.com/user-attachments/assets/d970e24b-1621-47ac-9ccf-7eb572c4203c)
&nbsp;&nbsp;&nbsp;&nbsp;
![output](https://github.com/user-attachments/assets/e5eb5310-d6f0-4f26-a51d-90cd73764c66)
&nbsp;&nbsp;&nbsp;&nbsp;
![output](https://github.com/user-attachments/assets/438a4e8e-88da-4061-bfa6-1c816d7e7d86)
&nbsp;&nbsp;&nbsp;&nbsp;
![output](https://github.com/user-attachments/assets/61dd875d-55b2-4536-a905-b3f149b1da08)

For Finetuning:
<img width="985" height="608" alt="Screenshot from 2026-02-17 17-15-00" src="https://github.com/user-attachments/assets/03ea90a7-7d42-4b6a-b7e0-bd7fa6bdf46d" />


For RL: 
<img width="969" height="1137" alt="Screenshot from 2026-02-17 17-20-51" src="https://github.com/user-attachments/assets/37e42260-438d-423b-bf56-6a85e8eccb9b" />


Memory Consumption:

<img width="100" height="25" alt="image" src="https://github.com/user-attachments/assets/7ae15709-debe-4d71-95c2-87d831d1cfd6" />


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
