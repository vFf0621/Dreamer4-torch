# Memory efficient Dreamer4
Compact implementation of Dreamer 4.

Please note that Attention Soft Capping is not implemented as no training instability is found. Instead, Scaled Dot-Product Attention is used.

Different from previous implementations, this default implementation uses RoPE1D(as stated in the paper) for time attention and learned PE for space attention, as well as the one directional masking for latent tokens in the encoder and decoder. Also, 
embedding lookup is implemented for continuous action inputs.

Input images are expected to be of shape (CH, H, W) and normalized to [0, 1].


Action inputs are expected to be normalized to [-1, 1]. Discrete action inputs have not been implemented. Instead, one can set the action bin size to the action resolution (for example, 2 for binary actions) and set num_actions accordingly. This is equivalent to discrete embedding lookup as actions are converted into onehot vectors anyways.

This will NOT be any more data efficient than other implementations; it just consists of fewer lines of code.

Action embeddings are interleaved with the latent, not added, as in previous implementations.

As well, there might be no need of curating a dataset if your environment is dense, like the car racing game(the dataset is generated through random interactions at first).

Training time for MAE is found to be rather fast (176 minutes for the 96 x 96 car racing game on a RTX PRO 6000 GPU).

Below are the training artifacts:


<img width="200" height="200" alt="W B Chart 1_30_2026, 11_35_06 PM" src="https://github.com/user-attachments/assets/350b72ad-b8ef-434a-9d9e-8c31e07f7f7d" />

<img width="200" height="200" alt="W B Chart 1_30_2026, 11_34_36 PM" src="https://github.com/user-attachments/assets/3d20bd7a-8496-4b55-b4d0-41fa962ca05b" />

<img width="200" height="200" alt="W B Chart 1_30_2026, 11_34_54 PM" src="https://github.com/user-attachments/assets/fd60f127-e38f-406d-b431-6d392f15ce45" />


And the reconstructed sequence:

![animation](https://github.com/user-attachments/assets/da93ffd6-1cfd-47c2-b455-19af9ccf7fb8)

For the dynamics:

<img width="200" height="200" alt="W B Chart 2_1_2026, 10_31_16 AM" src="https://github.com/user-attachments/assets/e350a4fe-c6d9-4a93-b995-57768b06b527" />

<img width="200" height="200" alt="W B Chart 2_1_2026, 10_31_28 AM" src="https://github.com/user-attachments/assets/c9a35ad2-d289-4317-b210-1898bef29282" />

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





Memory Consumption:
<img width="873" height="30" alt="image" src="https://github.com/user-attachments/assets/cbe37613-a936-4322-9f02-8c64b526d9f6" />

Policy training is still underway!!

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
