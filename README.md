# Minimalistic Dreamer4
Compact implementation of Dreamer 4, with the help of ChatGPT.

Please note that Attention Soft Capping is not implemented as no training instability is found. Instead, Scaled Product Attention is used.

Different from previous implementations, this default implementation uses RoPE1D and learned PE for space attention, as well as the one directional masking for latent tokens in the encoder and decoder. Also, 
embedding lookup is implemented for action inputs.

Input images are expected to be of shape (CH, H, W) and normalized in [0, 1].

Discrete action inputs have not been implemented.

Install via 
```python
pip install -r requirements.txt
```
To Install the MultiCarRacing environment, please see: 

https://github.com/vFf0621/Dreamer4-compact/tree/main/multi_car_racing

Some codes were borrowed from:

```bibtex
@inproceedings{hansen2024tdmpc2,
  title={TD-MPC2: Scalable, Robust World Models for Continuous Control}, 
  author={Nicklas Hansen and Hao Su and Xiaolong Wang},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2024}
}
@misc{Hansen2026Dreamer4PyTorch,
    title={Dreamer 4 in PyTorch},
    author={Nicklas Hansen},
    year={2026},
    publisher={GitHub},
    journal={GitHub repository},
    howpublished={\url{https://github.com/nicklashansen/dreamer4}},
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
