
### STUDY HALL 

#### Installation
```
conda install pytorch torchvision torchaudio -c pytorch
conda install tensorboardx opt_einsum pandas tqdm -c conda-forge 
```

#### Reports
- [Attention Mechanisms Cheatsheet](reports/A_quick_guide_to_attention_mechanisms.pdf)

#### Models
- [(ResNet; Deep Residual Learning for Image Recognition)](https://arxiv.org/abs/1512.03385) [[1](models/_resnet.py)]
- [(Squeeze-and-Excitation Networks)](https://arxiv.org/abs/1709.01507) [[1](models/se_baseline.py), [2](models/se_resnet.py)]
- [(Attention Augmented Convolutional Networks)](https://arxiv.org/abs/1904.09925) [[1](models/aa_baseline.py), [2](models/aa_resnet.py)]
- [(Stand-Alone Self-Attention in Vision Models)](https://arxiv.org/abs/1904.09925) [[1](models/sasa_baseline.py), [2](models/sasa_resnet.py)]