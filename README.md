# NeFL: Nested Federated Learning for Heterogeneous Clients


### TLDR
Method for (i) scaling down a global model into submodels and (ii) aggregating parameters of the submodels for federated learning with heterogeneous clients.

### Abstract
Federated learning (FL) is a promising approach in distributed learning keeping privacy. However, during the training pipeline of FL, slow or incapable clients (i.e., stragglers) slow down the total training time and degrade performance. System heterogeneity, including heterogeneous computing and network bandwidth, has been addressed to mitigate the impact of stragglers. Previous studies tackle the system heterogeneity by splitting a model into submodels, but with less degree-of-freedom in terms of model architecture. We propose nested federated learning (NeFL), a generalized framework that efficiently divides a model into submodels using both depthwise and widthwise scaling. NeFL is implemented by interpreting forward propagation of models as solving ordinary differential equations (ODEs) with adaptive step sizes. To address the inconsistency that arises when training multiple submodels of different architecture, we decouple a few parameters from parameters being trained for each submodel. NeFL enables resource-constrained clients to effectively join the FL pipeline and the model to be trained with a larger amount of data. Through a series of experiments, we demonstrate that NeFL leads to significant performance gains, especially for the worst-case submodel. Furthermore, we demonstrate NeFL aligns with recent studies in FL, regarding pre-trained models of FL and the statistical heterogeneity.


### Experiments

    python NeFL-toy.py --model_name resnet18 --device_id 0 --dataset cifar10 --learnable_step True --method WD # [W, DD, WD]
    python NeFL-pre-resnet.py --model_name wide_resnet101_2 --device_id 0 --learnable_step True --method W
    # FjORD
    python NeFL-toy.py --method W --dataset cifar10 --model_name resnetxx --device_id x --name x
    # NeFL-W
    python NeFL-toy.py --method W --dataset cifar10 --learnable_step True --model_name resnetxx --device_id x --name x
    # NeFL-ADD
    python NeFL-toy.py --method DD --dataset cifar10 --learnable_step True --model_name resnetxx --device_id x --name x
    # NeFL-WD
    python NeFL-toy.py --method WD --dataset cifar10 --learnable_step True --model_name resnetxx --device_id x --name x
    # if pretrained
    --pretrained True

### Citation

Honggu Kang, Seohyeon Cha, Jinwoo Shin, Jongmyeong Lee, and Joonhyuk Kang. NeFL: Nested federated learning for heterogeneous clients. *arXiv preprint arXiv:2308.07761*, 2023.

```
@article{kang2023nefl,
      title={{NeFL}: {N}ested Federated Learning for Heterogeneous Clients}, 
      author={Honggu Kang and Seohyeon Cha and Jinwoo Shin and Jongmyeong Lee and Joonhyuk Kang},
      year={2023},
      journal={arXiv preprint arXiv:2308.07761},
      eprint={2308.07761},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```