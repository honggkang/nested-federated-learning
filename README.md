# NeFL: Nested Federated Learning for Heterogeneous Clients


### TLDR
NeFL divides a model into submodels by widthwise or/and depthwise introducing inconsistent parameters. Then, NeFL aggregates the knowledge of submodels.


### Abstract
Federated learning (FL) enables distributed training while preserving data privacy, but stragglers—slow or incapable clients—can significantly slow down the total training time and degrade performance. To mitigate the impact of stragglers, system heterogeneity, including heterogeneous computing and network bandwidth, has been addressed. While previous studies have addressed system heterogeneity by splitting models into submodels, they offer limited flexibility in model architecture design without considering potential inconsistencies arising from training multiple submodel architectures. We propose nested federated learning (NeFL), a generalized framework that efficiently divides deep neural networks into submodels using both depthwise and widthwise scaling. NeFL interprets forward propagation as solving ordinary differential equations (ODEs) with adaptive step sizes, allowing for dynamic submodel architectures. To address the inconsistency arising from training multiple submodel architectures, NeFL decouples a subset of parameters from those being trained for each submodel. An averaging method is proposed to handle these decoupled parameters during aggregation. NeFL enables resource-constrained devices to effectively participate in the FL pipeline, facilitating larger datasets for model training. Experiments demonstrate that NeFL achieves performance gain, especially for the worst-case submodel compared to baseline approaches. Furthermore, NeFL aligns with recent advances in FL, such as leveraging pre-trained models and accounting for statistical heterogeneity. Our code is available online: https://honggkang.github.io/nefl.



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