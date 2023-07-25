# NeFL: Nested Federated Learning for Heterogeneous Clients

### Abstract
During the training pipeline of Federated learning (FL), slow or incapable clients (i.e., stragglers) slow down the total training time and degrade performance.
System heterogeneity, including heterogeneous computing and network bandwidth, has been addressed to mitigate the impact of stragglers.
Previous studies split models to tackle the issue, but with less degree-of-freedom in terms of model architecture.
We propose nested federated learning (NeFL), a generalized framework that efficiently divides a model into submodels using both depthwise and widthwise scaling.
NeFL is implemented by interpreting models as solving ordinary differential equations (ODEs) with adaptive step sizes.
To address the inconsistency that arises when training multiple submodels with different architecture, we decouple a few parameters.
NeFL enables resource-constrained clients to effectively join the FL pipeline and the model to be trained with a larger amount of data.
Through a series of experiments, we demonstrate that NeFL leads to significant gains, especially for the worst-case submodel.
Somewhat interestingly, by investigating how to utilize pre-trained models within the NeFL, we found that Vision Transformers (ViTs) hold promise in addressing both system and statistical heterogeneity.

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
