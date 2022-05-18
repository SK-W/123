# AlexNet

**Model Innovation**

1.Relu is used as the activation function instead of the traditional sigmoid and tanh.
<img src="mdImage/relu.jpg">
Relu is an unsaturated function. In this paper, it is verified that its effect exceeds sigmoid in a deeper network, and
the gradient dispersion problem of sigmoid in a deeper network is successfully solved.
<img src="mdImage/relu2.jpg">

2.Model training on multiple GPUs

Improve the training speed of the model and the use scale of data
<img src="mdImage/networks.jpg">

3.Using random drop technique (dropout)

Selectively ignore individual neurons in training,avoid overfitting of the model.

<img src="mdImage/dropout.jpg">


**Model Architecture**

The architecture of AlexNet is 5(convolution layer,relu and pool)+3(full connect layer).

<img src="mdImage/layers.jpg">

