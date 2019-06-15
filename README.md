# A tensorflow implementation from scratch of the "binary connect" model

## implemenation details
tensorflow 1.12.0
python3, opencv 3.4.4, numpy 1.16.2


The model is mainly constructed using tf.nn operations for flexibility on lower level variables & operations. Gradients are calculated on the binarized variables( a new tensor named "\_bi").

## analysis of the model
Binary connect is a kind of neural networks that have only binary weights (-1 and 1) when making decisions. In other words, it only uses the sign of weights during forward and backward calculation while keeping accurate weights for SGD to ever work. Yet the model shows its great expression ability on some typical classification tasks that reaches a near state of the art accuracy. This in part implies that we don't actually need such high float accuracy for neural network weights, the power of its expression comes more from the combination of discrete inner states.

## how to run
python3 binary_connect.py

## my testing
on mnist tried both fully connected and CNN
Seems necessary to use activation functions which have negative values, otherwise as my test(like using all Relu) it won't converge. 
Can easily reach 87-90% test accuracy with only -1 and 1 weigths.


## citation: 
paper: Courbariaux, M., Bengio, Y., & David, J. P. (2015). Binaryconnect: Training deep neural networks with binary weights during propagations. In Advances in neural information processing systems (pp. 3123-3131).

author's original implementation based on Lasagne:  https://github.com/MatthieuCourbariaux/BinaryConnect
