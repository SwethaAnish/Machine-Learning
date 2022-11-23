## Why weight Initialization?
Weight initialization is critical as it can impact the training process and the convergence. It can either slow down or even stop the training.
Neural Network Initialization depends on 4 main factors:
1. Number of Inputs
2. Number of outputs 
3. Type of Non-linearity used 
4. Type of Network used 

Some of the Initialization solutions could be  *uniform Initialization U(-a, a) *Normal Initialization N(0, var^sq) *Orthogonal initialization(CNN)

## Zero Initialization:
--> All the weights are initialized to 0
During forward propagation:  
    z = wx + b 
    if w = 0, => z = 0

During forward propagation, the activations are same. So during back propagation, the gradients will also be the same.  The weights are initially zero and will obtain the same update at each step. The neurons evlove symmetrically and it is almost similar to a Linear network. This leads to the "SYMMETRY PROBLEM"

### But why is this a problem? 
If the weight flowing into the Neurons are equal, all the neurons will learn the same thing. This is not a desired behaviour as neural networks are meant to learn different features to map complex inputs and outputs.
The same applies to the case when the weights are initialized with some constant k 

## Random Initialization

#### Initializing weights to small values: 

Considering that weights are drawn from a standard distribution with 0 mean and unit variance, all the weights drawn are scaled by 0.01. For smaller values of weights, the activation values keep decreasing as we go deeper into the network. During backpropagation, the computed gradients are propotional to the respective activation values. For lesser activation values, the gradients are lesser, thus the update is also negligible and the neurons do not learn. This problem is known as the VANISHING GRADIENTS. 
 

#### Initializing weights to large values:

Consider the weights are drawn from a standard distribution with large random values. If the activation function used is tanh or sigmoid, the activation values would come around the saturation region for larger weights. During backpropagation, the gradient of these activation values would close to zero. Thus the weight update is Negligible and Neurons do not learn. 

For some values, which are large enough but would not fall under the saturation region when an activation function is applied, the activation values would be large. In the cases, the corresponding gradient would also be large enough resulting to a larger update of weight. This can change the loss value drastically at each step resulting in oscillations. This may cause the optimizer to oscillate around the minima and might not reach the minima. This problem of larger update is known as EXPLODING GRADIENTS. This can sometimes introduce NaNs to the weights and the loss function. 

#####  Note: The Vanishing and Exploding gradients is applicable only for Sigmoid and tanh activation functions. For relu activation, However, the gradient is equal to 0 for negative input and 1 for positive input. 


## Best Practices:
1. Use Relu/ Leaky Relu as activation function as it is unaffected by the vanishing and exploding gradient problem.
2. Use a heurisitc to initialize weights depending on the non - linear activation function used. The weights are drawn from a normal distribution with variance k/n. (k depends on the choice of activation function)
weights are drawn from a standard normal distribution an scaled by a factor sqrt(k/n).[Refer reference 1 to understand more about how the scaling factor is chosen]

fan_in = Number of Inputs that flow into a neuron and fan_out = Number of outputs that flow out of the neuron

## Xavier or glorot Initialization:

According to Xavier initialization, the variance of the weights chosen is equal to:

image.png






















## REFERENCES
1. https://wandb.ai/sauravmaheshkar/initialization/reports/A-Gentle-Introduction-To-Weight-Initialization-for-Neural-Networks--Vmlldzo2ODExMTg
2. https://www.kdnuggets.com/2018/06/deep-learning-best-practices-weight-initialization.html
3. https://www.pinecone.io/learn/weight-initialization/#early-approaches-to-weight-initialization