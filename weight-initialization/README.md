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























## REFERENCES
1. https://wandb.ai/sauravmaheshkar/initialization/reports/A-Gentle-Introduction-To-Weight-Initialization-for-Neural-Networks--Vmlldzo2ODExMTg
2. https://www.kdnuggets.com/2018/06/deep-learning-best-practices-weight-initialization.html
3. https://www.pinecone.io/learn/weight-initialization/#early-approaches-to-weight-initialization