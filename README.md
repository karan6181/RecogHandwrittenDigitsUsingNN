# Recognizing Handwritten Digits using a Three-Layer Neural Network and the MNIST Dataset

### Problem Statement:

Every person is different in the world and people write in many different ways. Such a scenario causes a different patters in writing. The problem statement is to design a machine which recognize an Handwritten digits efficiently.

------

### Description:

Neural networks are used for recognizing the digit and sigmoid neuron is the basic building block of neural networks. A neuron takes several inputs **(x1,x2, ..)** and produces a single output. Inputs can take any value between 0 and 1. It has weights for each input such as **w1, w2, ...** and an overall **bias, b**. Its output is **σ(w.x+b)**, where **σ** is the **sigmoid** function.

I have trained and tested a model using neural network for digit recognition in python programming language. Neural network used for digit recognition contains 3 layers – Input layer containing **784 (28x28)** input neurons, hidden layer containing **700** neurons and output layer with **10** output neurons. I have used the **"MNIST"** dataset which provides test and validation images of handwritten digits. The digits have been size-normalized and centered in a fixed-size image. The data set consist of **60,000 training samples** or images and **10,000 test samples** or images and each sample is 28x28 pixel in dimension. The main limitation of this model is that it recognize only numbers(0-9) and able to predict numbers only if the images is in grey scale. 

------

### Dependencies:

- Python 2.7+

- Scikit-learn

- Numpy

- Matplotlib

- scipy

  ​

### Execution of Program:

The handwrittenNN.py file trains on 60,000 sample and tests on 10,000 sample. To run the file, execute the following step

```powershell
python3 handwrittenNN.py -hn 500 -lr 0.2 -e 10
```

where,

| Parameter | Details                                  |
| --------- | ---------------------------------------- |
| -in       | Number of Neurons in Input Layer(optional) ***[Default: 784]*** |
| -hn       | Number of Neurons in Hidden Layer, ***[Default: 800]*** |
| -on       | Number of Neurons in output Layer(optional) ***[Default: 10]*** |
| -lr       | Learning rate or step size ***[Default: 0.15]*** |
| -e        | Number of epochs ***[Default: 5]***      |

## Test Images Classification Output:

![Test Images Classification](https://github.com/karan6181/RecogHandwrittenDigitsUsingNN/blob/master/Output/Images/predictedOutput.png)

### Tweaking the Learning Rate:

The following plots a graph of different learning rate keeping the other parameters(number of neurons in hidden layer, epochs) constant. It’s not a very scientific approach because I should really
do these experiments many times to reduce the effect of randomness and bad journeys down the gradient
descent, but it is still useful to see the general idea that there is a sweet spot for learning rate.

![Performance Vs. Learning Rate](https://github.com/karan6181/RecogHandwrittenDigitsUsingNN/blob/master/Output/Images/performanceVSLearningRate.PNG)

The plot suggested that between a learning rate of 0.1 and 0.2 there might be better performance.

### Doing Multiple Runs:

The next improvement I did was to repeat the training several times against the data set. Intuition suggests the more training you do the better the performance. But as we know that too much training is actually bad because the network **overfits** to the training data, and then performs badly against new data that it hasn’t seen before. 

![Performance Vs. Number of Epochs](https://github.com/karan6181/RecogHandwrittenDigitsUsingNN/blob/master/Output/Images/performanceVSnumOfEpochs.PNG)

As we can see from the plot that results are not quite so predictable. We can see the high performance during epoch 6 to 10.

### Change Network Shape:

Here, I have changed the number of neurons in the hidden layer.  Too few hidden nodes is bad because there is not enough space for the network to learn from images and very large number of neurons is also not good because we might find it harder to train the network because now there are too many options for where the learning should go.

The following plots a graph showing number of hidden nodes vs performance

![Performance Vs. Number of neurons in Hidden Layer](https://github.com/karan6181/RecogHandwrittenDigitsUsingNN/blob/master/Output/Images/performanceVsHiddenNode.PNG)

The plot suggested that between a Hidden nodes of 600 to 1000 there might be better performance.

------

### Outputs:

```powershell
Input Node: 784 | Hidden Node: 800 | Output Node: 10 | Learning Rate: 0.15 | Epochs: 5 |
Performance =  0.9749
```

The resultant performance is 0.9749. This performance is already amongst the better ones listed on [Yann LeCunn’s](http://yann.lecun.com/exdb/mnist/) website.

Following figure shows the performance and confusion matrix.

1. **Confusion matrix, without normalization**

   ![Confusion Matrix without Normalization](https://github.com/karan6181/RecogHandwrittenDigitsUsingNN/blob/master/Output/Images/withoutNormalize.png)

2. **Normalized confusion matrix**

   ![Confusion Matrix with Normalization](https://github.com/karan6181/RecogHandwrittenDigitsUsingNN/blob/master/Output/Images/withNormalize.png)

------

**Trained CSV File:**

Since the MNIST train csv file is huge(107MB), I can't upload on Github. So please download the MNIST train csv from here:

https://drive.google.com/open?id=0B81PAaZPCK-yeXk3ZXlndktCVms

------

### Future work:

- 3-layer Neural Network with softmax, cross entropy, weight decay
- Convolutional Neural Network approach

------

