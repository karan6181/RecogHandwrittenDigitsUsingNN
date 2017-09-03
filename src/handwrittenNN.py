"""
File: handwrittenNN.py
Language: Python 3.5.1
Author: Karan Jariwala( kkj1811@rit.edu )
Description: Recognizing Handwritten Digits using a
Three-Layer Neural Network and the MNIST Dataset
"""
__author__ = "Karan Jariwala"

import numpy
import scipy.special # For sigmoid activation function
import matplotlib.pyplot as plt # For ploting the graph
from sklearn.metrics import confusion_matrix
import itertools
import sys, argparse

# Data file names
TRAIN_DATA_FILE = "mnist_train.csv"
TEST_DATA_FILE = "mnist_test.csv"

class NeuralNetwork:
    """
    Neural Network class definition
    """
    __slots__ = ("inputNodes", "hiddenNodes", "outputNodes", "learningRate",
                 "wIH", "wHO", "activationFunc", "SSE_History", "SSE", "actualLabel", "predLabel")
    #Initialize the Neural Network
    def __init__(self, inputNodes, hiddenNodes, outputNodes, learningRate):
        """
        Initialize the parameters
        :param inputNodes: Number of input neurons
        :param hiddenNodes: Number of hidden neurons
        :param outputNodes: Number of output neurons
        :param learningRate: Learning rate or Step size
        """
        # set the number of nodes in each input, hidden, and output layer
        self.inputNodes = inputNodes
        self.hiddenNodes = hiddenNodes
        self.outputNodes = outputNodes
        self.learningRate = learningRate
        # self.wIH = (numpy.random.rand(self.hiddenNodes, self.inputNodes) - 0.5) # ranges from -0.5 to 0.5
        # self.wHO = (numpy.random.rand(self.outputNodes, self.hiddenNodes) - 0.5) # ranges from -0.5 to 0.5

        # weights ranges from 0 to 1 / sqrt(number of incoming links)
        self.wIH = numpy.random.normal(0.0, pow(self.hiddenNodes, -0.5), (self.hiddenNodes, self.inputNodes))
        self.wHO = numpy.random.normal(0.0, pow(self.outputNodes, -0.5), (self.outputNodes, self.hiddenNodes))
        self.activationFunc = lambda x : scipy.special.expit(x) # Sigmoid activation function
        self.SSE_History = []   # Store the sum of square error for every epochs
        self.SSE = 0            # Sum of Square error for the last epochs
        self.actualLabel = []   # Actual or expected labels or classes
        self.predLabel = []     # Predicted labels or classes by Neural Network

    # Train the Neural Network
    def train(self, inputList, targetList):
        """
        Train the Neural Network using forward and back propogation and
        apply sigmoid activation function.

        :param inputList: Input data
        :param targetList: Target label
        :return: None
        """
        # Convert the list into 2-Dimensional array
        inputs = numpy.array(inputList, ndmin=2).T
        targets = numpy.array(targetList, ndmin=2).T

        # Calculate signals into hidden layer
        hiddenInpus = numpy.dot(self.wIH, inputs)

        # Calculate signals emerging from hidden layer
        # i.e.: Apply sigmoid activation function on hidden inputs
        hiddenOutputs = self.activationFunc(hiddenInpus)

        # Calculate signals into output layer
        finalInput = numpy.dot(self.wHO, hiddenOutputs)

        # Calculate signals emerging from output layer
        # i.e.: Apply sigmoid activation function on final input of output layer
        finalOutput = self.activationFunc(finalInput)

        # Output layer error is the (target - actual)
        outputErrors = targets - finalOutput
        self.calculateSSE(outputErrors)

        # Hidden layer error is the output Errors, split by weights, recombined at hidden nodes
        hiddenErrors = numpy.dot(self.wHO.T, outputErrors)

        # update the weights for the links between the hidden and output layers
        self.wHO += self.learningRate * numpy.dot((outputErrors * finalOutput * (1.0 - finalOutput)),
                                                  numpy.transpose(hiddenOutputs))

        # update the weights for the links between the input and hidden layers
        self.wIH += self.learningRate * numpy.dot((hiddenErrors * hiddenOutputs * (1.0 - hiddenOutputs)),
                                                  numpy.transpose(inputs))


    def query(self, inputList):
        """
        Query the neural network for predicting the class or label

        :param inputList: Test input data
        :return: List or vector of label number
        """
        # Convert input list to 2-Dimensional array
        inputs = numpy.array(inputList, ndmin=2).T

        # Calculate signals into hidden layer
        hiddenInpus = numpy.dot(self.wIH, inputs)

        # Calculate signals emerging from hidden layer
        # i.e.: Apply sigmoid activation function on hidden inputs
        hiddenOutputs = self.activationFunc(hiddenInpus)

        # Calculate signals into output layer
        finalInput = numpy.dot(self.wHO, hiddenOutputs)

        # Calculate signals emerging from output layer
        # i.e.: Apply sigmoid activation function on final input of output layer
        finalOutput = self.activationFunc(finalInput)

        return finalOutput

    def calculateSSE(self, outputErrors):
        """
        Calculate sum of square error.

        :param outputErrors: List or vector of output error(target - actual)
        :return: None
        """
        for err in outputErrors:
            self.SSE += err * err

def readFile(filename):
    """
    Read the file and store the data into a variable
    :param filename: Name of the file to be read
    :return: A data from the file
    """
    with open(filename, "r") as fp:
        dataList = fp.readlines()
    return dataList

def plotNumbers(inputData, predLabel):
    """
    Plot the grey scale images of pixel data

    :param inputData: A pixel data
    :param predLabel: Predicted label
    :return: None
    """
    allValues = inputData.split(",")
    imageArray = numpy.asfarray(allValues[1:]).reshape((28, 28))
    text = "Original Label: " + str(allValues[0]) + "    " + "Predicted Label: " + str(predLabel)
    plt.title(text)
    plt.imshow(imageArray, cmap="Greys", interpolation="None")
    plt.show()


def trainNN(nnInstance, trainingDataList, epochs):
    """
    Loop over through number of epochs to trained the Neural Network

    :param nnInstance: A Neural Network Instance
    :param trainingDataList: Training data set
    :param epochs: Number of epochs or iteration over a network
    :return: None
    """
    #

    for _ in range(epochs):
        # Go through all records in the training data set
        nnInstance.SSE = 0
        for record in trainingDataList:
            imagePixelList = record.split(",")  # split the record by comma
            # rescale the input colour values from the larger range 0 to 255 to the much smaller range
            # 0.01 - 1.0. Deliberately chosen 0.01 as the lower end of the range to avoid the problems
            # with zero valued inputs because they can artificially kill weight updates. We don’t have
            # to choose 0.99 for the upper end of the input because we don’t need to avoid 1.0 for the inputs.
            inputs = (numpy.asfarray(imagePixelList[1:]) / 255.0 * 0.99) + 0.01
            # Create the target output values (all 0.01, except the desired label which is 0.99)
            targets = numpy.zeros(outputNodes) + 0.01
            # imagePixelList[0] is the target label for this record
            targets[int(imagePixelList[0])] = 0.99
            nnInstance.train(inputs, targets)
        nnInstance.SSE_History.append(nnInstance.SSE / len(trainingDataList))

def testNN(nnInstance, testDataList):
    """
    Test the Neural Network on test data

    :param nnInstance:  A Neural Network Instance
    :param testDataList: Test data set
    :return: None
    """

    # scorecard for how well the network performs, initially empty
    scorecard = list()

    # Go through all the records in the test data set
    for record in testDataList:

        imagePixelList = record.split(",")  # Split the record by comma
        # correct answer is first value
        correctLabel = int(imagePixelList[0])
        nnInstance.actualLabel.append(int(correctLabel))
        # print("Correct Label: ", correctLabel)
        # scale and shift the input
        inputs = (numpy.asfarray(imagePixelList[1:]) / 255.0 * 0.99) + 0.01
        # query the network
        outputs = nnInstance.query(inputs)
        # index of the highest value corresponds to the label
        label = numpy.argmax(outputs)
        nnInstance.predLabel.append(int(label))
        # print("Network answer: ", label)
        # append correct or incorrect to list
        # plotNumbers(record, label)
        if (label == correctLabel):
            # network answer matches correct answer, add 1 to scorecard
            scorecard.append(1)
        else:
            # network answer doesn't matches correct answer, add 0 to scorecard
            scorecard.append(0)
    return scorecard

def checkPerformance(scorecard):
    """
    calculate the performance score, the fraction of correct answers

    :param scorecard: A list representing 1 for correctly classify
                        and 0 for incorrectly classify
    :return: None
    """

    scorecardArrays = numpy.asarray(scorecard)
    print("Performance = ", scorecardArrays.sum() / scorecardArrays.size)


def plotConfusionMatrix(confMtrx, classes, normalize=False, title='Confusion matrix',
                        cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        confMtrx = confMtrx.astype('float') / confMtrx.sum(axis=1)[:, numpy.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(confMtrx)

    plt.imshow(confMtrx, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = numpy.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = confMtrx.max() / 2.
    for i, j in itertools.product(range(confMtrx.shape[0]), range(confMtrx.shape[1])):
        plt.text(j, i, format(confMtrx[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if confMtrx[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


if __name__ == "__main__":
    """
    Main Method
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-in", "--inputnode", dest="inputNode", default=784,
                        type=int, help="Number of Neuron in Input Layer")
    parser.add_argument("-hn", "--hiddennode", dest="hiddenNode", default=200,
                        type=int, help="Number of Neuron in Hidden Layer")
    parser.add_argument("-on", "--outputnode", dest="outputNode", default=10,
                        type=int,help="Number of Neuron in output Layer")
    parser.add_argument("-lr", "--learningrate", dest="learningRate", default=0.20,
                        type=float, help="Learning rate or step size")
    parser.add_argument("-e", "--epochs", dest="epochs", default=5,
                        type=int, help="Number of epochs")

    args = parser.parse_args()

    print("Input Node: {} | Hidden Node: {} | Output Node: {} | Learning Rate: {} | Epochs: {} |".format(
        args.inputNode,
        args.hiddenNode,
        args.outputNode,
        args.learningRate,
        args.epochs
    ))

    inputNodes = int(args.inputNode)
    hiddenNodes = int(args.hiddenNode)
    outputNodes = int(args.outputNode)
    learningRate = float(args.learningRate)
    epochs = int(args.epochs)

    # Create instance of Neural Network
    nn = NeuralNetwork(inputNodes, hiddenNodes, outputNodes, learningRate)

    # Load the mnist training data CSV file into list
    trainingDataList = readFile(TRAIN_DATA_FILE)
    trainNN(nn, trainingDataList, epochs)

    # Load the mnist testing data CSV file into list
    testDataList = readFile(TEST_DATA_FILE)
    scorecard = testNN(nn, testDataList)

    # calculate the performance score, the fraction of correct answers
    checkPerformance(scorecard)
    # print(nn.SSE_History)
    # print("Actual:\t", nn.actualLabel)
    # print("Predic:\t", nn.predLabel)

    # Compute confusion matrix
    cnfMatrix = confusion_matrix(nn.actualLabel, nn.predLabel)
    numpy.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plotConfusionMatrix(cnfMatrix, classes="0123456789",
                        title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plt.figure()
    plotConfusionMatrix(cnfMatrix, classes="0123456789", normalize=True,
                        title='Normalized confusion matrix')

    plt.show()


