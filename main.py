from keras.datasets import mnist
from matplotlib import pyplot
from NeuralNet import NeuralNet
import numpy as np



def load_dataset_class_problem():
    trainX = np.array([
        [0.4, -0.7],
        [0.3, -0.5],
        [0.6, 0.1],
        [0.2, 0.4],
       
    ])
    trainY = np.array([[0.1], [0.05], [0.3], [0.25]])
    testX =  np.array([[0.1,-0.2]])
    testY = np.array([[0.12]])
    return trainX,trainY,testX,testY
    

def load_dataset(num_classes):
    # load dataset
    (trainX, trainY), (testX, testY) = mnist.load_data()
    # summarize loaded dataset
    print("Summary of the Dataset:")
    print('Train: X=%s, y=%s' % (trainX.shape, trainY.shape))
    print('Test: X=%s, y=%s' % (testX.shape, testY.shape))
    # plot first few images
    #     for i in range(9):
    #         # define subplot
    #         pyplot.subplot(330 + 1 + i)
    #         # plot raw pixel data
    #         pyplot.imshow(trainX[i], cmap=pyplot.get_cmap('gray'))
        # show the figure
    #pyplot.show()
    # reshape dataset to have a single channel
    X_train = trainX.reshape(trainX.shape[0], trainX.shape[1]*trainX.shape[2]).astype('float32') #flatten 28x28 to 784x1 vectors, [60000, 784]
    trainX = X_train / 255 #normalization
    trainY = np.eye(num_classes)[trainY] #convert label to one-hot
    
    X_test = testX.reshape(testX.shape[0], testX.shape[1]*testX.shape[2]).astype('float32') #flatten 28x28 to 784x1 vectors, [60000, 784]
    testX = X_test / 255 #normalization
    
    return trainX, trainY, testX, testY

    

num_classes = 10
(trainX, trainY, testX, testY) = load_dataset(num_classes)
neural_network = NeuralNet(
                 no_of_nodes_in_layers = [784, 20, 10], 
                 batch_size = 1,
                 epochs = 5,
                 learning_rate = 0.001
             )

neural_network.train(trainX, trainY)
neural_network.plotGraph()

print("Testing...")
neural_network.test(testX, testY)

