import numpy as np 
import helper
from bokeh.plotting import figure, output_file, show

class NeuralNet:

    def __init__(self, 
                 no_of_nodes_in_layers, 
                 batch_size,
                 epochs,
                 learning_rate
                 ):

        self.no_of_nodes_in_layers = no_of_nodes_in_layers
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate

        self.weight1 = np.random.normal(0, 1, [self.no_of_nodes_in_layers[0], self.no_of_nodes_in_layers[1]])
        self.bias1 = np.zeros((1, self.no_of_nodes_in_layers[1]))
        self.weight2 = np.random.normal(0, 1, [self.no_of_nodes_in_layers[1], self.no_of_nodes_in_layers[2]])
        self.bias2 = np.zeros((1, self.no_of_nodes_in_layers[2]))
        self.loss = []
        self.x = np.array([])
        self.y = np.array([])
        
    def train(self, inputs, labels):
   
        for epoch in range(self.epochs): 
            iteration = 0
            while iteration < len(inputs):
    
                # batch input
                inputs_batch = inputs[iteration:iteration+self.batch_size]
                labels_batch = labels[iteration:iteration+self.batch_size]
                        
                # forward pass
                z1 = np.dot(inputs_batch, self.weight1) + self.bias1
                a1 = helper.relu(z1)
                z2 = np.dot(a1, self.weight2) + self.bias2
                y = helper.softmax(z2)
                    
                # calculate loss
                loss = helper.cross_entropy(y, labels_batch)
                loss += helper.L2_regularization(0.01, self.weight1, self.weight2)
                self.loss.append(loss)
    
               
                # backward pass
                delta_y = (y - labels_batch) / y.shape[0]
                delta_hidden_layer = np.dot(delta_y, self.weight2.T) 
                delta_hidden_layer[a1 <= 0] = 0  
    
                # backpropagation
                weight2_gradient = np.dot(a1.T, delta_y)  # forward * backward
                bias2_gradient = np.sum(delta_y, axis=0, keepdims=True)
                
                weight1_gradient = np.dot(inputs_batch.T, delta_hidden_layer)
                bias1_gradient = np.sum(delta_hidden_layer, axis = 0, keepdims = True)
    
                # L2 regularization
                weight2_gradient += 0.01 * self.weight2
                weight1_gradient += 0.01 * self.weight1
    
                #------Stochastic Gradient Descent -------
                self.weight1 -= self.learning_rate * weight1_gradient #update weight and bias
                self.bias1 -= self.learning_rate * bias1_gradient
                self.weight2 -= self.learning_rate * weight2_gradient
                self.bias2 -= self.learning_rate * bias2_gradient
                
                print('Epoch: '+str(epoch+1)+'   Iteration:'+str(iteration+1)+'  Loss: '+str(loss))
                iteration += self.batch_size
            self.x = np.append(self.x, epoch+1)
            self.y = np.append(self.y, loss)
     

    def test(self, inputs, labels):
        input_layer = np.dot(inputs, self.weight1)
        hidden_layer = helper.relu(input_layer + self.bias1)
        scores = np.dot(hidden_layer, self.weight2) + self.bias2
        probs = helper.softmax(scores)
        acc = float(np.sum(np.argmax(probs, 1) == labels)) / float(len(labels))
        print('Test accuracy: '+ str(acc*100))     
        
    #This graph needs to be connected to network for displaying graph    
    def plotGraph(self):  
        output_file("LossVsEpoch.html")

        #Generate graph
        plot = figure(title= "Y = X**0.5", 
                        x_axis_label= 'Epoch', 
                        y_axis_label= 'Loss')
        plot.line(self.x, self.y, legend= 'f(X)', line_width = 2)
        
        show(plot)
