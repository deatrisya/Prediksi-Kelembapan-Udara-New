import numpy as np

class NeuralNetwork:

    def __init__(self,input,hidden,output):
        self.input = input
        self.hidden = hidden
        self.output = output

    def initialize_weights(self,  bias=False):
        self.hidden_weights=np.random.uniform(size=(self.input,self.hidden))
        self.output_weights=np.random.uniform(size=(self.hidden,self.output))
        self.bias = False
        if bias:
            self.hidden_bias_weights=np.random.uniform(size=(1,self.hidden))
            self.output_bias_weights=np.random.uniform(size=(1,self.output))
            self.bias = True

import numpy as np

class Sigmoid:
    def activate(self, x):
        return 1/(1 + np.exp(-x))
    def derivative(self, x):
        return x * (1 - x)

"""## Class Feedforward Backward"""

class Backpropagation:

    def __init__(self, neuralnet, epochs=2000, lr=0.1, activation_function=Sigmoid()):
        self.neuralnet = neuralnet
        self.epochs = epochs
        self.lr = lr
        self.activation_function = activation_function

    def feedForward(self, input):
        hidden_layer = np.dot(input, self.neuralnet.hidden_weights)
        if self.neuralnet.bias:
            hidden_layer += self.neuralnet.hidden_bias_weights
        hidden_layer = self.activation_function.activate(hidden_layer)

        output_layer = np.dot(hidden_layer, self.neuralnet.output_weights)
        if self.neuralnet.bias:
            output_layer += self.neuralnet.output_bias_weights
        output_layer = self.activation_function.activate(output_layer)

        return hidden_layer, output_layer

    def train(self, input, target):
        for epoch in range(self.epochs):
            # print(f"\nEpoch {epoch + 1}/{self.epochs}:")

            # Feed Forward
            hidden_layer, output_layer = self.feedForward(input)

            # Error term for each output unit k
            derivative_output = self.activation_function.derivative(output_layer)
            del_k = output_layer * derivative_output * (target - output_layer)

            # Error term for each hidden unit h
            sum_del_h = del_k.dot(self.neuralnet.output_weights.T)
            derivative_hidden = self.activation_function.derivative(hidden_layer)
            del_h = hidden_layer * derivative_hidden * sum_del_h


            # Weight Update
            self.neuralnet.output_weights += hidden_layer.T.dot(del_k) * self.lr
            self.neuralnet.hidden_weights += input.T.dot(del_h) * self.lr

    def predict(self, input, actual_output):
        hidden_layer, output_layer = self.feedForward(input)
        predicted_values = []  # List untuk menyimpan nilai output_layer[i][0]

        for i in range(len(input)):
          for j in range(len(actual_output)):
            if i==j:
              predicted_value = output_layer[i][j]
              actual_value = actual_output[i][0]

              # print(f"For input {input[i]}, the predicted output is {predicted_value} and the actual output is {actual_value}")

              # Simpan nilai predicted_value ke dalam list
              predicted_values.append(predicted_value)

        return predicted_values
    
    def predict_new_value(self, input):
        hidden_layer, output_layer = self.feedForward(input)
        predicted_values = []  # List untuk menyimpan nilai output_layer[i][0]
        for i in range(len(input)):
            for j in range(len(input)):
                if i==j:
                    predicted_value = output_layer[i][j]
                # print(f"For input {input[i]}, the predicted output is {predicted_value} and the actual output is {actual_value}")
                # Simpan nilai predicted_value ke dalam list
            predicted_values.append(predicted_value)
 
# import pickle
# with open('model.pkl', 'wb') as f:
#     pickle.dump(bp, f)