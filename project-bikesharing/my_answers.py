import numpy as np


class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5, 
                                       (self.input_nodes, self.hidden_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                       (self.hidden_nodes, self.output_nodes))
        self.lr = learning_rate
        
        #### TODO: Set self.activation_function to your implemented sigmoid function ####
        #
        # Note: in Python, you can define a function with a lambda expression,
        # as shown below.
        # CHECKED
        # changed 1 to 1.0

        self.activation_function = lambda x : 1.0/(1.0+np.exp(-x))  # Replace 0 with your sigmoid calculation.
        self.activation_prime = lambda x: self.activation_function(x) * (1.0-self.activation_function(x))
        ### If the lambda code above is not something you're familiar with,
        # You can uncomment out the following three lines and put your 
        # implementation there instead.
        #
        #def sigmoid(x):
        #    return 0  # Replace 0 with your sigmoid calculation here
        #self.activation_function = sigmoid


    def train(self, features, targets):
        ''' Train the network on batch of features and targets. 
        
            Arguments
            ---------
            
            features: 2D array, each row is one data record, each column is a feature
            targets: 1D array of target values
        
        '''
        n_records = features.shape[0] #number of rows

        # delt weights must correspond to the dimensions it is handling
        # element wise addition or subtraction
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape) # delta weights in shape of input_to_hidden
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape) #delta weights in shape of hidden_to_output
        
        # for each X,y pair of features and target do
        # forward pass and train
        # do back propogation and get delta weights
        # 
        for X, y in zip(features, targets):
        ## comment:  final output and hidden output is a single time pass
        ## comments: delta_weight is accumulative, it accumulates after processing
        ## each row of data
            
            final_outputs, hidden_outputs = self.forward_pass_train(X)  # Implement the forward pass function below
            # Implement the backpropagation function below

            # comment because this network only has two layers
            # just need to pass in final outputs and hidden outputs
            # X y
            # delta_weights_i_h, delta_weights_h_o
            # else needs to pass in even more. 


            # for each record calculate delta weights and then feed this delta weight into backpropagation
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(final_outputs, hidden_outputs, X, y, 
                                                                            delta_weights_i_h, delta_weights_h_o)
        
        # because delta accumulates, it eventually needs to be averaged. 
        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)
        

    def forward_pass_train(self, X):
        ''' Implement forward pass here 
         
            Arguments
            ---------
            X: features batch

        '''
        #### Implement the forward pass here ####
        ### Forward pass ###
        # TODO: Hidden layer - Replace these values with your calculations.

    
        hidden_inputs = np.dot(X,self.weights_input_to_hidden) # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer

        # TODO: Output layer - Replace these values with your calculations.
        # hidden out is already activated and sigmoided, no need to 
        final_inputs = np.dot(hidden_outputs,self.weights_hidden_to_output) # signals into final output layer
        final_outputs = final_inputs # signals from final output layer #f(x) = x 

        
        return final_outputs, hidden_outputs



    def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):
        ''' Implement backpropagation
         
            Arguments
            ---------
            final_outputs: output from forward pass
            y: target (i.e. label) batch
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers

        '''
        #### Implement the backward pass here ####
        ### Backward pass ###

        # TODO: Output error - Replace this value with your calculations.
        error = y - final_outputs # Output layer error is the difference between desired target and actual output.
        # dimension 1,1
        
        # TODO: Calculate the hidden layer's contribution to the error
        hidden_error = np.dot(self.weights_hidden_to_output, error)  #2x1 * 1x1 = 2x1
        # dimen weights_hidden_to_output  2x1
        
        # TODO: Backpropagated error terms - Replace these values with your calculations.
        output_error_term = error * 1 # f(x) = x is 1 no sigmoid activation on the output layer
        # 1x1
        
        #hidden_error_term = hidden_error * self.activation_prime(hidden_outputs) # is hidden outputs already activated, don't need extra implementation!!
        
        hidden_error_term = hidden_error * hidden_outputs * (1 - hidden_outputs)

        #do I also need to multiply by weights?
        # dimen weights_hidden_to_output ?
        # 2x1 * 1x2 = 2x2 ?
        

        #delta weight is ACCUMULATIVE
        # it accumulates over the entire batch of records
        # it is later averaged out by n_records during weight update

        # X is not necessarily the initial input but each row of record

        # Weight step (input to hidden)
        #delta_weights_i_h += np.dot(X[:,None],hidden_error_term[None,:]) #line in the middle 1 to 1
        delta_weights_i_h += np.dot(X[:,None],hidden_error_term[None,:])
        
        
        #delta_weights_i_h += np.dot(X,hidden_error_term)
        # delta_weights_i_h += np.dot(X,hidden_error_term) ValueError: shapes (3,) and (2,) not aligned: 3 (dim 0) != 2 (dim 0)

        # first term dimen weights_hidden_to_output (2, 1)

        # second term  
        # delta_weights_i_h same as weights_input_to_hidden 1x2

        # Weight step (hidden to output)
        #delta_weights_h_o += np.dot(hidden_outputs, output_error_term)
        # forcing inner matrix multiplication dimension to line up
        # [:,None] [None,:] pattern 
        delta_weights_h_o += np.dot(hidden_outputs[:, None], output_error_term[None,:])
        
        #ValueError: shapes (1,) and (2,) not aligned: 1 (dim 0) != 2 (dim 0)
        return delta_weights_i_h, delta_weights_h_o

    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        ''' Update weights on gradient descent step
         
            Arguments
            ---------
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers
            n_records: number of records

        '''
        
        # update real weights with only the average, learning rate discounted delta weights
        self.weights_hidden_to_output += self.lr * delta_weights_h_o/n_records # update hidden-to-output weights with gradient descent step
        self.weights_input_to_hidden += self.lr * delta_weights_i_h/n_records # update input-to-hidden weights with gradient descent step



    def run(self, features):
        ''' Run a forward pass through the network with input features 
        
            Arguments
            ---------
            features: 1D array of feature values
        '''
        
        #### Implement the forward pass here ####
        # TODO: Hidden layer - replace these values with the appropriate calculations.
        hidden_inputs = np.dot(features,self.weights_input_to_hidden) # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer
        
        # TODO: Output layer - Replace these values with the appropriate calculations.
        final_inputs = np.dot(hidden_outputs,self.weights_hidden_to_output) # signals into final output layer
        final_outputs = final_inputs # signals from final output layer 
        
        return final_outputs


#########################################################
# Set your hyperparameters here
##########################################################
iterations = 5000
learning_rate = 0.6
hidden_nodes = 25
output_nodes = 1

'''
good result 2000, 0.1, 40

iterations = 2000
learning_rate = 0.1
hidden_nodes = 20
output_nodes = 1

iterations = 2000
learning_rate = 0.1
hidden_nodes = 15
output_nodes = 1

iterations = 3000
learning_rate = 0.1
hidden_nodes = 20
output_nodes = 1
iterations = 3000
learning_rate = 0.15
hidden_nodes = 30
output_nodes = 1
'''
