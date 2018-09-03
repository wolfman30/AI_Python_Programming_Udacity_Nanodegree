
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd

admissions = pd.read_csv('binary.csv')

# Make dummy variables for rank
data = pd.concat([admissions, pd.get_dummies(admissions['rank'], prefix='rank')], axis=1)
data = data.drop('rank', axis=1)

# Standarize features which subtracts the mean from each input value, then divides each value by the standard deviation
for field in ['gre', 'gpa']:
    mean, std = data[field].mean(), data[field].std()#captures and defines mean and standard deviation of gre and gpa columns
    data.loc[:,field] = (data[field]-mean)/std#performans standardization on each value in each column
    
# Split off random 10% of the data for testing
np.random.seed(21)
sample = np.random.choice(data.index, size=int(len(data)*0.9), replace=False)#
data, test_data = data.iloc[sample], data.drop(sample)#captures & defines training set and testing set, respectively

# Split into features and targets
features, targets = data.drop('admit', axis=1), data['admit']
features_test, targets_test = test_data.drop('admit', axis=1), test_data['admit']

np.random.seed(21)
data.head()


# In[3]:


#####Implementation of Backpropagation#######
def sigmoid(x):
    """
    Calculate sigmoid
    """
    return 1 / (1 + np.exp(-x))


# Hyperparameters
n_hidden = 2  # number of hidden units
epochs = 900
learnrate = 0.005

n_records, n_features = features.shape
last_loss = None
# Initialize weights
#weights_input_hidden = (6,2)
weights_input_hidden = np.random.normal(scale=1 / n_features ** .5,
                                        size=(n_features, n_hidden))
#weights_hidden_output.shape = (2,)
weights_hidden_output = np.random.normal(scale=1 / n_features ** .5,
                                         size=n_hidden)

for e in range(epochs):
    
    #delta weights from inputs to the two hidden units
    del_w_input_hidden = np.zeros(weights_input_hidden.shape)
    
    #delta weights from hidden units to the single output unit
    del_w_hidden_output = np.zeros(weights_hidden_output.shape)
    
    #iterates over the features and targets, 
    #assigning features to x and targets to y
    for x, y in zip(features.values, targets):
        ## Forward pass ##
        # DONE: CalculateS the output
        hidden_input = np.dot(x, weights_input_hidden)#input to hidden units
        hidden_output = sigmoid(hidden_input)#output of hidden units
        
        #output of final output unit
        output = sigmoid(np.dot(hidden_output, weights_hidden_output))

        ## Backward pass ##
        # DONE: Calculate the network's prediction error
        error = y - output #error = difference between predited output and target

        # DONE: Calculate error term for the output unit
        #output * (1 - output) = sigmoid_prime, derivative of the output
        output_error_term = error * output * (1 - output)
        

        ## propagate errors to hidden layer

        # DONE: Calculate the hidden layer's contribution to the error
        hidden_error = np.dot(weights_input_hidden, output_error_term)
        
        # DONE: Calculate the error term for the hidden layer
        hidden_error_term = hidden_error * hidden_output * (1-hidden_output)
        
        # DONE: Update the change in weights
        del_w_hidden_output += output_error_term*output
        del_w_input_hidden += hidden_error_term*np.array(x, ndmin=2).T

    # DONE: Update weights
    #adds the product of learning rate, change in weights from adding negative gradients, and 
    #takes the average by dividing by total number of data points or number of records
    weights_input_hidden += learnrate* del_w_input_hidden/n_records
    weights_hidden_output += learnrate * del_w_hidden_output/n_records

    # Printing out the mean square error on the training set
    if e % (epochs / 10) == 0:
        hidden_output = sigmoid(np.dot(x, weights_input_hidden))
        out = sigmoid(np.dot(hidden_output,
                             weights_hidden_output))
        loss = np.mean((out - targets) ** 2)

        if last_loss and last_loss < loss:
            print("Train loss: ", loss, "  WARNING - Loss Increasing")
        else:
            print("Train loss: ", loss)
        last_loss = loss

# Calculate accuracy on test data
hidden = sigmoid(np.dot(features_test, weights_input_hidden))
out = sigmoid(np.dot(hidden, weights_hidden_output))
predictions = out > 0.5
accuracy = np.mean(predictions == targets_test)
print("Prediction accuracy: {:.3f}".format(accuracy))

