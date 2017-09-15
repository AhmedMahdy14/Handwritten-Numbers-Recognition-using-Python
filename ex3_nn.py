import scipy.io as spy
import numpy as np
from displayData import displayData
from predict import predict

## Setup the parameters you will use for this exercise
input_layer_size  = 400  # 20x20 Input Images of Digits
hidden_layer_size = 25   # 25 hidden units
num_labels = 10          # 10 labels, from 1 to 10   
                         # (note that we have mapped "0" to label 10)

## =========== Part 1: Loading and Visualizing Data =============
#  We start the exercise by first loading and visualizing the dataset. 
#  You will be working with a dataset that contains handwritten digits.
#

# Load Training Data
print('Loading and Visualizing Data ...')


Data = spy.loadmat('ex3data1.mat')#load .mat data 

X = Data["X"] #extract training set frim .mat file 
y = Data["y"]  #extract actual output coressponding to this training example

y = y.flatten() #Return a copy of the array collapsed into one dimension-- more effecient in my code --

m = X.shape[0] #return number of rows in X numpy array 


rand_indices = np.random.permutation(m) #Randomly permute a sequence from 0 to m=5000=#of training set
sel = X[rand_indices[:100],:]           # Randomly select 100 data points to display

displayData(sel)

raw_input('Program paused. Press enter to continue.\n')

## ================ Part 2: Loading Pameters ================
# In this part of the exercise, we load some pre-initialized 
# neural network parameters.

print('Loading Saved Neural Network Parameters ...')
    


Data_Weight = spy.loadmat('ex3weights.mat') # Load the weights into variables Theta1 and Theta2
Theta1 = Data_Weight["Theta1"] #get Theta1 in np.array
Theta2 = Data_Weight["Theta2"] #get Theta2 in np.array

## ================= Part 3: Implement Predict =================
#  After training the neural network, we would like to use it to predict
#  the labels. You will now implement the "predict" function to use the
#  neural network to predict the labels of the training set. This lets
#  you compute the training set accuracy.

pred = predict(Theta1, Theta2, X)

print('Training Set Accuracy: {:f}'.format((np.mean(pred == y)*100)))

raw_input('Program paused. Press enter to continue.\n')

#  To give you an idea of the network's output, you can also run
#  through the examples one at a time to see what it is predicting.

#  Randomly permute examples
rp = np.random.permutation(m)

for i in xrange(m):

    # Display 
    print('Displaying Example Image')
    displayData(X[rp[i], :])

    pred = predict(Theta1, Theta2, X[rp[i], :])
    print('Neural Network Prediction: {:d} (digit {:d})'.format(pred[0], (pred%10)[0]))
    raw_input('Program paused. Press enter to continue.\n')
