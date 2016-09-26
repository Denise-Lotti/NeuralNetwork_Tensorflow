"""
Neural Network for Soil data with Backpropagation

Cite: http://blog.aloni.org/posts/backprop-with-tensorflow/

Learning Algorithm: Neural Network
Supervised Learning: Regression

Algorithm content:
activation function: sigmoid function
hypothesis function: polynomial function
Cost function: square error
Solving partial derivatives of Cost function: Backpropagation
Minimise cost function: Gradient descent

Data set divided into training, cross-validation, test set
"""

""" First test: only using all 28 Karlsruhe Data 
training data: 18
cross-validation data: 5
test data: 5
"""

#import tensorflow as tf
import Datafiles as soil
#import Error_Plots as Plots
import numpy as np
import matplotlib.pyplot as plt

""" Number of training data 
Splitting:      training data     - 60%
                cross-validation  - 20%
                test data         - 20%
"""

m_training = soil.m
x_input = soil.input_scale[0:m_training]
y_true = soil.output_scale[0:m_training] 

'initial values for theta and bias'
n_start = 0.01
n_end = 10.0

theta1 = np.random.uniform(n_start, n_end, size=[soil.weight_1_1,soil.weight_1_2])
theta2 = np.random.uniform(n_start, n_end, size=[soil.weight_2_1,soil.weight_2_2])
theta3 = np.random.uniform(n_start, n_end, size=[soil.weight_3_1,soil.weight_3_2])

bias1 = np.random.uniform(n_start, n_end, size=[1,soil.bias_1])
bias2 = np.random.uniform(n_start, n_end, size=[1,soil.bias_2])
bias3 = np.random.uniform(n_start, n_end, size=[1,soil.bias_3])


def Z_Layer(activation,theta,bias):
    return np.add(np.dot(activation,theta),bias)

def Sigmoid(z):
    return np.true_divide(1.0,np.add(1.0, np.exp(-z)))

def Diff_Sigmoid(z):
    return np.multiply(1.0 ,np.divide(np.exp(z), np.add(np.exp(z),1)**2 ))
 

cost_training = [] 
training_accuracy=[]
threshold_error = 1e-1
training_error = []

'Testing - Initialise lists'
z2_test = []
hyp2_test = []
z3_test = []
hyp3_test = []
zL_test = []
hypL_test = []
cost_function_test =[]
complete_error_test = []
deltaL_test = []
delta3_test = []
delta2_test = []
part_theta3_test = []
part_theta2_test = []
part_theta1_test = []
part_bias3_test = []
part_bias2_test = []
part_bias1_test = []
theta1_it_test = []
theta2_it_test = []
theta3_it_test = []
bias1_it_test = []
bias2_it_test = []
bias3_it_test = []

for i in range(0,m_training):
    'Forward Propagation'
    
    ' layer 1 - input layer '
    hyp1 = x_input[i]
    
    for iteration in range(500):   
        ' layer 2 '
        z2 = Z_Layer(hyp1, theta1, bias1)
        hyp2 = Sigmoid(z2)
        ' layer 3 '
        z3 = Z_Layer(hyp2, theta2, bias2)
        hyp3 = Sigmoid(z3)
        ' layer 4 - output layer '
        zL = Z_Layer(hyp3, theta3, bias3)
        hypL = np.add( np.add(zL**3, zL**2 ), zL) 
        #hypL = zL
        #hypL = tf.div(tf.constant(1.0),tf.add(tf.constant(1.0), tf.exp(tf.neg(zL))))
        #hypL = tf.add(tf.exp(zL), tf.pow(zL,2))
        
        """ Cost function """
        cost_function = np.multiply( 0.5, np.subtract(hypL, y_true[i])**2) 
        #cost_function = tf.reduce_sum(cost,0)
    
        complete_error = np.mean(np.subtract(hypL,y_true[i]),axis=0)
        print 'number iterations: %g, error (S1, S2, S3): %g, %g, %g' % (iteration, complete_error[0], complete_error[1], complete_error[2])
        
               
        'Backpropagation'
    
        ' error terms delta '
        deltaL = np.multiply(np.subtract(hypL,y_true[i]) , np.add(3*zL**2 + 2*zL , 1.0))
        delta3 = np.multiply(np.dot(deltaL, np.transpose(theta3)) , Diff_Sigmoid(z3) )
        delta2 = np.multiply(np.dot(delta3, np.transpose(theta2)) , Diff_Sigmoid(z2) )
        
        ' partial derivatives '
        part_theta3 = np.dot(np.transpose(hyp3) , deltaL)
        part_theta2 = np.dot(np.transpose(hyp2) , delta3)
        part_theta1 = np.dot(np.transpose(hyp1) , delta2)
       
        part_bias3 = np.mean(deltaL, axis=0)
        part_bias2 = np.mean(delta3, axis=0)
        part_bias1 = np.mean(delta2, axis=0)

        ' Update weights and biases '
        alpha = 0.003
        theta1_it = np.subtract(theta1, np.multiply(alpha, part_theta1))
        theta2_it = np.subtract(theta2, np.multiply(alpha, part_theta2))
        theta3_it = np.subtract(theta3, np.multiply(alpha, part_theta3))
        bias1_it = np.subtract(bias1, np.multiply(alpha, part_bias1))
        bias2_it = np.subtract(bias2, np.multiply(alpha, part_bias2))
        bias3_it = np.subtract(bias3, np.multiply(alpha, part_bias3))
        
        'Testing - adding to list'
        z2_test.append(z2)
        hyp2_test.append(hyp2)
        z3_test.append(z3)
        hyp3_test.append(hyp3)
        zL_test.append(zL)
        hypL_test.append(hypL)
        cost_function_test.append(cost_function)
        complete_error_test.append(complete_error)
        deltaL_test.append(deltaL)
        delta3_test.append(delta3)
        delta2_test.append(delta2)
        part_theta3_test.append(part_bias3)
        part_theta2_test.append(part_theta2)
        part_theta1_test.append(part_theta1)
        part_bias3_test.append(part_bias3)
        part_bias2_test.append(part_bias2)
        part_bias1_test.append(part_bias1)
        theta1_it_test.append(theta1_it)
        theta2_it_test.append(theta2_it)
        theta3_it_test.append(theta3_it)
        bias1_it_test.append(bias1_it)
        bias2_it_test.append(bias2_it)
        bias3_it_test.append(bias3_it)
        
        theta1 = theta1_it
        theta2 - theta2_it
        theta3 = theta3_it
        bias1 = bias1_it
        bias2 = bias2_it
        bias3 = bias3_it
        
        
       
    'Evaluation'
    correct_prediction = np.equal(np.argmax(hypL, axis=1), np.argmax(y_true[i], axis=1))
    
    #correct_prediction = np.absolute(np.subtract(np.divide(hypL,y_true[i]),1))
    #correct_prediction = np.divide(np.fabs(np.subtract(hypL,y_true[i])),y_true[i])
    
    accuracy = np.mean(correct_prediction)
    #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("step %d, training accuracy %g" % (i, accuracy))
    #print("Weights 3: %f" % theta3[0])
    
    training_accuracy.append(accuracy)
    cost_training.append(cost_function)
    training_error.append(1 - accuracy)
    #print 'some'
   

" Learning Curves "

plt.plot(training_error,'--ro')
plt.show()
#figure1 = Plots.LearningCurvesT(cv_error, training_error,'error plot')
#figure2 = Plots.LearningCurvesT(cost_function_validation, cost_function_training, 'cost functions')


