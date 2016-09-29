"""
input in 'Datafiles_...' is needed (NN structure)

First Test - Neural Network for Soil data

Learning Algorithm: Neural Network
Supervised Learning: Regression

Algorithm content:
activation function: sigmoid function
hypothesis function: polynomial function
Cost function: square error
Solving partial derivatives of Cost function: GradientDescentOptimizer (Tensorflow)
Minimise cost function: Gradient descent

Data set divided into training, cross-validation, test set
"""

""" First test: only using all 28 Karlsruhe Data 
training data: 18
cross-validation data: 5
test data: 5
"""

import tensorflow as tf
'Datafiles_Order : the same as Datafiles, but with ordered datafiles'
import Datafiles_mechanical as soil
import Error_Plots as Plots
import numpy as np

""" Number of training data 
Splitting:      training data     - 60%
                cross-validation  - 20%
                test data         - 20%
"""


m_training = int(soil.training)
m_cv = int(soil.cv)
m_test = int(soil.test)
total_examples = soil.total

" range for running "
range_training = xrange(0,m_training)
range_cv = xrange(m_training,(m_training+m_cv))
range_test = xrange((m_training+m_cv),total_examples)

""" Using interactive Sessions"""
sess = tf.InteractiveSession()

""" creating input and output vectors """
x = tf.placeholder(tf.float32, shape=[None, soil.input_number])
y_true = tf.placeholder(tf.float32, shape=[None, soil.output_number])

""" Standard Deviation Calculation"""
def st_dev(input_units):
    stdev = np.divide(2.0,np.sqrt(np.prod(input_units.get_shape().as_list()[1:])))
    return stdev

stdev = np.divide(2.0,np.sqrt(np.prod(x.get_shape().as_list()[1:])))

#stdev_1 = st_dev(soil.input_number)
#stdev_2 = st_dev(soil.weight_1_2)
#stdev_3 = st_dev(soil.weight_2_2)

""" Weights and Biases """

def weights(shape):
    """ shape = [row,column]"""
    initial = tf.truncated_normal(shape, stddev=stdev)
    return tf.Variable(initial)

def bias(shape):
    """ shape = [row,column]"""
    #initial = tf.constant(0.1, shape=shape)
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)

""" Creating weights and biases for all layers """
theta1 = weights([soil.weight_1_1,soil.weight_1_2])
bias1 = bias([1,soil.bias_1])

theta2 = weights([soil.weight_2_1,soil.weight_2_2])
bias2 = bias([1,soil.bias_2])

theta3 = weights([soil.weight_3_1,soil.weight_3_2])
bias3 = bias([1,soil.bias_3])

"Last layer"
#theta4 = weights([soil.weight_4_1,soil.weight_4_2])
#bias4 = bias([1,soil.bias_4])


""" Hidden layer input (Sum of weights, activation functions and bias)
z = theta^T * activation + bias
"""
def Z_Layer(activation,theta,bias):
    return tf.add(tf.matmul(activation,theta),bias)

""" Creating the sigmoid function 
sigmoid = 1 / (1 + exp(-z))
"""
def Sigmoid(z):
    return tf.div(tf.constant(1.0),tf.add(tf.constant(1.0), tf.exp(tf.neg(z))))

""" hypothesis functions - predicted output """    
' layer 1 - input layer '
hyp1 = x
' layer 2 '
z2 = Z_Layer(hyp1, theta1, bias1)
hyp2 = Sigmoid(z2)
' layer 3 '
z3 = Z_Layer(hyp2, theta2, bias2)
hyp3 = Sigmoid(z3)
' layer 4 - output layer '
zL = Z_Layer(hyp3, theta3, bias3)
#hypL = Sigmoid(zL)
hypL = tf.add( tf.add(tf.pow(zL,3), tf.pow(zL,2) ), zL)


""" Cost function """
" Cost function can't be the logarithmic version because there are negative stress increments "

"Adding L2 Regularisation term "
'regularisation parameter lamb'
#lamb = 7.0
#costfunction_reg = tf.mul( tf.div(0.5, m_training), tf.pow( tf.sub(hypL, y_true), 2)) + tf.mul(tf.div(lamb, tf.mul(2.0,m_training)), tf.reduce_mean(tf.pow(theta3,2)))
costfunction = tf.reduce_mean(tf.mul( 0.5, tf.pow( tf.sub(hypL, y_true), 2)),1) 

cost_function = costfunction
#cross_entropy2 = -tf.reduce_sum(y_true*tf.log(tf.clip_by_value(hypL,1e-10,1.0)))
cross_entropy = -tf.reduce_sum(y_true*tf.log(hypL) + (1-y_true)*tf.log(1-hypL))
""" Gradient Descent """
train_step = tf.train.GradientDescentOptimizer(learning_rate=1e-50).minimize(cost_function)       
#train_step2 = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
     

"""    Training and Evaluation     """

correct_prediction = tf.equal(tf.arg_max(hypL, 1), tf.arg_max(y_true, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.initialize_all_variables())

keep_prob = tf.placeholder(tf.float32)

""" Testing - Initialise lists  """
hyp1_test = []
z2_test = []
hyp2_test = []
z3_test = []
hyp3_test = []
zL_test = []
hypL_test = []
cost_function_test =[]
complete_error_test = []
theta1_test = []
theta2_test = []
theta3_test = []
bias1_test = []
bias2_test = []
bias3_test = []
""" -------------------------   """

complete_error_init = tf.abs(tf.reduce_mean(tf.sub(hypL,y_true),1))
    
cost_training = []
training_error=[]
for j in range_training:
    feedj = {x: soil.input_matrixs[j], y_true: soil.output_matrixs[j] , keep_prob: 1.0}
    
    """ -------------------------   """
    'Testing - adding to list'
    #hyp1_test.append(hyp1)
    z2_init = z2.eval(feed_dict=feedj)
    z2_test.append(z2_init)
    
    hyp2_init = hyp2.eval(feed_dict=feedj)
    hyp2_test.append(hyp2_init)
    
    z3_init = z3.eval(feed_dict=feedj)
    z3_test.append(z3_init)
    
    hyp3_init = hyp3.eval(feed_dict=feedj)
    hyp3_test.append(hyp3_init)
    
    zL_init = zL.eval(feed_dict=feedj)
    zL_test.append(zL_init)
    
    hypL_init = hypL.eval(feed_dict=feedj)
    hypL_test.append(hypL_init)
    
    cost_function_init = cost_function.eval(feed_dict=feedj)
    cost_function_test.append(cost_function_init)
    
    complete_error = complete_error_init.eval(feed_dict=feedj)
    complete_error_test.append(complete_error)
    print 'number iterations: %g, error (S1, S2, S3): %g, %g, %g' % (j, complete_error[0], complete_error[1], complete_error[2])
    #print('complete error - S1: %g, S2: %g, S3: %g' % (complete_error[0], complete_error[1],complete_error[2]))
     
    theta1_init = theta1.eval()
    theta1_test.append(theta1_init)
      
    theta2_init = theta2.eval()
    theta2_test.append(theta2_init)
      
    theta3_init = theta3.eval()
    theta3_test.append(theta3_init)
      
    bias1_init = bias1.eval()
    bias1_test.append(bias1_init)
      
    bias2_init = bias2.eval()
    bias2_test.append(bias2_init)
      
    bias3_init = bias3.eval()
    bias3_test.append(bias3_init)
    #print 'one'
    """ -------------------------   """
    
    train_accuracy = accuracy.eval(feed_dict=feedj)
    #cost = cost_function.eval(feed_dict=feedj)
    #cost_training.append(cost)
    print("step %d, training accuracy %g" % (j, train_accuracy))
    #print("cost function %g" % cost[50])
    train_step.run(feed_dict=feedj)
    training_error.append(1 - train_accuracy)
        
cv_error=[]    
for k in range_cv:
    feedk = {x: soil.input_matrixs[k], y_true: soil.output_matrixs[k] , keep_prob: 1.0}
    cv_accuracy = accuracy.eval(feed_dict=feedk)
    print("cross-validation accuracy %g" % cv_accuracy)
    cv_error.append(1-cv_accuracy) 

for l in range_test:
    print("test accuracy %g" % accuracy.eval(feed_dict={x: soil.input_matrixs[l], y_true: soil.output_matrixs[l], keep_prob: 1.0}))


""" Error Analysis """

" Learning Curves "

figure1 = Plots.LearningCurves(cv_error, training_error)
 



