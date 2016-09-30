""" Machine Learning (Supervised Learning) - Neural Network code (for sand data)

output: number of iteration, training accuracy, cross-validation accuracy, test accuracy
        total error of all three outputs
        Error-iteration Plot



Learning Algorithm: Neural Network
Supervised Learning: Regression

Algorithm content:
    activation function: sigmoid function
    hypothesis function: polynomial function
    cost function: square error
    solving partial derivatives of cost function: Backpropagation
    minimise cost function: Gradient descent


sand data:
    input parameters (9): S1, S2, S3, E1, E2, E3, E1_i+1, E2_i+1, E3_i+1 
    input parameters (3): S1_i+1, S2_i+1, S3_i+1
    
    stress in [tf] = ton-force; [1 tf] = [9.80665 kN]
    strain in [%]
"""


import tensorflow as tf
import numpy as np

'simple way to plot error curves etc.'
import Error_Plots as Plots

'importing all data files as input and output matrices; all examples are stored in a list'
import Datafiles_mechanical as soil
""" contains the following:

input/ output data

neural network structure
    all parameters for the structure are inputs in the file 
    parameters: matrices shape (input,output,weights,biases)

amount of total examples

amount of training, cross-validation, test examples (60%/20%/20%)

"""


""" Neural network / algorithm parameters                  """

'learning rate'
alpha = 3e-10

'standard deviation for the biases'
stdev_bias = 0.01
# or 1.0 (Nielson) ?

'regularisation parameter'



""" ----------------------------------------------------- """


""" Number of training data 
Splitting:      training data     - 60%
                cross-validation  - 20%
                test data         - 20%
devision was calculated in Datafiles_... 
"""
m_training = int(soil.training)
m_cv = int(soil.cv)
m_test = int(soil.test)
total_examples = soil.total

" range for running "
range_training = xrange(0,m_training)
range_cv = xrange(m_training,(m_training+m_cv))
range_test = xrange((m_training+m_cv),total_examples)

""" Using interactive Sessions
methods such as 'eval' and 'run' can run their operations without passing session
(contrast to normal 'Session' that's not interactive)
"""
sess = tf.InteractiveSession()

" creating input and output matrices "
'placeholder: matrices are empty until the data is fed during the loop'
x = tf.placeholder(tf.float32, shape=[None, soil.input_number])
y_true = tf.placeholder(tf.float32, shape=[None, soil.output_number])

""" standard deviation calculation 
equation for standard deviation: stdev = 1/sqrt(F) with F=number of input features
equation according to Nielson (http://neuralnetworksanddeeplearning.com/chap3.html)
"""
stdev = np.divide(2.0,np.sqrt(np.prod(x.get_shape().as_list()[1:])))

""" Weights and Biases 
initialise weights and biases randomly with a gaussian distribution
mean value = 0.0 (default value)
standard deviation = stdev
"""
def weights(shape):
    'shape = [line,column]'
    initial = tf.truncated_normal(shape, stddev=stdev)
    return tf.Variable(initial)

def bias(shape):
    'shape = [line,column]'
    initial = tf.truncated_normal(shape, stddev=stdev_bias)
    return tf.Variable(initial)


""" Creating weights and biases for all layers """
theta1 = weights([soil.weight_1_1,soil.weight_1_2])
bias1 = bias([1,soil.bias_1])

theta2 = weights([soil.weight_2_1,soil.weight_2_2])
bias2 = bias([1,soil.bias_2])

'Last layer'
theta3 = weights([soil.weight_3_1,soil.weight_3_2])
bias3 = bias([1,soil.bias_3])



""" hidden layer input (Sum of weights, activation functions and bias)
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



""" Cost function / Cross entropy
Error function that needs to be minimised
"""

'simple quadratic cost function'
cost_function = tf.reduce_mean(tf.mul( 0.5, tf.pow( tf.sub(hypL, y_true), 2)),1) 

'logarithmic cross entropy (advantage over cost function: reduces learning slow down)'
#Cost function can't be the logarithmic version because there are negative stress increments 
#cross_entropy2 = -tf.reduce_sum(y_true*tf.log(tf.clip_by_value(hypL,1e-10,1.0)))
cross_entropy = -tf.reduce_sum(y_true*tf.log(hypL) + (1-y_true)*tf.log(1-hypL))

"Adding L2 Regularisation term "
'regularisation parameter lamb'
#lamb = 7.0
#costfunction_reg = tf.mul( tf.div(0.5, m_training), tf.pow( tf.sub(hypL, y_true), 2)) + tf.mul(tf.div(lamb, tf.mul(2.0,m_training)), tf.reduce_mean(tf.pow(theta3,2)))

""" using dropout for regularisation - only used while training!
keep_prob: reduces overfitting by using all connection only if probability is keep_prob
https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf
"""
keep_prob = tf.placeholder(tf.float32)


""" Gradient Descent 
Minimisation of the error funtion
simple gradient descent algorithm (without regularisation): theta := theta - alpha * (d J / d theta)
"""
train_step = tf.train.GradientDescentOptimizer(learning_rate=alpha).minimize(cost_function)       
#train_step2 = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

     

"""    Training and Evaluation     """

'Checking if hypothesis (in last layer) and real output are equal'
correct_prediction = tf.equal(tf.arg_max(hypL, 1), tf.arg_max(y_true, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

'Initialising all variables'
sess.run(tf.initialize_all_variables())


""" Testing - Initialise lists  
after running, each parameters can be checked for errors / looked at individually
"""
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

'computing absolute error between hypothesis and real output'
complete_error_init = tf.abs(tf.reduce_mean(tf.sub(hypL,y_true),1))


""" Training """
'creating list training error'
training_error=[]

'using for loop over all training examples'
for j in range_training:
    'creating the feeding for the placeholders'
    feedj = {x: soil.input_matrixs[j], y_true: soil.output_matrixs[j]}
    'with dropout:'
    #feedj = {x: soil.input_matrixs[j], y_true: soil.output_matrixs[j] , keep_prob: 1.0}
    
    """ Testing - adding to list
        1. evaluate parameter by feeding input/output data
        2. adding to list """
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
    'printing absolute error between hypothesis and real output for each iteration'
    print 'number iterations: %g, error (S1, S2, S3): %g, %g, %g' % (j, complete_error[0], complete_error[1], complete_error[2])
    
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
    """ -------------------------   """
    
    'evaluate training accuracy by feeding input/output data'
    train_accuracy = accuracy.eval(feed_dict=feedj)
    'printing accuracy for each iteration'
    print("step %d, training accuracy %g" % (j, train_accuracy))
    'compute gradient descent algorithm'
    train_step.run(feed_dict=feedj)
    'adding training error to list'
    training_error.append(1 - train_accuracy)

""" Cross-Validation """ 
'using for loop over all cv examples'       
cv_error=[]    
for k in range_cv:
    feedk = {x: soil.input_matrixs[k], y_true: soil.output_matrixs[k]}
    'with dropout:'
    #feedk = {x: soil.input_matrixs[k], y_true: soil.output_matrixs[k] , keep_prob: 1.0}
    cv_accuracy = accuracy.eval(feed_dict=feedk)
    print("cross-validation accuracy %g" % cv_accuracy)
    cv_error.append(1-cv_accuracy) 

'using for loop over all test examples'
for l in range_test:
    print("test accuracy %g" % accuracy.eval(feed_dict={x: soil.input_matrixs[l], y_true: soil.output_matrixs[l]}))
    'with dropout:'
    #feedl = {x: soil.input_matrixs[l], y_true: soil.output_matrixs[l] , keep_prob: 1.0}



""" Error Analysis """

" Learning Curves "

figure1 = Plots.LearningCurves(cv_error, training_error)
 



