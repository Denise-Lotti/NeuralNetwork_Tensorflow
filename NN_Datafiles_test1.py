"""
First Test - Neural Network for Soil data

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

import tensorflow as tf
import Datafiles as soil
import Error_Plots as Plots

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
range_training = range(0,m_training)
range_cv = range(m_training,(m_training+m_cv))
range_test = range((m_training+m_cv),total_examples)

""" Using interactive Sessions"""
sess = tf.InteractiveSession()

""" creating input and output vectors """
x = tf.placeholder(tf.float32, shape=[None, 11])
y_true = tf.placeholder(tf.float32, shape=[None, 3])

""" Weights and Biases """

def weights(shape):
    """ shape = [row,column]"""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias(shape):
    """ shape = [row,column]"""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

""" Creating weights and biases for all layers """
theta1 = weights([11,7])
bias1 = bias([1,7])

theta2 = weights([7,7])
bias2 = bias([1,7])

#theta3 = weights([7,7])
#bias3 = bias([1,7])

"Last layer"
thetaL = weights([7,3])
biasL = bias([1,3])


""" hypothesis functions - predicted output """    
' layer 1 - input layer '
hpy1 = x
' layer 2 '
z2 = tf.matmul(x,theta1) + bias1
hyp2 = 100.0 / (1.0 - tf.exp(-z2))
' layer 3 '
z3 = tf.matmul(hyp2,theta2) + bias2
hyp3 = 100.0 / (1.0 - tf.exp(-z3))
' layer 4 - output layer '
zL = tf.matmul(hyp3,thetaL) + biasL
hypL = tf.pow(zL,3) + tf.pow(zL,2) + zL

""" Cost function """
" Cost function can't be the logarithmic version because there are negative stress increments "
cost_function = (0.5/m_training) *  (hypL - y_true)**2 


""" Gradient Descent """
train_step = tf.train.GradientDescentOptimizer(learning_rate=0.003).minimize(cost_function)       
        

"""    Training and Evaluation     """

correct_prediction = tf.equal(tf.arg_max(hypL, 1), tf.arg_max(y_true, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.initialize_all_variables())

keep_prob = tf.placeholder(tf.float32)

training_error=[]
for j in range_training:
    train_accuracy = accuracy.eval(feed_dict={x: soil.input_matrixs[j], y_true: soil.output_matrixs[j]})
    print("step %d, training accuracy %g" % (j, train_accuracy))
    train_step.run(feed_dict={x: soil.input_matrixs[j], y_true: soil.output_matrixs[j]})
    training_error.append(1 - train_accuracy)

cv_error=[]    
for k in range_cv:
    cv_accuracy = accuracy.eval(feed_dict={x: soil.input_matrixs[k], y_true: soil.output_matrixs[k], keep_prob: 1.0})
    print("cross-validation accuracy %g" % cv_accuracy)
    cv_error.append(1-cv_accuracy) 

for l in range_test:
    print("test accuracy %g" % accuracy.eval(feed_dict={x: soil.input_matrixs[k], y_true: soil.output_matrixs[k], keep_prob: 1.0}))


""" Error Analysis """

" Learning Curves "

figure1 = Plots.LearningCurves(cv_error, training_error)
 



