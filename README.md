# NeuralNetwork_Tensorflow

Main file: 'NN_Datafiles_test1'

Main file contains/uses 'Datafiles' and 'Error_Plots'

'Error_Plots': plots the training and cross-validation error

'Datafiles': imports all necessary datafilesand and stores them into input and output matrices
input matrix is named 'input_matrix' and contains: principle stresses, principle strains, principle strain increments
output matrix is named 'output_matrix' and contains: principle stress increments

'Datafiles' contains: 'increment_calculation'

'increment_calculation': calculates the stress and strain increments and returns them in a stress and strain matrix
