""" ----------------------------------
Creation of the input and output matrices by loading the data files
    only mechanical parameters are used (geometrical ones are ignored)
  
  
1.    loading data files + calculating increments
2.    normalise/scale data
3.    data is stored in input and output matrices in a random order

4.    calculating amount of training, cv, and test data 

    
    input parameters (9): S1, S2, S3, E1, E2, E3, E1_i+1, E2_i+1, E3_i+1 
    input parameters (3): S1_i+1, S2_i+1, S3_i+1
    
    S1, S2, S3 / E1, E2, E3 : principle stresses/strains
    E1_i+1, E2_i+1, E3_i+1 / S1_i+1, S2_i+1, S3_i+1 : principle stress/strain increments
    
    stress in [tf] = ton-force; [1 tf] = [9.80665 kN]
    strain in [%]

Input is needed about the filenames and the neural network structure 

    start         number of the first data-file
    end           number of the last data-file
    filename      name of the folder / type of sand
    
    input_number    number of input features
    output_number     number of output features
    
    weight structure: shape = [line,column]
    
    weight_1_1         first weight - number of lines = number of input features
    weight_1_2         first weight - number of columns
    weight_2_1         second weight - number of lines = lines of first weight
    weight_2_2         second weight - number of columns
    ...
    columns last weight = number output features
            
    bias_1             first bias - number of columns = columns first weight
    bias_2             second bias - number of columns = columns second weight
    ...
    columns last bias = number output features 
    
output: input and output array/list which contains each example 
        each example is a matrix with input/output values

"""

start = 89
end = 117
foldername = 'karlsruhe'


""" Neural Network parameter/structure """
'number of hidden units = mean value of input and output units'
input_number = 9
output_number = 3
weight_1_1 = 9
weight_1_2 = 6
weight_2_1 =6
weight_2_2 = 6
weight_3_1 = 6
weight_3_2 = 3
#weight_4_1 = 6
#weight_4_2 = 3
bias_1 = 6
bias_2 = 6
bias_3 = 3
#bias_4 = 3

'-------------------------------------------------------------------------------------------'

'yoolbox to import datafiles in an easy way '
'necessary: each column of the data-sheet contains the name/symbol of the its column '
import tb_gosl as gosl
' calculates the increments for each principle stress and strain value '
import increment_calculation as incr
 
import numpy as np
import random



class DataFiles(object):
    
    def __init__(self, filenumbers, filename):
        
        'file number = numbers between start and end'
        self.filnumbers = filenumbers
        'directory of the data files'
        filenames = "sand/%s/{0}.dat".format(filenumbers) % (foldername)
        'function to read data files'
        datafiles = gosl.Read(filenames)
        self.datafiles = datafiles
  
    def create_input_matrix(self):
        'loading stress and strains and storing it in one matrix'
        stresses_strains = np.column_stack((self.datafiles['S1'],self.datafiles['S2'],self.datafiles['S3'],self.datafiles['E1'],self.datafiles['E2'],self.datafiles['E3']))
        'calculating strain increments'
        Strain_increments = incr.Increment_calculations(stresses_strains).strain_increment()
        'storing loaded stress/strain and strain increments in one matrix'
        input_matrix = np.column_stack((stresses_strains,Strain_increments))
        return input_matrix 
  
    def create_output_matrix(self):
        'loading stress and storing it in one matrix'
        stresses = np.column_stack((self.datafiles['S1'],self.datafiles['S2'],self.datafiles['S3']))
        'calculating stress increments (output)'
        Stress_increments = incr.Increment_calculations(stresses).stress_increment()
        return Stress_increments



""" creating list of all input/output data with for loop over all examples """

'creating empty list for input and output (storing each example with append comment in loop)'
input_matrixs = []
output_matrixs = []

""" Creating file numbers as random list """
filenumbers = []
for entry in range(start,end):
    filenumbers.append(entry)
random.shuffle(filenumbers)

'checking NaN problems by storing all norms of the output data'
'problem: E2,E3 are constant for some tests, so their increments are 0; zero vector has norm 0; dividing by zero results in NaN'

for i in range(0,len(filenumbers)):
    'creating one input and output matrix for each example'
    input_matrix = DataFiles(filenumbers[i],foldername).create_input_matrix()
    output_matrix = DataFiles(filenumbers[i],foldername).create_output_matrix()
       
          
    """ scaling data by dividing each value with its norm """
    'calculating norm of each vector in each example'
    norm_x = np.linalg.norm(input_matrix,axis=0)
    norm_y = np.linalg.norm(output_matrix,axis=0)  
                
    'Referring to NaN problem: changing each norm value which has a zero value to 1'
    for j in range(0,len(norm_x)):
        if norm_x[j] == 0.0: 
            norm_x[j]= 1.0
    for j in range(0,len(norm_y)):    
        if norm_y[j] == 0.0: 
            norm_y[j]= 1.0
         
    """ --------------- """
    'Dividing each value by its norm'
    scale_input = np.divide(input_matrix,norm_x)
    scale_output = np.divide(output_matrix,norm_y)
    
    
    """ normalising/scaling by max and min values
    equation: y = (x - min) / (max - min) """
#     denominator_x = np.subtract(np.amax(input_matrix, axis=0),np.amin(input_matrix, axis=0))
#     denominator_y = np.subtract(np.amax(output_matrix, axis=0),np.amin(output_matrix, axis=0))
#                    
#     'Referring to NaN problem: changing each denominator value which has a zero value to 1'
#     for j in range(0,len(denominator_x)):
#         if denominator_x[j] == 0.0: 
#             denominator_x[j]= 1.0
#     for j in range(0,len(denominator_y)):    
#         if denominator_y[j] == 0.0: 
#             denominator_y[j]= 1.0
#     
#     scale_input = np.divide(np.subtract(input_matrix,np.amin(input_matrix, axis=0)) , denominator_x)
#     scale_output = np.divide(np.subtract(output_matrix,np.amin(output_matrix, axis=0)) , denominator_y)
    
    'append each single example into the input/output list (output of the whole code)'
    input_matrixs.append(scale_input)
    output_matrixs.append(scale_output)


    
""" separate all examples into training (60%), cv (20%), test (20%) data """
'if conditions are used because of possible non-natural training numbers'

'total number of examples'
total = len(filenumbers)
training = 0.6*total
if (total-training) % 2 != 0:
    training = round(training)
cv = (total - training) / 2
test = (total - training) / 2
if cv != int(cv):
    training += 1
    cv = (total - training) / 2
    test = (total - training) / 2 
if (training+cv+test) != total:
    print 'number of separated data is not right'
if training != int(training) or cv != int(cv) or test != int(test):
    print 'Error - some float number'



'---------------------------------------------------------------------------------------------------------'
""" Testing """

# 
# """ testing NaN values """
# test_input = np.isnan(np.sum(input_matrixs[i]))
# test_output = np.isnan(np.sum(output_matrixs[i]))
# print 'test input %g: %s' % (i, test_input)
# print 'test output %g: %s' % (i, test_output) 
# 
# 
# """ testing scaling """
# test_betrag_all = []
# 
# for k in range(0,total):
#     test_matrix = input_matrixs[k]
#     test_betrag = np.linalg.norm(test_matrix,axis=0)
#     test_betrag_all.append(test_betrag)
# ' all entries in test_betrag_all should be around 1.0'
    
""" --- """

