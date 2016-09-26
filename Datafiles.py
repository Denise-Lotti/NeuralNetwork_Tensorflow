

""" Input needed !!! 
start = number of the first data-file
end = number of the last data-file
filename = name of the folder / type of sand
"""

start = 89
end = 117
foldername = 'karlsruhe'
'number of examples'
m = end - start

""" Neural Network parameter """
input_number = 11
output_number = 3
weight_1_1 = 11
weight_1_2 = 7
weight_2_1 =7
weight_2_2 = 7
weight_3_1 = 7
weight_3_2 = 3
bias_1 = 7
bias_2 = 7
bias_3 = 3

'-------------------------------------------------------------------------------------------'

""" Toolbox to import datafiles in an easy way 
Necessary: each column of the data-sheet contains the name/symbol of the its column 
"""
import tb_gosl as gosl
' Calculates the increments for each stress and strain value '
import increment_calculation as incr
 
import numpy as np
import random



class DataFiles(object):
    
    def __init__(self, filenumbers, filename):
        """
        constructor
        ===========
        
        EXAMPLE:
            datafiles = DataFiles(i) with i = start:end
        INPUT:
            filenumbers -- the filenumbers of each data-file
            foldername    -- the name of the folder (type of sand / sand name); example: 'karlsruhe'
        STORED:
            datafiles -- dictionaries read by Read (raw data)
        """
        
        self.filnumbers = filenumbers
        filenames = "sand/%s/{0}.dat".format(filenumbers) % (foldername)
        datafiles = gosl.Read(filenames)
        self.datafiles = datafiles
  
    def create_input_matrix(self):
        stresses_strains = np.column_stack((self.datafiles['S1'],self.datafiles['S2'],self.datafiles['S3'],self.datafiles['E1'],self.datafiles['E2'],self.datafiles['E3']))
        Strain_increments = incr.Increment_calculations(stresses_strains).strain_increment()
        '-------------------'
        'Adding zeros to v and p'
        v_p = np.column_stack((self.datafiles['v'][0],self.datafiles['p'][0]))
        lines = len(stresses_strains) - 1
        zeros_matrix = np.zeros((lines,2))
        v_p_matrix = np.vstack((v_p,zeros_matrix))
        '-------------------'
        input_matrix = np.column_stack((stresses_strains,Strain_increments,v_p_matrix))
        return input_matrix 
  
    def create_output_matrix(self):
        stresses = np.column_stack((self.datafiles['S1'],self.datafiles['S2'],self.datafiles['S3']))
        Stress_increments = incr.Increment_calculations(stresses).real_stress_increment()
        return Stress_increments


input_matrixs = []
output_matrixs = []

""" Creating filenumbers as random list """
filenumbers = []
for entry in range(start,end):
    filenumbers.append(entry)
random.shuffle(filenumbers)

for i in range(0,len(filenumbers)):
    input_matrixs.append(DataFiles(filenumbers[i],foldername).create_input_matrix())
    output_matrixs.append(DataFiles(filenumbers[i],foldername).create_output_matrix())


'Without Feature Scaling'
input_scale = input_matrixs
output_scale = output_matrixs

# 'Feature Scaling'
# betrag_x=[]
# betrag_y = []
# for b in range(0,m):
#     betrag_x.append(np.linalg.norm(input_matrixs[b]))
#     betrag_y.append(np.linalg.norm(output_matrixs[b]))
#  
# output_scale = []
# input_scale = []
# for i in range(0,m):
#     input_scale.append(np.divide(input_matrixs[i],betrag_x[i]))
#     output_scale.append(np.divide(output_matrixs[i],betrag_y[i]))


""" seperate all examples into training, cv, test data """
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
    print 'number of seperated data is not right'
if training != int(training) or cv != int(cv) or test != int(test):
    print 'Error - some float number'



