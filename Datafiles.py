

""" Input needed !!! 
start = number of the first data-file
end = number of the last data-file
filename = name of the folder / type of sand
"""

start = 89
end = 117
foldername = 'karlsruhe'



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
        stresses_strains = np.column_stack((self.datafiles['S1'],self.datafiles['S2'],self.datafiles['S3'],self.datafiles['E1'],self.datafiles['E2'],self.datafiles['E3'],self.datafiles['v'],self.datafiles['p']))
        Strain_increments = incr.Increment_calculations(stresses_strains).strain_increment()
        input_matrix = np.column_stack((stresses_strains,Strain_increments))
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




