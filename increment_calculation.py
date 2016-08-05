import numpy as np

class Increment_calculations(object):
    
    def __init__(self, datafiles):
        self.matrix = datafiles
        
    def strain_increment(self):
        """ Initialise strain increment vectors """
        E1_i1 = np.zeros((len(self.matrix),1))
        E2_i1 = np.zeros((len(self.matrix),1))
        E3_i1 = np.zeros((len(self.matrix),1))
        """ Setting first entry to zero (no strain increment at the first time step) """
        E1_i1[0,0] = 0
        E2_i1[0,0] = 0
        E3_i1[0,0] = 0
        """ Computing strain increment by calculating the difference between next time step and current time step """
        for j in range(0,len(self.matrix)-1):
            E1_i1[j+1] = self.matrix[j+1,3] - self.matrix[j,3]
            E2_i1[j+1] = self.matrix[j+1,4] - self.matrix[j,4] 
            E3_i1[j+1] = self.matrix[j+1,5] - self.matrix[j,5] 
        E_i1 = np.column_stack((E1_i1,E2_i1,E3_i1))
        return E_i1  

    def real_stress_increment(self):
        """ Initialise stress increment vectors """
        S1_i1 = np.zeros((len(self.matrix),1))
        S2_i1 = np.zeros((len(self.matrix),1))
        S3_i1 = np.zeros((len(self.matrix),1))
        """ Setting first entry to zero (no stress increment at the first time step) """
        S1_i1[0,0] = 0
        S2_i1[0,0] = 0
        S3_i1[0,0] = 0
        """ Computing stress increment by calculating the difference between next time step and current time step """
        for j in range(0,len(self.matrix)-1):
            S1_i1[j+1] = self.matrix[j+1,0] - self.matrix[j,0]
            S2_i1[j+1] = self.matrix[j+1,1] - self.matrix[j,1] 
            S3_i1[j+1] = self.matrix[j+1,2] - self.matrix[j,2] 
   
        S_i1 = np.column_stack((S1_i1,S2_i1,S3_i1))
        return S_i1 
        

""" for testing """

"""
filename = "karlsruhe/{0}.dat".format(89)
datafile = gosl.Read(filename)
stress = np.column_stack((datafile['S1'],datafile['S2'],datafile['S3']))
strain = np.column_stack((datafile['E1'],datafile['E2'],datafile['E3']))

Stress_matrix = Increment_calculations(stress).real_stress_increment()
Strain_matrix = Increment_calculations(strain).strain_increment()
print Strain_matrix
print 'hello'
"""
