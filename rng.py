import numpy as np
import random
import pandas as pd
import os
import glob
import datetime

def multiModal1():
    for i in range(1, 11):
        temp1 = np.random.normal(78, 5, size = (1, 1000))
        temp2 = np.random.normal(68, 5, size = (1, 1000))
        temp3 = np.random.normal(88, 5, size = (1, 1000))
        temp4 = np.random.normal(38, 20, size = (1, 1000))
        temp5 = np.random.normal(118, 20, size = (1,1000))
        temp1 = np.append(temp1, temp2)
        temp3 = np.append(temp3, temp4)
        temp1 = np.append(temp1, temp3)
        temp1 = np.append(temp1, temp5)

        temp6 = temp1.tolist()
        #print(temp6)
        #random.shuffle(temp6)
        #print(temp6)
        #print(type(temp6))

        co21 = np.random.normal(1.0, 0.01, size = (1, 1000))
        co22 = np.random.normal(1.2, 0.02, size = (1, 1000))
        co23 = np.random.normal(0.8, 0.01, size = (1, 1000))
        co24 = np.random.normal(0.4, 0.02, size = (1, 1000))
        co25 = np.random.normal(0.2, 0.02, size = (1,1000))
        co21 = np.append(co21, co22)
        co23 = np.append(co23, co24)
        co21 = np.append(co21, co23)
        co21 = np.append(co21, co25)
        co21 = np.absolute(co21)

        co26 = co21.tolist()
        #random.shuffle(co26)

        ppm1 = np.random.normal(10, 10, size = (1, 1000))
        ppm2 = np.random.normal(50, 20, size = (1, 1000))
        ppm3 = np.random.normal(100, 30, size = (1, 1000))
        ppm4 = np.random.normal(250,50, size = (1, 1000))
        ppm5 = np.random.normal(500, 50, size = (1,1000))
        ppm1 = np.append(ppm1, ppm2)
        ppm3 = np.append(ppm3, ppm4)
        ppm1 = np.append(ppm1, ppm3)
        ppm1 = np.append(ppm1, ppm5)
        ppm1 = np.absolute(ppm1)

        ppm6 = ppm1.tolist()
        #random.shuffle(ppm6)

        stress1 = np.random.normal(10, 10, size = (1, 1000))
        stress2 = np.random.normal(30, 10, size = (1, 1000))
        stress3 = np.random.normal(50, 10, size = (1, 1000))
        stress4 = np.random.normal(70, 10, size = (1, 1000))
        stress5 = np.random.normal(90, 10, size = (1,1000))
        stress1 = np.append(stress1, stress2)
        stress3 = np.append(stress3, stress4)
        stress1 = np.append(stress1, stress3)
        stress1 = np.append(stress1, stress5)
        stress1 = np.absolute(stress1)

        stress6 = stress1.tolist()
        #random.shuffle(stress6)

        hour = np.random.randint(0, 23, size = (1, 5000))
        minute = np.random.randint(0,59, size = (1, 5000))

        oxypress1 = np.random.normal(3.2, 0.25, size = (1, 1000))
        oxypress2 = np.random.normal(2.95, 0.5, size = (1, 1000))
        oxypress3 = np.random.normal(3.45, 0.5, size = (1, 1000))
        oxypress4 = np.random.normal(0, 1, size = (1, 1000))
        oxypress5 = np.random.normal(5, 1, size = (1,1000))
        oxypress1 = np.append(oxypress1, oxypress2)
        oxypress3 = np.append(oxypress3, oxypress4)
        oxypress1 = np.append(oxypress1, oxypress3)
        oxypress1 = np.append(oxypress1, oxypress5)
        oxypress1 = np.absolute(oxypress1)

        oxypress6 = oxypress1.tolist()
        #random.shuffle(oxypress6)

        nitpress1 = np.random.normal(14.7, 0.2, size = (1, 1000))
        nitpress2 = np.random.normal(14.9, 0.4, size = (1, 1000))
        nitpress3 = np.random.normal(14.5, 0.4, size = (1, 1000))
        nitpress4 = np.random.normal(12, 1, size = (1, 1000))
        nitpress5 = np.random.normal(16, 1, size = (1,1000))
        nitpress1 = np.append(nitpress1, nitpress2)
        nitpress3 = np.append(nitpress3, nitpress4)
        nitpress1 = np.append(nitpress1, nitpress3)
        nitpress1 = np.append(nitpress1, nitpress5)
        nitpress1 = np.absolute(nitpress1)

        nitpress6 = nitpress1.tolist()
        #random.shuffle(nitpress6)

        df = pd.DataFrame({'temperature':temp6, 'CO2': co26, 'PPM': ppm6, 'Stress': stress6, 'Oxygen Pressure': oxypress6, 'Nitrogen Pressure': nitpress6}) 

        df.to_csv('heex/cancer.csv', index = False) 

def biModal():
    for i in range(1, 2):
        temp1 = np.random.normal(78, 5, size = (1, 100))
        temp2 = np.random.normal(68, 5, size = (1, 100))
        temp1 = np.append(temp1, temp2)
        temp1 = np.absolute(temp1)

        temp6 = temp1.tolist()

        co21 = np.random.normal(1.0, 0.01, size = (1, 100))
        co22 = np.random.normal(1.2, 0.02, size = (1, 100))
        co21 = np.append(co21, co22)
        co21 = np.absolute(co21)

        co26 = co21.tolist()

        ppm1 = np.random.normal(10, 10, size = (1, 100))
        ppm2 = np.random.normal(50, 20, size = (1, 100))
        ppm1 = np.append(ppm1, ppm2)
        ppm1 = np.absolute(ppm1)

        ppm6 = ppm1.tolist()

        stress1 = np.random.normal(10, 10, size = (1, 100))
        stress2 = np.random.normal(30, 10, size = (1, 100))
        stress1 = np.append(stress1, stress2)
        stress1 = np.absolute(stress1)

        stress6 = stress1.tolist()

        hour = np.random.randint(0, 23, size = (1, 5000))
        minute = np.random.randint(0,59, size = (1, 5000))

        oxypress1 = np.random.normal(3.2, 0.25, size = (1, 100))
        oxypress2 = np.random.normal(2.95, 0.5, size = (1, 100))
        oxypress1 = np.append(oxypress1, oxypress2)
        oxypress1 = np.absolute(oxypress1)

        oxypress6 = oxypress1.tolist()

        nitpress1 = np.random.normal(14.7, 0.2, size = (1, 100))
        nitpress2 = np.random.normal(14.9, 0.4, size = (1, 100))
        nitpress1 = np.append(nitpress1, nitpress2)
        nitpress1 = np.absolute(nitpress1)

        nitpress6 = nitpress1.tolist()

        df = pd.DataFrame({'temperature':temp6, 'CO2': co26, 'PPM': ppm6, 'Stress': stress6, 'Oxygen Pressure': oxypress6, 'Nitrogen Pressure': nitpress6}) 

        df.to_csv('heex/cancer.csv', index = False) 
        

def multiModal2():
        divide = random.uniform(1./4.,1.)
        
        temp1 = np.random.normal(78, 5 / divide, size = (1, 1000))
        temp2 = np.random.normal(68, 5 / divide, size = (1, 1000))
        temp3 = np.random.normal(88, 5 / divide, size = (1, 1000))
        temp4 = np.random.normal(38, 20 / divide, size = (1, 1000))
        temp5 = np.random.normal(118, 20 / divide, size = (1, 1000))
        temp1 = np.append(temp1, temp2)
        temp3 = np.append(temp3, temp4)
        temp1 = np.append(temp1, temp3)
        temp1 = np.append(temp1, temp5)
        temp1 = np.absolute(temp1)
        temp6 = temp1.tolist()
        #print(temp6)
        #random.shuffle(temp6)
        #print(temp6)
        #print(type(temp6))

        co21 = np.random.normal(1.0, 0.01 / divide, size = (1, 1000))
        co22 = np.random.normal(1.2, 0.02 / divide, size = (1, 1000))
        co23 = np.random.normal(0.8, 0.01 / divide, size = (1, 1000))
        co24 = np.random.normal(0.4, 0.02 / divide, size = (1, 1000))
        co25 = np.random.normal(0.2, 0.02 / divide, size = (1, 1000))
        co21 = np.append(co21, co22)
        co23 = np.append(co23, co24)
        co21 = np.append(co21, co23)
        co21 = np.append(co21, co25)
        co21 = np.absolute(co21)

        co26 = co21.tolist()
        #random.shuffle(co26)

        ppm1 = np.random.normal(10, 10 / divide, size = (1, 1000))
        ppm2 = np.random.normal(50, 20 / divide, size = (1, 1000))
        ppm3 = np.random.normal(100, 30 / divide, size = (1, 1000))
        ppm4 = np.random.normal(250, 50 / divide, size = (1, 1000))
        ppm5 = np.random.normal(500, 50 / divide, size = (1, 1000))
        ppm1 = np.append(ppm1, ppm2)
        ppm3 = np.append(ppm3, ppm4)
        ppm1 = np.append(ppm1, ppm3)
        ppm1 = np.append(ppm1, ppm5)
        ppm1 = np.absolute(ppm1)

        ppm6 = ppm1.tolist()
        #random.shuffle(ppm6)

        stress1 = np.random.normal(10, 10 / divide, size = (1, 1000))
        stress2 = np.random.normal(30, 10 / divide, size = (1, 1000))
        stress3 = np.random.normal(50, 10 / divide, size = (1, 1000))
        stress4 = np.random.normal(70, 10 / divide, size = (1, 1000))
        stress5 = np.random.normal(90, 10 / divide, size = (1, 1000))
        stress1 = np.append(stress1, stress2)
        stress3 = np.append(stress3, stress4)
        stress1 = np.append(stress1, stress3)
        stress1 = np.append(stress1, stress5)
        stress1 = np.absolute(stress1)

        stress6 = stress1.tolist()
        #random.shuffle(stress6)

        hour = np.random.randint(0, 23, size = (1, 5000))
        minute = np.random.randint(0, 59, size = (1, 5000))

        oxypress1 = np.random.normal(3.2, 0.25 / divide, size = (1, 1000))
        oxypress2 = np.random.normal(2.95, 0.5 / divide, size = (1, 1000))
        oxypress3 = np.random.normal(3.45, 0.5 / divide, size = (1, 1000))
        oxypress4 = np.random.normal(0, 1 / divide, size = (1, 1000))
        oxypress5 = np.random.normal(5, 1 / divide, size = (1, 1000))
        oxypress1 = np.append(oxypress1, oxypress2)
        oxypress3 = np.append(oxypress3, oxypress4)
        oxypress1 = np.append(oxypress1, oxypress3)
        oxypress1 = np.append(oxypress1, oxypress5)
        oxypress1 = np.absolute(oxypress1)

        oxypress6 = oxypress1.tolist()
        #random.shuffle(oxypress6)

        nitpress1 = np.random.normal(14.7, 0.2 / divide, size = (1, 1000))
        nitpress2 = np.random.normal(14.9, 0.4 / divide, size = (1, 1000))
        nitpress3 = np.random.normal(14.5, 0.4 / divide, size = (1, 1000))
        nitpress4 = np.random.normal(12, 1 / divide, size = (1, 1000))
        nitpress5 = np.random.normal(16, 1 / divide, size = (1, 1000))
        nitpress1 = np.append(nitpress1, nitpress2)
        nitpress3 = np.append(nitpress3, nitpress4)
        nitpress1 = np.append(nitpress1, nitpress3)
        nitpress1 = np.append(nitpress1, nitpress5)
        nitpress1 = np.absolute(nitpress1)

        nitpress6 = nitpress1.tolist()
        #random.shuffle(nitpress6)

        df = pd.DataFrame({'temperature':temp6, 'CO2': co26, 'PPM': ppm6, 'Stress': stress6, 'Oxygen Pressure': oxypress6, 'Nitrogen Pressure': nitpress6}) 

        df.to_csv('datasets/cancer.csv', index = False) 
