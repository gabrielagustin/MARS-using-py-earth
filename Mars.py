# -*- coding: utf-8 -*-
import numpy as np
from pyearth import Earth
import lectura
import selection
import statistics
import statsmodels.formula.api as smf
from pyearth import export



def mars(porc, file, rand):

#    file = "tabla_calibration_validation.csv"
    #file = 'tabla_Completa.csv'

    data = lectura.lecturaCompletaMLP_etapa1(file)

    #data = lectura.lecturaCompleta_etapa1(file)
    #print data



    ## se mezclan las observaciones de las tablas
    ## semilla para mezclar los datos en forma aleatoria


    np.random.seed(rand)
    dataNew = selection.shuffle(data)
    dataNew = dataNew.reset_index(drop=True)
    #porc = 0.75
    print ("Porcentaje de datos de calculo: " + str(porc))

    ### division de los datos para entrenamiento y prueba
    nRow = len(dataNew.index)
    numTraining=int(round(nRow)*porc)
    print("Cantidad de elementos para el calculo de coeff: " + str(numTraining))
    numTest=int((nRow)-numTraining)
    print("Cantidad de elementos para prueba: " +str(numTest))


    dataTraining =  dataNew.ix[:numTraining, :]
    dataTraining = selection.shuffle(dataTraining)
    dataTraining = dataTraining.reset_index(drop=True)
    dataTest = dataNew.ix[numTraining + 1:, :]


    #OldRange = (np.max(dataTraining.T_aire)  - np.min(dataTraining.T_aire))
    #NewRange = (1)
    #dataTraining.T_aire = (((dataTraining.T_aire - np.min(dataTraining.T_aire)) * NewRange) / OldRange)

    #OldRange = (np.max(dataTraining.HR)  - np.min(dataTraining.HR))
    #NewRange = (1)
    #dataTraining.HR = (((dataTraining.HR - np.min(dataTraining.HR)) * NewRange) / OldRange)

    #OldRange = (np.max(dataTraining.PP)  - np.min(dataTraining.PP))
    #NewRange = (1)
    #dataTraining.PP = (((dataTraining.PP - np.min(dataTraining.PP)) * NewRange) / OldRange)

    #OldRange = (np.max(dataTraining.Sigma0)  - np.min(dataTraining.Sigma0))
    #NewRange = (1)
    #dataTraining.Sigma0 = (((dataTraining.Sigma0 - np.min(dataTraining.Sigma0)) * NewRange) / OldRange)

    print ("DataTraining")
    print(dataTraining)
    print (dataTraining.describe())


    #OldRange = (np.max(dataTest.T_aire)  - np.min(dataTest.T_aire))
    #NewRange = (1)
    #dataTest.T_aire = (((dataTest.T_aire - np.min(dataTest.T_aire)) * NewRange) / OldRange)

    #OldRange = (np.max(dataTest.HR)  - np.min(dataTest.HR))
    #NewRange = (1)
    #dataTest.HR = (((dataTest.HR - np.min(dataTest.HR)) * NewRange) / OldRange)

    #OldRange = (np.max(dataTest.PP)  - np.min(dataTest.PP))
    #NewRange = (1)
    #dataTest.PP = (((dataTest.PP - np.min(dataTest.PP)) * NewRange) / OldRange)

    #OldRange = (np.max(dataTest.Sigma0)  - np.min(dataTest.Sigma0))
    #NewRange = (1)
    #dataTest.Sigma0 = (((dataTest.Sigma0 - np.min(dataTest.Sigma0)) * NewRange) / OldRange)

    print ("DataTest")
    print (dataTest.describe())


    yTraining = dataTraining['SM_CONAE']
    del dataTraining['SM_CONAE']
    yTest = dataTest['SM_CONAE']
    del dataTest['SM_CONAE']

    # definicion del modelo
    model = Earth(max_degree=1, max_terms=16, penalty=1.25)
    ### max_terms=1000, max_degree=1, verbose=1 , penalty=2
    # Calibracion del modelo
    model.fit(dataTraining,yTraining)
    yCal = model.predict(dataTraining)

    #Print the model
    print(model.trace())
    print(model.summary())

    #return sympy expression
    print('Resulting sympy expression:')
    print(export.export_sympy(model))


    # Validacion del modelo
    yAprox = model.predict(dataTest)

     # Error de Calibracion
    print ("Calibracion MARS: ")
    rmse = statistics.RMSE(np.array(yTraining), yCal)
    print ("RMSE:" + str(rmse))
    RR = smf.ols('yTraining ~ 1+ yCal', dataTraining).fit().rsquared
    print ("R^2: "+str(RR))

    bias = statistics.bias(np.array(yTraining), yCal)
    print ("Bias:" + str(bias))


    #fig = plt.figure(2,facecolor="white")
    #fig2 = fig.add_subplot(111,aspect='equal')
    ##fig1, ax = plt.subplots()

    #fig2.set_title('Validacion')
    #fig2.scatter(yTest, yAprox, s=10, color='black',linewidth=3, label='MARS')
    #fig2.axis([5,45, 5,45])
    #plt.grid(True)
    #x = np.linspace(*fig2.get_xlim())
    #fig2.plot(x, x, linestyle="--", color='black')

    print ("Validacion MARS:")
    rmse = statistics.RMSE(np.array(yTest), yAprox)
    print ("RMSE:" + str(rmse))
    RR = smf.ols('yTest ~ 1+ yAprox', dataTest).fit().rsquared
    print ("R^2: "+str(RR))

    bias = statistics.bias(np.array(yTest), yAprox)
    print ("Bias:" + str(bias))


    return model, yCal, yAprox
####---------------------------------------------------------------------------
    
def mars_SAR_SMAP(porc, file, rand):

#    file = "tabla_calibration_validation.csv"
    #file = 'tabla_Completa.csv'

    data = lectura.lecturaCompletaMLP_etapa1_SAR_SMAP(file)

    #data = lectura.lecturaCompleta_etapa1(file)
    #print data



    ## se mezclan las observaciones de las tablas
    ## semilla para mezclar los datos en forma aleatoria


    np.random.seed(rand)
    dataNew = selection.shuffle(data)
    dataNew = dataNew.reset_index(drop=True)
    #porc = 0.75
    print ("Porcentaje de datos de calculo: " + str(porc))

    ### division de los datos para entrenamiento y prueba
    nRow = len(dataNew.index)
    numTraining=int(round(nRow)*porc)
    print("Cantidad de elementos para el calculo de coeff: " + str(numTraining))
    numTest=int((nRow)-numTraining)
    print("Cantidad de elementos para prueba: " +str(numTest))


    dataTraining =  dataNew.ix[:numTraining, :]
    dataTraining = selection.shuffle(dataTraining)
    dataTraining = dataTraining.reset_index(drop=True)
    dataTest = dataNew.ix[numTraining + 1:, :]


    #OldRange = (np.max(dataTraining.T_aire)  - np.min(dataTraining.T_aire))
    #NewRange = (1)
    #dataTraining.T_aire = (((dataTraining.T_aire - np.min(dataTraining.T_aire)) * NewRange) / OldRange)

    #OldRange = (np.max(dataTraining.HR)  - np.min(dataTraining.HR))
    #NewRange = (1)
    #dataTraining.HR = (((dataTraining.HR - np.min(dataTraining.HR)) * NewRange) / OldRange)

    #OldRange = (np.max(dataTraining.PP)  - np.min(dataTraining.PP))
    #NewRange = (1)
    #dataTraining.PP = (((dataTraining.PP - np.min(dataTraining.PP)) * NewRange) / OldRange)

    #OldRange = (np.max(dataTraining.Sigma0)  - np.min(dataTraining.Sigma0))
    #NewRange = (1)
    #dataTraining.Sigma0 = (((dataTraining.Sigma0 - np.min(dataTraining.Sigma0)) * NewRange) / OldRange)

    print ("DataTraining")
    print(dataTraining)
    print (dataTraining.describe())


    #OldRange = (np.max(dataTest.T_aire)  - np.min(dataTest.T_aire))
    #NewRange = (1)
    #dataTest.T_aire = (((dataTest.T_aire - np.min(dataTest.T_aire)) * NewRange) / OldRange)

    #OldRange = (np.max(dataTest.HR)  - np.min(dataTest.HR))
    #NewRange = (1)
    #dataTest.HR = (((dataTest.HR - np.min(dataTest.HR)) * NewRange) / OldRange)

    #OldRange = (np.max(dataTest.PP)  - np.min(dataTest.PP))
    #NewRange = (1)
    #dataTest.PP = (((dataTest.PP - np.min(dataTest.PP)) * NewRange) / OldRange)

    #OldRange = (np.max(dataTest.Sigma0)  - np.min(dataTest.Sigma0))
    #NewRange = (1)
    #dataTest.Sigma0 = (((dataTest.Sigma0 - np.min(dataTest.Sigma0)) * NewRange) / OldRange)

    print ("DataTest")
    print (dataTest.describe())


    yTraining = dataTraining['SM_CONAE']
    del dataTraining['SM_CONAE']
    yTest = dataTest['SM_CONAE']
    del dataTest['SM_CONAE']

    # definicion del modelo
    model = Earth(max_degree=1,  penalty=1.25)
    ### max_terms=1000, max_degree=1, verbose=1 , penalty=2
    # Calibracion del modelo
    model.fit(dataTraining,yTraining)
    yCal = model.predict(dataTraining)

    #Print the model
    print(model.trace())
    print(model.summary())

    #return sympy expression
    print('Resulting sympy expression:')
    print(export.export_sympy(model))


    # Validacion del modelo
    yAprox = model.predict(dataTest)

     # Error de Calibracion
    print ("Calibracion MARS: ")
    rmse = statistics.RMSE(np.array(yTraining), yCal)
    print ("RMSE:" + str(rmse))
    RR = smf.ols('yTraining ~ 1+ yCal', dataTraining).fit().rsquared
    print ("R^2: "+str(RR))

    bias = statistics.bias(np.array(yTraining), yCal)
    print ("Bias:" + str(bias))


    #fig = plt.figure(2,facecolor="white")
    #fig2 = fig.add_subplot(111,aspect='equal')
    ##fig1, ax = plt.subplots()

    #fig2.set_title('Validacion')
    #fig2.scatter(yTest, yAprox, s=10, color='black',linewidth=3, label='MARS')
    #fig2.axis([5,45, 5,45])
    #plt.grid(True)
    #x = np.linspace(*fig2.get_xlim())
    #fig2.plot(x, x, linestyle="--", color='black')

    print ("Validacion MARS:")
    rmse = statistics.RMSE(np.array(yTest), yAprox)
    print ("RMSE:" + str(rmse))
    RR = smf.ols('yTest ~ 1+ yAprox', dataTest).fit().rsquared
    print ("R^2: "+str(RR))

    bias = statistics.bias(np.array(yTest), yAprox)
    print ("Bias:" + str(bias))


    return model, yCal, yAprox
    

####---------------------------------------------------------------------------

def normalizado(c):
    min = np.min(c)
    max = np.max(c)
    new = (c -min)/(max-min)
    OldRange = (max  - min)
    NewRange = (1 - 0.1)
    new = (((c - min) * NewRange) / OldRange) + 0.1
    return new
    

def mars2(fileCal, fileVal, rand):

    dataCal = lectura.lecturaCompletaMLP_etapa3(fileCal)
    dataVal = lectura.lecturaCompletaMLP_etapa3(fileVal)
    

#    print(dataVal)

    np.random.seed(rand)
    dataNew = selection.shuffle(dataCal)
    dataNew = dataNew.reset_index(drop=True)
    dataTraining =dataNew

    np.random.seed(rand)
    dataNew = selection.shuffle(dataVal)
    dataNew = dataNew.reset_index(drop=True)
    dataTest =dataNew
    
#    del dataTraining['T_s']
#    del dataTest['T_s']

#    dataTraining.T_s = normalizado(dataTraining.T_s)
#
#    dataTraining.Et = normalizado(dataTraining.Et)
#
#    dataTraining.PP = normalizado(dataTraining.PP)
#    
#    dataTraining.Sigma0 = normalizado(dataTraining.Sigma0)

#    print ("DataTraining")}
#    print("MARS #####################################################################")
#    print (dataTraining.describe())
#    print("#####################################################################")

#    dataTest.T_s = normalizado(dataTest.T_s)
#
#    dataTest.Et = normalizado(dataTest.Et)
#
#    dataTest.PP = normalizado(dataTest.PP)
#    
#    dataTest.Sigma0 = normalizado(dataTest.Sigma0)

    print ("DataTest")
    print (dataTest.describe())


    yTraining = dataTraining['SM_SMAP']
    del dataTraining['SM_SMAP']
    yTest = dataTest['SM_SMAP']
    del dataTest['SM_SMAP']

    ##### --------------- calibration-----------------------------------------------
    # definicion del modelo
    model = Earth(max_terms=10, max_degree=1,penalty=0.5, verbose=1)#max_degree=1, , penalty=0.000015  penalty=1.25)
    ### max_terms=1000, max_degree=1, verbose=1 , penalty=2
    # Calibracion del modelo
    model.fit(dataTraining,yTraining)
    yCal = model.predict(dataTraining)

    #Print the modelallow_linear
    print(model.trace())
    print(model.summary())

    #return sympy expression
    print('Resulting sympy expression:')
    print(export.export_sympy(model))


    # Validacion del modelo
    yAprox = model.predict(dataTest)

     # Error de Calibracion
    print ("Calibracion MARS: ")
    rmse = statistics.RMSE(np.array(yTraining), yCal)
    print ("RMSE:" + str(rmse))
    RR = smf.ols('yTraining ~ 1+ yCal', dataTraining).fit().rsquared
    print ("R^2: "+str(RR))

    bias = statistics.bias(np.array(yTraining), yCal)
    print ("Bias:" + str(bias))


    #fig = plt.figure(2,facecolor="white")
    #fig2 = fig.add_subplot(111,aspect='equal')
    ##fig1, ax = plt.subplots()

    #fig2.set_title('Validacion')
    #fig2.scatter(yTest, yAprox, s=10, color='black',linewidth=3, label='MARS')
    #fig2.axis([5,45, 5,45])
    #plt.grid(True)
    #x = np.linspace(*fig2.get_xlim())
    #fig2.plot(x, x, linestyle="--", color='black')

#### --------------- validation-----------------------------------------------    
    print ("Validacion MARS:")
    rmse = statistics.RMSE(np.array(yTest), yAprox)
    print ("RMSE:" + str(rmse))
    RR = smf.ols('yTest ~ 1+ yAprox', dataTest).fit().rsquared
    print ("R^2: "+str(RR))

    bias = statistics.bias(np.array(yTest), yAprox)
    print ("Bias:" + str(bias))


    return model, yCal, yAprox





if __name__ == '__main__':

#    file = "tabla_calibration_validation.csv"
#    #file = 'tabla_Completa.csv'
#
#    # se aplica el modelo MARS
#    porc = 0.75
#    rand = 0
#    MARSmodel, yCalMARS, yAproxMARS = mars(porc, file, rand)
#    
    
    
    fileCal = 'tabla_completa_Calibracion_dataSet_1km.csv'
    fileVal = 'tabla_completa_Validacion_dataSet_1km.csv'
    rand = 0
    MARSmodel, yCalMARS, yAproxMARS = mars2(fileCal, fileVal, rand)
    

    # Calibracion
    #fig = plt.figure(1,facecolor="white")
    #fig1 = fig.add_subplot(111,aspect='equal')
    #fig1.set_title('calibracion')
    #fig1.scatter(yTraining, yCal, s=10, color='black',linewidth=3, label='MARS')
    #fig1.axis([5,45, 5,45])
    #plt.grid(True)

    #x = np.linspace(*fig1.get_xlim())
    #fig1.plot(x, x, linestyle="--", color='black')


    #plt.show()