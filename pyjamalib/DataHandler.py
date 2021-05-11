import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv

class DataHandler:
    def import_raw_data (pathData, pathTime):
        """Reads the raw data and removes extra characters.
        Input data from the ESP's.

        Parameters
        ----------
        pathData : .txt file with all data in strings
        pathTime : .txt file with all timestamp in strings

        Returns
        -------
        data: data in list
            List with all data without extra characters.
        time_stamp: data in list
            List with all time stamp data without extra characters.

        See Also
        --------
        Developed by T.F Almeida in 25/03/2021

        For more information see:
        https://github.com/tuliofalmeida/pyjama    
        """     
        data = open(pathData,"r").read()
        tempo = open(pathTime,"r")

        var = data.split(" ")
        newString = ""
        for i in var:
            newString = newString+i

        var = newString.split(",b'")
        newString = ""
        for i in var:
            newString = newString+i

        var = newString.split("b'")
        newString = ""
        for i in var:
            newString = newString+i

        var = newString.split("'")
        newString = ""
        for i in var:
            newString = newString+i

        var = newString.split('"')
        newString = ""
        for i in var:
            newString = newString+i

        data = newString.split('[')
        time_stamp = tempo.read().split('[')

        return data,time_stamp

    def split_raw_data(esp_data):
        """Split the raw data for each ESP32.
        Input data processed by 'import_raw_data'.

        Parameters
        ----------
        esp_data : variable in str of all clean data

        Returns
        -------
        list_data : data in list
            List with all separated by each ESP32 in different lines.

        See Also
        --------
        Developed by T.F Almeida in 25/03/2021

        For more information see:
        https://github.com/tuliofalmeida/pyjama    
        """  

        temp = (esp_data.split(';'))
        list_data = []
        for i in temp:
            x = i.split("'")[-1]
            toF = x.split(',')
            listaEspTemp = []
            for i in toF:
                if (i != ']' and i != ''):
                    listaEspTemp.append(float(i))

            list_data.append(listaEspTemp)
        return list_data

    def get_imu_data(dataSplited,IsGyrInRad = True):
        """Split the raw data of the specific ESP's in arrays.

        Parameters
        ----------
        dataSplited : list
                    data of a specific ESP 
        IsGyrInRad: bool optional
                    Determine if the esp is in degrees or in radians

        Returns
        -------
        time: ndarray
            Array with all timestamp data according with the acquisition frequency
            in rad/s
        acc: ndarray
            Array with all accelerometer data (XYZ) according with the acquisition frequency.
        gyr: ndarray
            Array with all gyroscope data (XYZ) according with the acquisition frequency.
            in rad/s.
        mag: ndarray
            Array with all magnetometer data (XYZ) according with the acquisition frequency
            in mG.

        See Also
        --------
        Developed by T.F Almeida in 25/03/2021

        For more information see:
        https://github.com/tuliofalmeida/pyjama    
        """  
        time = []
        acc = []
        gyr = []
        mag = []
        for i in range(len(dataSplited)-1):
            time.append(dataSplited[i][0])
        for i in range(len(dataSplited)-1):
            acc.append((dataSplited[i][1],dataSplited[i][2],dataSplited[i][3]))
        for i in range(len(dataSplited)-1):
            gyr.append((dataSplited[i][4],dataSplited[i][5],dataSplited[i][6]))
        for i in range(len(dataSplited)-1):
            mag.append((dataSplited[i][7],dataSplited[i][8],dataSplited[i][9]))
        if IsGyrInRad == True:
            return np.asarray(time),np.asarray(acc),np.asarray(gyr)*np.pi/180,np.asarray(mag)
        else:
            return np.asarray(time),np.asarray(acc),np.asarray(gyr),np.asarray(mag)

    def calibration_imu(acc,gyr,mag,mag_calib,tempoBasal = 5, freq = 75):
        """Calibrate the data according to the location's magnetic field. 
        This procedure is 'optional' and magnetometer data is expected in 
        which the sensor is moved in all directions in the same environment 
        in which the data was collected. Can be used with functional calibration data.

        Parameters
        ----------
        acc: ndarray
            Accelerometer data (XYZ).
        gyr: ndarray
            Gyroscope data (XYZ).
        mag: ndarray
            Magnetometer data (XYZ).
        mag_calib: ndarray
            Magnetometer data (XYZ) from the calibration procedure.
        basal_time: int
            Determine the time in seconds that the sensor that will
            be calibrated is completely static.
        freq: int
            Frequency of data acquisition.

        Returns
        -------
        acc_cab: ndarray
            Array of calibrated accelerometer data (XYZ) transposed.
        gyr_cab: ndarray
            Array of calibrated gyroscopic data (XYZ) transposed.
        mag_cab: ndarray
            Array of calibrated magnetometer data (XYZ) transposed.

        See Also
        --------
        Developed by T.F Almeida in 25/03/2021

        For more information see:
        https://github.com/tuliofalmeida/pyjama    
        """  
        tempo = tempoBasal * freq
        #ACC  
        acc_mean_x = np.mean(acc[0:(tempo)][:,0])
        acc_mean_y = np.mean(acc[0:(tempo)][:,1])-1 #MUDAR A PORRA DO Y PARA O Z (+1)
        acc_mean_z = np.mean(acc[0:(tempo)][:,2])   #-1

        for i in range(len(acc)):
            acc[i][0] = acc[i][0] - acc_mean_x
            acc[i][1] = acc[i][1] - acc_mean_y
            acc[i][2] = acc[i][2] - acc_mean_z
        #GYR
        gyr_mean_x = np.mean(gyr[0:(tempo)][:,0])
        gyr_mean_y = np.mean(gyr[0:(tempo)][:,1])
        gyr_mean_z = np.mean(gyr[0:(tempo)][:,2])

        for i in range(len(gyr)):
            gyr[i][0] = gyr[i][0] - gyr_mean_x
            gyr[i][1] = gyr[i][1] - gyr_mean_y
            gyr[i][2] = gyr[i][2] - gyr_mean_z

        #MAG
        min_x = min(mag_calib[:,0])
        max_x = max(mag_calib[:,0])
        min_y = min(mag_calib[:,1])
        max_y = max(mag_calib[:,1])
        min_z = min(mag_calib[:,2])
        max_z = max(mag_calib[:,2])
        
        mag_calibration = [ (max_x + min_x) / 2, (max_y + min_y) / 2, (max_z + min_z) / 2]

        for i in range(len(mag)):
            mag[i][0] = mag[i][0] - mag_calibration[0]
            mag[i][1] = mag[i][1] - mag_calibration[1]
            mag[i][2] = mag[i][2] - mag_calibration[2]

        return acc,gyr,mag

    def saveCSV(time,acc,gyr,mag, filename = 'Data.csv', path = '/content/drive/MyDrive/Dissertação/Dados/CSV/'):
        """Saves the accelerometer, gyroscope, magnetometer 
        and time vector data to a csv file.

        Parameters
        ----------
        acc: ndarray
            Accelerometer data (XYZ).
        gyr: ndarray
            Gyroscope data (XYZ).
        mag: ndarray
            Magnetometer data (XYZ).
        filename: str optional
            Determines the name of the file to be saved. 
            The default name is 'Data.csv'
        path: str
            Determines the local path of the file to 
            be saved. The default path is '/content/'

        Returns
        -------
        csv file with all data
        See Also
        --------
        Developed by T.F Almeida in 25/03/2021

        For more information see:
        https://github.com/tuliofalmeida/pyjama    
        """  
        with open(path + filename,'w') as file:
            x = csv.writer(file)
            head = 'time','acc_x','acc_y','acc_z','gyr_x','gyr_y','gyr_z','mag_x','mag_y','mag_z'
            x.writerow(head)     
            for i in range(len(time)):
                line = time[i], acc[i][0],acc[i][1],acc[i][2], gyr[i][0],gyr[i][1],gyr[i][2], mag[i][0],mag[i][1],mag[i][2]
                x.writerow(line)

    def csvToFloat(data):
        """Saves the accelerometer, gyroscope, magnetometer 
        and time vector data to a csv file.

        Parameters
        ----------
        data: dataframe column
            The dataframe column that will be 
            converted to ndarray.

        Returns
        -------
        The df column in numpy array
        
        See Also
        --------
        Developed by T.F Almeida in 25/03/2021

        For more information see:
        https://github.com/tuliofalmeida/pyjama    
        """  
        x = pd.to_numeric(data[0:])
        return x.to_numpy()

    def csvFloatMerge(dfX,dfY,dfZ):
        """Merge the X,Y,Z datafarme into a single ndarray.

        Parameters
        ----------
        dfX: dataframe column
            The dataframe column that will be 
            converted to ndarray.
        dfY: dataframe column
            The dataframe column that will be 
            converted to ndarray.
        dfZ: dataframe column
            The dataframe column that will be 
            converted to ndarray.

        Returns
        -------
        A single matrix with 3 columns transposed

        See Also
        --------
        Developed by T.F Almeida in 25/03/2021

        For more information see:
        https://github.com/tuliofalmeida/pyjama    
        """  
        return np.asarray([DataHandler.csvToFloat(dfX),DataHandler.csvToFloat(dfY),DataHandler.csvToFloat(dfZ)]).T
    
    def vicon2dict(dataPath, freq):
        """Convert Vicon's orientation data into a Dictionary.
        Device probably used Vicon Blade. File type = '.txt'.

        Parameters
        ----------
        dataPath:  string
            Data path.

        freq: int
            Data aquisition frequency to calculate
            the vector time for plots.
        
        Return
        ------
        dataDict: dictionary
            Dictionary separating the data of 
            each part of the body and orientation 
            in quaternion. The autocomplete works.

        See Also
        --------
        Developed by T.F Almeida in 23/04/2021

        For more information see:
        Dataset used to develop this function was Total Capture.
        https://cvssp.org/data/totalcapture/

        Real-time Full-Body Motion Capture from Video and IMUs
        https://ieeexplore.ieee.org/document/8374599

        https://github.com/tuliofalmeida/pyjama  
        """
        df = pd.read_csv(dataPath, sep='\t')
        dataDict = {}
        columns = df.columns
        size = int(len(df[columns[0]]))
        
        for ç in range(len(columns)-1):
            k = 0
            quaternion = np.zeros((size,4))
            for i in df[columns[ç]]:
                t = i.split(' ')
                temp2 = []
                for j in t:
                    temp2.append(float(j))
                quaternion[k,:] = temp2
                k +=1
            dataDict[columns[ç]] = quaternion
        
        dataDict['Time'] = np.arange(0,size/freq,1/freq)

        return dataDict
        
    def xSens2dict(dataPath,freq):
        """Convert Xsens MTw wireless IMU data 
        into a dictionary.Possibly it works for 
        other sensors of the same company, since
        it returns quaternion ([1,0,0,0]), 
        accelerometer (XYZ), gyroscope (XYZ) and 
        magnetometer (XYZ).

        Parameters
        ----------
        dataPath:  string
            Data path.
        
        freq: int
            Data aquisition frequency to calculate
            the vector time for plots.
        
        Return
        ------
        dataDict: dictionary
            Dictionary separating the data of 
            each part of the body and orientation 
            (Quaternion, Acelerometer, Gyroscope 
            and Magnetometer). The autocomplete 
            works. File type = '.sensors'.

        See Also
        --------
        Developed by T.F Almeida in 23/04/2021

        For more information see:
        Dataset used to develop this function was Total Capture.
        https://cvssp.org/data/totalcapture/

        Real-time Full-Body Motion Capture from Video and IMUs
        https://ieeexplore.ieee.org/document/8374599

        https://github.com/tuliofalmeida/pyjama  
        """
        df = pd.read_csv(dataPath)
        nameList = []
        output = []
        outputNames = []
        for i in range(len(df)):
            strTemp = ((df[df.columns[0]][i]).split('\t'))
            if len(strTemp) == 14:
                listTemp = []
                for j in range(len(strTemp)-1):
                    listTemp.append(float(strTemp[j+1]))
                outputNames.append(strTemp[0])
                output.append(listTemp)
        name = set(outputNames)
        nameList = list(name)
        dataDict = {}
        size = int(len(outputNames)/len(name))
        
        for ç in name:
            k = 0
            quaternion = np.zeros((size,4))
            accel = np.zeros((size,3))
            gyro  = np.zeros((size,3))
            mag   = np.zeros((size,3))
            for i in range(len(outputNames)):
                if outputNames[i] == ç:
                    quaternion[k,:] = output[i][0:4]
                    accel[k,:] = output[i][4:7]
                    gyro[k,:] = output[i][7:10]
                    mag[k,:] = output[i][10:13]
                    k += 1
            dataDict[ç] = {'Quaternion':quaternion,
                        'Accelerometer':accel,
                        'Gyroscope':gyro,
                        'Magnetometer':mag}
            
        dataDict['Time'] = np.arange(0,size/freq,1/freq)
        
        return dataDict