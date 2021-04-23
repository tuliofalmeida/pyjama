import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import csv
import os
import time
import collections
from time import sleep, time
from math import pi, sin, cos, asin, acos, atan2, sqrt, atan
from scipy import signal,stats

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
        https://github.com/tuliofalmeida    
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
        https://github.com/tuliofalmeida    
        """  

        temp = (esp_data.split(';'))
        list_data = []
        for i in temp:
            x = i.split("'")[-1]
            #print(x)
            toF = x.split(',')
            #print(toF)
            listaEspTemp = []
            for i in toF:
                if (i != ']' and i != ''): #and i != ',' and i != ' ' and i != ''):
                    #print('x: ',x, 'i: ',i)
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
        https://github.com/tuliofalmeida    
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
            return np.asarray(time),np.asarray(acc),np.asarray(gyr)*pi/180,np.asarray(mag)
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
        https://github.com/tuliofalmeida    
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

    def toDataframe(data,data_calib,filter=.04,freq=75,dt=1/75,alpha=.01,beta=.05,beta_mag=.9,beta_mag2=.01,conj=True):
        """This function receives the data from an ESP32 
        performs the data calibration and applies all filters. 
        After all manipulations the results are saved to a
        pandas datafarme.

        Parameters
        ----------
        data: ndarray
            Esp data returned by 'split_raw_data'.
        data_calib: ndarray
            Esp calibration data returned by 
            'split_raw_data'.
        filter: float
            Low-pass filter intensity.
        freq: int
            Frequency of data acquisition.
        dt: float
            Sample time.
        alpha: float
            Determines the influence of the accelerometer 
            on the formation of the angle by the complementary filter
        beta: float
            Factor to improve the effectiveness of 
            integrating the accelerometer with gyroscope.
            Must be determined between 0 and 1.
        beta_mag: float
           There is a tradeoff in the beta parameter 
           between accuracy and response speed. Must
           be determined according with the Gyroscope
           error. Used for the first quaternion 
           orientation. For MadgwickAHRS filte.
        beta_mag2: float
           There is a tradeoff in the beta parameter 
           between accuracy and response speed. Must
           be determined according with the Gyroscope
           error. Used for the others orientations.
           For MadgwickAHRS filte.
        conj: bool
            Determine if the quaternion resulted will be
            conjugated or not.

        Returns
        -------
        df: pandas dataframe
            A pandas dataframe with the euler angles computed
            using quaternions formulations.

        See Also
        --------
        Developed by T.F Almeida in 25/03/2021

        For more information see:
        https://github.com/tuliofalmeida    
        """         
        time,acc,gyr,mag= DataHandler.get_imu_data(data)
        time_calib,acc_calib,gyr_calib,mag_calib = DataHandler.get_imu_data(data_calib)
        time = np.arange(0, len(time)/freq, dt)

        acc, gyr, mag = DataHandler.calibration_imu(acc,gyr,mag,mag_calib)
        accf = IMUDataProcessing.low_pass_filter(acc,filter)
        gyrf = IMUDataProcessing.low_pass_filter(gyr,filter)
        magf = IMUDataProcessing.low_pass_filter(mag,filter)

        df = pd.DataFrame({'Time':time[:]                                                               ,
                        'Acc_X':acc[:,0]         ,'Acc_Y': acc[:,1]         ,'Acc_Z': acc[:,2]       ,
                        'Gyr_X':gyr[:,0]         ,'Gyr_Y': gyr[:,1]         ,'Gyr_Z': gyr[:,2]       ,
                        'Mag_X':mag[:,0]         ,'Mag_Y': mag[:,1]         ,'Mag_Z': mag[:,2]       ,
                        'Acc_X_Filt':accf[:,0]   ,'Acc_Y_Filt':accf[:,1]    ,'Acc_Z_Filt': accf[:,2] ,
                        'Gyr_X_Filt':gyrf[:,0]   ,'Gyr_Y_Filt':gyrf[:,1]    ,'Gyr_Z_Filt': gyrf[:,2] ,
                        'Mag_X_Filt':magf[:,0]   ,'Mag_Y_Filt':magf[:,1]    ,'Mag_Z_Filt': magf[:,2] ,
                        'Roll':acc[:,0]          ,'Pitch': acc[:,0]         ,'Yaw': acc[:,0]         ,
                        'CF_Roll':acc[:,0]       ,'CF_Pitch':acc[:,0]       ,'CF_Yaw':acc[:,0]       ,
                        'CF_GD_Roll':acc[:,0]    ,'CF_GD_Pitch':acc[:,0]    ,'CF_GD_Yaw':acc[:,0]    ,
                        'CF_GN_Roll':acc[:,0]    ,'CF_GN_Pitch':acc[:,0]    ,'CF_GN_Yaw':acc[:,0]    ,       
                        'Kalman_GD_Roll':acc[:,0],'Kalman_GD_Pitch':acc[:,0],'Kalman_GD_Yaw':acc[:,0],
                        'Kalman_GN_Roll':acc[:,0],'Kalman_GN_Pitch':acc[:,0],'Kalman_GN_Yaw':acc[:,0],
                        'Madgwick_Roll':acc[:,0] ,'Madgwick_Pitch':acc[:,0] ,'Madgwick_Yaw':acc[:,0]})
        
        acc_df   = DataHandler.csvFloatMerge(df['Acc_X'],df['Acc_Y'],df['Acc_Z'])
        gyr_df   = DataHandler.csvFloatMerge(df['Gyr_X'],df['Gyr_Y'],df['Gyr_Z'])
        mag_df   = DataHandler.csvFloatMerge(df['Mag_X'],df['Mag_Y'],df['Mag_Z'])
        acc_df_f = DataHandler.csvFloatMerge(df['Acc_X_Filt'],df['Acc_Y_Filt'],df['Acc_Z_Filt'])
        gyr_df_f = DataHandler.csvFloatMerge(df['Gyr_X_Filt'],df['Gyr_Y_Filt'],df['Gyr_Z_Filt'])
        mag_df_f = DataHandler.csvFloatMerge(df['Mag_X_Filt'],df['Mag_Y_Filt'],df['Mag_Z_Filt'])

        Roll, Pitch, Yaw = IMUDataProcessing.get_euler(q=[1,0,0,0],Acc=acc_df_f,Mag=mag_df_f,conj=conj)
        CF    = IMUDataProcessing.complementaryFilter(Roll,Pitch,Yaw,gyr_df_f[:,0],gyr_df_f[:,1],gyr_df_f[:,2],alpha=.05,dt=dt)
        CF_GD = IMUDataProcessing.ComplementaryFilterGD(acc_df_f,gyr_df_f,mag_df_f,dt=dt,alpha=alpha,beta=beta,conj=conj)
        CF_GN = IMUDataProcessing.ComplementaryFilterGN(acc_df_f,gyr_df_f,mag_df_f,dt=dt,alpha=alpha,beta=beta,conj=conj)
        Kalman_GD = IMUDataProcessing.KalmanGD(acc_df_f,gyr_df_f,mag_df_f,dt=dt,beta=beta,conj=conj)
        Kalman_GN = IMUDataProcessing.KalmanGN(acc_df_f,gyr_df_f,mag_df_f,dt=dt,beta=beta,conj=conj)
        Madgwick  = IMUDataProcessing.MadgwickAHRS(acc_df,gyr_df,mag_df,freq=freq,beta1=beta_mag,beta2=beta_mag2)

        df['Roll'],df['Pitch'],df['Yaw'] = Roll, Pitch, Yaw
        df['CF_Roll'],df['CF_Pitch'],df['CF_Yaw'] = CF[:,0],CF[:,1],CF[:,2]
        df['CF_GD_Roll'],df['CF_GD_Pitch'],df['CF_GD_Yaw'] = CF_GD[:,0],CF_GD[:,1],CF_GD[:,2]
        df['CF_GN_Roll'],df['CF_GN_Pitch'],df['CF_GN_Yaw'] = CF_GN[:,0],CF_GN[:,1],CF_GN[:,2]
        df['Kalman_GD_Roll'],df['Kalman_GD_Pitch'],df['Kalman_GD_Yaw'] = Kalman_GD[:,0],Kalman_GD[:,1],Kalman_GD[:,2]
        df['Kalman_GN_Roll'],df['Kalman_GN_Pitch'],df['Kalman_GN_Yaw'] = Kalman_GN[:,0],Kalman_GN[:,1],Kalman_GN[:,2]
        df['Madgwick_Roll'],df['Madgwick_Pitch'],df['Madgwick_Yaw'] = Madgwick[:,0],Madgwick[:,1],Madgwick[:,2]

        return df

    def joint_measures(df_first_joint,df_second_joint,patternRoll=False,patternPitch=False,patternYaw=False,init=0,end=None,freq=75,threshold=0,cicle=2,bias=0,poly_degree=9,IC=1.96):
        """This function is used to calculate the angle 
        of a given joint. If the movement performed has 
        a clear pattern, it is possible to extract it by 
        determining at what angle of euler it happens 
        (Flexion and extension - Roll, Adduction and 
        Abduction - Pitch and Yaw Rotations), being possible 
        to extract it in just one orientation. Example: 
        in gait movement, the knee has small variations in 
        rotations and adduction / abduction, so the pattern
        may be present only in flexion / extension, the axis 
        of primary movement of the joint. This function returns 
        two data frames, one with an angle only in each filter 
        and the other with statistical metrics.

        Parameters
        ----------
        df_first_joint: pandas dataframe
            Dataframe with ESP32 data positioned 
            above the target joint returned by the 
            'toDataFrame' function.
        df_second_joint: pandas dataframe
            Dataframe with ESP32 data positioned 
            below the target joint returned by the 
            'toDataFrame' function.
        patternRoll: bool
            If true it will calculate the roll pattern.
            If there is no clear pattern and well 
            determined by the threshold, the function 
            will give an error.            
        patternPitch: bool
            If true it will calculate the pitch pattern.
            If there is no clear pattern and well 
            determined by the threshold, the function 
            will give an error.            
        patternYaw: bool
            If true it will calculate the yaw pattern.
            If there is no clear pattern and well 
            determined by the threshold, the function 
            will give an error.
        init: int optional
            Determines where the data will start 
            to be read from. Used to cut an initial 
            piece of data or read only a portion.
        end: int optional
            Determines where the data reading will 
            end. Used to cut a final piece of data 
            or read only a portion..
        freq: float
            Frequency of data acquisition.
        treshold: float
            Point at which the data moves between 
            movements. Example: flexion and extension.
        cicle: int
            Number of points to be considered a pattern.
        bias: int optional
            Value to compensate the cicle adjust.
        poly_degree: int
            Degree of the polynomial to fit the data curve.
        IC: float
            Reference value for calculating the 95% 
            confidence interval.

        Returns
        -------
        df: pandas dataframe
            A pandas dataframe with the joint angles in
            each filter.
        df_metrics: pandas dataframe
            A panda dataframe containing the statistical 
            metrics of the analyzed movement.
        
        See Also
        --------
        Developed by T.F Almeida in 25/03/2021

        For more information see:
        https://github.com/tuliofalmeida    
        """         
        if end == None:
            end = len(df_first_joint['Time'])
        else:
            end = end
        df = pd.DataFrame({'Time':df_first_joint['Time'][init:end]                                                                                                                            ,
                        'Flex/Ext':df_first_joint['Time'][init:end]          ,'Adu/Abd':df_first_joint['Time'][init:end]          ,'Int/Ext_Rot':df_first_joint['Time'][init:end]          ,
                        'Flex/Ext_CF':df_first_joint['Time'][init:end]       ,'Adu/Abd_CF':df_first_joint['Time'][init:end]       ,'Int/Ext_Rot_CF':df_first_joint['Time'][init:end]       ,
                        'Flex/Ext_CF_GD':df_first_joint['Time'][init:end]    ,'Adu/Abd_CF_GD':df_first_joint['Time'][init:end]    ,'Int/Ext_Rot_CF_GD':df_first_joint['Time'][init:end]    ,
                        'Flex/Ext_CF_GN':df_first_joint['Time'][init:end]    ,'Adu/Abd_CF_GN':df_first_joint['Time'][init:end]    ,'Int/Ext_Rot_CF_GN':df_first_joint['Time'][init:end]    ,
                        'Flex/Ext_Kalman_GD':df_first_joint['Time'][init:end],'Adu/Abd_Kalman_GD':df_first_joint['Time'][init:end],'Int/Ext_Rot_Kalman_GD':df_first_joint['Time'][init:end],
                        'Flex/Ext_Kalman_GN':df_first_joint['Time'][init:end],'Adu/Abd_Kalman_GN':df_first_joint['Time'][init:end],'Int/Ext_Rot_Kalman_GN':df_first_joint['Time'][init:end],
                        'Flex/Ext_Madgwick':df_first_joint['Time'][init:end] ,'Adu/Abd_Madgwick':df_first_joint['Time'][init:end] ,'Int/Ext_Rot_Madgwick':df_first_joint['Time'][init:end]       
                        })

        #Calcular o delay
        df['Flex/Ext'] = 180-(df_first_joint['Roll']+df_second_joint['Roll'])
        df['Adu/Abd'] = 180-(df_first_joint['Pitch']+df_second_joint['Pitch'])
        df['Int/Ext_Rot'] = 180-(df_first_joint['Yaw']+df_second_joint['Yaw'])

        df['Flex/Ext_CF'] = 180-(df_first_joint['CF_Roll']+df_second_joint['CF_Roll'])
        df['Adu/Abd_CF'] = 180-(df_first_joint['CF_Pitch']+df_second_joint['CF_Pitch'])
        df['Int/Ext_Rot_CF'] = 180-(df_first_joint['CF_Yaw']+df_second_joint['CF_Yaw'])

        df['Flex/Ext_CF_GD'] = 180-(df_first_joint['CF_GD_Roll']+df_second_joint['CF_GD_Roll'])
        df['Adu/Abd_CF_GD'] = 180-(df_first_joint['CF_GD_Pitch']+df_second_joint['CF_GD_Pitch'])
        df['Int/Ext_Rot_CF_GD'] = 180-(df_first_joint['CF_GD_Yaw']+df_second_joint['CF_GD_Yaw'])

        df['Flex/Ext_CF_GN'] = 180-(df_first_joint['CF_GN_Roll']+df_second_joint['CF_GN_Roll'])
        df['Adu/Abd_CF_GN'] = 180-(df_first_joint['CF_GN_Pitch']+df_second_joint['CF_GN_Pitch'])
        df['Int/Ext_Rot_CF_GN'] = 180-(df_first_joint['CF_GN_Yaw']+df_second_joint['CF_GN_Yaw'])

        df['Flex/Ext_Kalman_GD'] = 180-(df_first_joint['Kalman_GD_Roll']+df_second_joint['Kalman_GD_Roll'])
        df['Adu/Abd_Kalman_GD'] = 180-(df_first_joint['Kalman_GD_Pitch']+df_second_joint['Kalman_GD_Pitch'])
        df['Int/Ext_Rot_Kalman_GD'] = 180-(df_first_joint['Kalman_GD_Yaw']+df_second_joint['Kalman_GD_Yaw'])
    
        df['Flex/Ext_Kalman_GN'] = 180-(df_first_joint['Kalman_GN_Roll']+df_second_joint['Kalman_GN_Roll'])
        df['Adu/Abd_Kalman_GN'] = 180-(df_first_joint['Kalman_GN_Pitch']+df_second_joint['Kalman_GN_Pitch'])
        df['Int/Ext_Rot_Kalman_GN'] = 180-(df_first_joint['Kalman_GN_Yaw']+df_second_joint['Kalman_GN_Yaw'])

        df['Flex/Ext_Madgwick'] = 180-(df_first_joint['Madgwick_Roll']+df_second_joint['Madgwick_Roll'])
        df['Adu/Abd_Madgwick'] = 180-(df_first_joint['Madgwick_Pitch']+df_second_joint['Madgwick_Pitch'])
        df['Int/Ext_Rot_Madgwick'] = 180-(df_first_joint['Madgwick_Yaw']+df_second_joint['Madgwick_Yaw'])

        index = []
        Rom = []
        Mean = []
        Std = []
        CI = []
        Var = []
        Min = []
        Max = []
        MinEst = []
        MaxEst = []

        if patternRoll:
            rR,rawRrom = pattern_extraction(df['Flex/Ext'],df['Time'],threshold=np.mean(df['Flex/Ext']), bias=bias, cicle=cicle);
            print(rR)
            rawRoll = patternIC(rR[:,0],rR[:,1],poly_degree=poly_degree,IC=IC, df=False);
            rm_R = rom_mean(rawRrom)
            CFR,CFRrom=pattern_extraction(df['Flex/Ext_CF'],df['Time'],threshold=np.mean(df['Flex/Ext_CF']), bias=bias, cicle=cicle);
            cfRoll=patternIC(CFR[:,0],CFR[:,1],poly_degree=poly_degree,IC=IC, df=False);
            cf_R = rom_mean(CFRrom)
            CFGDR,CFGDRrom=pattern_extraction(df['Flex/Ext_CF_GD'],df['Time'],threshold=np.mean(df['Flex/Ext_CF_GD']), bias=bias, cicle=cicle);
            cfgdRoll=patternIC(CFGDR[:,0],CFGDR[:,1],poly_degree=poly_degree,IC=IC, df=False);
            cfgd_R= rom_mean(CFGDRrom)
            CFGNR,CFGNRrom=pattern_extraction(df['Flex/Ext_CF_GN'],df['Time'],threshold=np.mean(df['Flex/Ext_CF_GN']), bias=bias, cicle=cicle);
            cfgnRoll=patternIC(CFGNR[:,0],CFGNR[:,1],poly_degree=poly_degree,IC=IC, df=False);
            cfgn_R = rom_mean(CFGNRrom)
            KGDR,KGDRrom=pattern_extraction(df['Flex/Ext_Kalman_GD'],df['Time'],threshold=np.mean(df['Flex/Ext_Kalman_GD']), bias=bias, cicle=cicle);
            kgdRoll=patternIC(KGDR[:,0],KGDR[:,1],poly_degree=poly_degree,IC=IC, df=False);
            kgd_R = rom_mean(KGDRrom)
            KGNR,KGNRRrom=pattern_extraction(df['Flex/Ext_Kalman_GN'],df['Time'],threshold=np.mean(df['Flex/Ext_Kalman_GN']), bias=bias, cicle=cicle);
            kgnRoll=patternIC(KGNR[:,0],KGNR[:,1],poly_degree=poly_degree,IC=IC, df=False);
            kgn_R = rom_mean(KGNRRrom)
            MADR,MADRrom=pattern_extraction(df['Flex/Ext_Madgwick'],df['Time'],threshold=np.mean(df['Flex/Ext_Madgwick']), bias=bias, cicle=cicle);
            madRoll=patternIC(MADR[:,0],MADR[:,1],poly_degree=poly_degree,IC=IC, df=False);
            mad_R = rom_mean(MADRrom)
            romRawRoll = max(df['Flex/Ext'])-min(df['Flex/Ext'])
            romRollCF = max(df['Flex/Ext_CF'])-min(df['Flex/Ext_CF'])
            romRollCfGd = max(df['Flex/Ext_CF_GD'])-min(df['Flex/Ext_CF_GD'])
            romRollCfGn = max(df['Flex/Ext_CF_GN'])-min(df['Flex/Ext_CF_GN'])
            romRollKalmanGd = max(df['Flex/Ext_Kalman_GD'])-min(df['Flex/Ext_Kalman_GD'])
            romRollKalmanGn = max(df['Flex/Ext_Kalman_GN'])-min(df['Flex/Ext_Kalman_GN'])
            romRollMad = max(df['Flex/Ext_Madgwick'])-min(df['Flex/Ext_Madgwick'])
            minRawRoll = min(df['Flex/Ext'])
            minRollCF = min(df['Flex/Ext_CF'])
            minRollCfGd = min(df['Flex/Ext_CF_GD'])
            minRollCfGn = min(df['Flex/Ext_CF_GN'])
            minRollKalmanGd = min(df['Flex/Ext_Kalman_GD'])
            minRollKalmanGn = min(df['Flex/Ext_Kalman_GN'])
            minRollMad = max(df['Flex/Ext_Madgwick'])
            maxRawRoll = max(df['Flex/Ext'])
            maxRollCF = max(df['Flex/Ext_CF'])
            maxRollCfGd = max(df['Flex/Ext_CF_GD'])
            maxRollCfGn = max(df['Flex/Ext_CF_GN'])
            maxRollKalmanGd = max(df['Flex/Ext_Kalman_GD'])
            maxRollKalmanGn = max(df['Flex/Ext_Kalman_GN'])
            maxRollMad = max(df['Flex/Ext_Madgwick'])

            indexRoll = ['Flex/Ext','Flex/Ext_CF','Flex/Ext_CF_GD','Flex/Ext_CF_GN','Flex/Ext_Kalman_GD','Flex/Ext_Kalman_GN','Flex/Ext_Madgwick']
            meanRoll = [rm_R,cf_R,cfgd_R,cfgn_R,kgd_R,kgn_R,mad_R]
            stdRoll = [rawRoll[1],cfRoll[1],cfgdRoll[1],cfgnRoll[1],kgdRoll[1],kgnRoll[1],madRoll[1]]
            ciRoll = [rawRoll[0],cfRoll[0],cfgdRoll[0],cfgnRoll[0],kgdRoll[0],kgnRoll[0],madRoll[0]]
            varRoll = [rawRoll[7],cfRoll[7],cfgdRoll[7],cfgnRoll[7],kgdRoll[7],kgnRoll[7],madRoll[7]]
            minEstRoll = [rawRoll[4],cfRoll[4],cfgdRoll[4],cfgnRoll[4],kgdRoll[4],kgnRoll[4],madRoll[4]]
            maxEstRoll = [rawRoll[5],cfRoll[5],cfgdRoll[5],cfgnRoll[5],kgdRoll[5],kgnRoll[5],madRoll[5]]
            romRoll = [romRawRoll,romRollCF,romRollCfGd,romRollCfGn,romRollKalmanGd,romRollKalmanGn,romRollMad]
            minRoll = [minRawRoll,minRollCF,minRollCfGd,minRollCfGn,minRollKalmanGd,minRollKalmanGn,minRollMad]
            maxRoll = [maxRawRoll,maxRollCF,maxRollCfGd,maxRollCfGn,maxRollKalmanGd,maxRollKalmanGn,maxRollMad]
            index = np.concatenate((index,indexRoll))
            Rom = np.concatenate((Rom,romRoll))
            Mean = np.concatenate((Mean,meanRoll))
            Std = np.concatenate((Std,stdRoll))
            CI = np.concatenate((CI,ciRoll))
            Var = np.concatenate((Var,varRoll))
            Min = np.concatenate((Min,minRoll))
            Max = np.concatenate((Max,maxRoll))
            MinEst = np.concatenate((MinEst,minEstRoll))
            MaxEst = np.concatenate((MaxEst,maxEstRoll))

        if patternPitch:    
            rP,rawProm=pattern_extraction(df['Adu/Abd'],df['Time'],threshold=300, bias=bias, cicle=cicle);
            rawPitch=patternIC(rP[:,0],rP[:,1],poly_degree=poly_degree,IC=IC,df=False);
            rm_P = rom_mean(rawProm) 
            CFP,CFProm=pattern_extraction(df['Adu/Abd_CF'],df['Time'],threshold=260, bias=bias, cicle=cicle);
            cfPitch=patternIC(CFP[:,0],CFP[:,1],poly_degree=poly_degree,IC=IC, df=False);
            cf_P = rom_mean(CFProm)
            CFGDP,CFGDProm=pattern_extraction(df['Adu/Abd_CF_GD'],df['Time'],threshold=175, bias=bias, cicle=cicle);
            cfgdPitch=patternIC(CFGDP[:,0],CFGDP[:,1],poly_degree=poly_degree,IC=IC, df=False);
            cfgd_P = rom_mean(CFGDProm)
            CFGNP,CFGNProm=pattern_extraction(df['Adu/Abd_CF_GN'],df['Time'],threshold=178.5, bias=bias, cicle=cicle);
            cfgnPitch=patternIC(CFGNP[:,0],CFGNP[:,1],poly_degree=poly_degree,IC=IC, df=False);
            cfgn_P = rom_mean(CFGNProm)
            KGDP,KGDProm=pattern_extraction(df['Adu/Abd_Kalman_GD'],df['Time'],threshold=178, bias=bias, cicle=cicle);
            kgdPitch=patternIC(KGDP[:,0],KGDP[:,1],poly_degree=poly_degree,IC=IC, df=False);
            kgd_P = rom_mean(KGDProm)
            KGNP,KGNProm=pattern_extraction(df['Adu/Abd_Kalman_GN'],df['Time'],threshold=178, bias=bias, cicle=cicle);
            kgnPitch=patternIC(KGNP[:,0],KGNP[:,1],poly_degree=poly_degree,IC=IC, df=False);
            kgn_P = rom_mean(KGNProm)
            MADP,MADProm=pattern_extraction(df['Adu/Abd_Madgwick'],df['Time'],threshold=180, bias=bias, cicle=cicle);
            madPitch=patternIC(MADP[:,0],MADP[:,1],poly_degree=poly_degree,IC=IC, df=False);
            mad_P = rom_mean(MADProm)
            romRawPitch = max(df['Adu/Abd'])-min(df['Adu/Abd'])
            romPitchCF = max(df['Adu/Abd_CF'])-min(df['Adu/Abd_CF'])
            romPitchCfGd =max(df['Adu/Abd_CF_GD'])-min(df['Adu/Abd_CF_GD'])
            romPitchCfGn = max(df['Adu/Abd_CF_GN'])-min(df['Adu/Abd_CF_GN'])
            romPitchKalmanGd = max(df['Adu/Abd_Kalman_GD'])-min(df['Adu/Abd_Kalman_GD'])
            romPitchKalmanGn = max(df['Adu/Abd_Kalman_GN'])-min(df['Adu/Abd_Kalman_GN'])
            romPitchMad = max(df['Adu/Abd_Madgwick'])-min(df['Adu/Abd_Madgwick'])
            minRawPitch = min(df['Adu/Abd'])
            minPitchCF = min(df['Adu/Abd_CF'])
            minPitchCfGd = min(df['Adu/Abd_CF_GD'])
            minPitchCfGn = min(df['Adu/Abd_CF_GN'])
            minPitchKalmanGd = min(df['Adu/Abd_Kalman_GD'])
            minPitchKalmanGn = min(df['Adu/Abd_Kalman_GN'])
            minPitchMad = max(df['Adu/Abd_Madgwick'])
            maxRawPitch = max(df['Adu/Abd'])
            maxPitchCF = max(df['Adu/Abd_CF'])
            maxPitchCfGd = max(df['Adu/Abd_CF_GD'])
            maxPitchCfGn = max(df['Adu/Abd_CF_GN'])
            maxPitchKalmanGd = max(df['Adu/Abd_Kalman_GD'])
            maxPitchKalmanGn = max(df['Adu/Abd_Kalman_GN'])
            maxPitchMad = max(df['Adu/Abd_Madgwick'])

            indexPitch = ['Adu/Abd','Adu/Abd_CF','Adu/Abd_CF_GD','Adu/Abd_CF_GN','Adu/Abd_Kalman_GD','Adu/Abd_Kalman_GN','Adu/Abd_Madgwick']
            meanPitch = [rm_P,cf_P,cfgd_P,cfgn_P,kgd_P,kgn_P,mad_P]
            stdPitch = [rawPitch[1],cfPitch[1],cfgdPitch[1],cfgnPitch[1],kgdPitch[1],kgnPitch[1],madPitch[1]]
            ciPitch = [rawPitch[0],cfPitch[0],cfgdPitch[0],cfgnPitch[0],kgdPitch[0],kgnPitch[0],madPitch[0]]
            varPitch = [rawPitch[7],cfPitch[7],cfgdPitch[7],cfgnPitch[7],kgdPitch[7],kgnPitch[7],madPitch[7]]
            minEstPitch = [rawPitch[4],cfPitch[4],cfgdPitch[4],cfgnPitch[4],kgdPitch[4],kgnPitch[4],madPitch[4]]
            maxEstPitch = [rawPitch[5],cfPitch[5],cfgdPitch[5],cfgnPitch[5],kgdPitch[5],kgnPitch[5],madPitch[5]]
            romPitch = [romRawPitch,romPitchCF,romPitchCfGd,romPitchCfGn,romPitchKalmanGd,romPitchKalmanGn,romPitchMad]
            minPitch = [minRawPitch,minPitchCF,minPitchCfGd,minPitchCfGn,minPitchKalmanGd,minPitchKalmanGn,minPitchMad]
            maxPitch = [maxRawPitch,maxPitchCF,maxPitchCfGd,maxPitchCfGn,maxPitchKalmanGd,maxPitchKalmanGn,maxPitchMad]
            index = np.concatenate((index,indexPitch ))
            Rom = np.concatenate((Rom,romPitch ))
            Mean = np.concatenate((Mean,meanPitch ))
            Std = np.concatenate((Std,stdPitch ))
            CI = np.concatenate((CI,ciPitch ))
            Var = np.concatenate((Var,varPitch ))
            Min = np.concatenate((Min,minPitch ))
            Max = np.concatenate((Max,maxPitch ))
            MinEst = np.concatenate((MinEst,minEstPitch ))
            MaxEst = np.concatenate((MaxEst,maxEstPitch ))

        if patternYaw:
            rY,rawYrom=pattern_extraction(df['Int/Ext_Rot'],df['Time'],threshold=170, bias=bias, cicle=cicle);
            rawYaw=patternIC(rY[:,0],rY[:,1],poly_degree=poly_degree,IC=IC,df=False);
            rm_Y = rom_mean(rawYrom)            
            CFY,CFYrom=pattern_extraction(df['Int/Ext_Rot_CF'],df['Time'],threshold=170, bias=bias, cicle=cicle);
            cfYaw=patternIC(CFY[:,0],CFY[:,1],poly_degree=poly_degree,IC=IC, df=False);
            cf_Y = rom_mean(CFYrom)
            CFGDY,CFGDYrom=pattern_extraction(df['Int/Ext_Rot_CF_GD'],df['Time'],-49.2, bias=bias, cicle=cicle);
            cfgdYaw=patternIC(CFGDY[:,0],CFGDY[:,1],poly_degree=poly_degree,IC=IC, df=False);
            cfgd_Y = rom_mean(CFGDYrom)   
            CFGNY,CFGNYrom=pattern_extraction(df['Int/Ext_Rot_CF_GN'],df['Time'],threshold=-50.5, bias=bias, cicle=cicle);
            cfgnYaw=patternIC(CFGNY[:,0],CFGNY[:,1],poly_degree=poly_degree,IC=IC, df=False);
            cfgn_Y = rom_mean(CFGNYrom)
            KGDY,KGDYrom=pattern_extraction(df['Int/Ext_Rot_Kalman_GD'],df['Time'],threshold=-0.05, bias=bias, cicle=cicle);
            kgdYaw=patternIC(KGDY[:,0],KGDY[:,1],poly_degree=poly_degree,IC=IC, df=False); 
            kgd_Y = rom_mean(KGDYrom)
            KGNY,KGNYrom=pattern_extraction(df['Int/Ext_Rot_Kalman_GN'],df['Time'],threshold=np.mean(df['Int/Ext_Rot_Kalman_GN']), bias=bias, cicle=cicle);
            kgnYaw=patternIC(KGNY[:,0],KGNY[:,1],poly_degree=poly_degree,IC=IC, df=False); 
            kgn_Y = rom_mean(KGNYrom)
            MADY,MADYrom=pattern_extraction(df['Int/Ext_Rot_Madgwick'],df['Time'],threshold=-56.5, bias=bias, cicle=cicle);
            madYaw=patternIC(MADY[:,0],MADY[:,1],poly_degree=poly_degree,IC=IC, df=False);
            mad_Y = rom_mean(MADYrom)
            romRawYaw = max(df['Int/Ext_Rot'])-min(df['Int/Ext_Rot'])
            romYawCF = max(df['Int/Ext_Rot_CF'])-min(df['Int/Ext_Rot_CF'])
            romYawCfGd =max(df['Int/Ext_Rot_CF_GD'])-min(df['Int/Ext_Rot_CF_GD'])
            romYawCfGn = max(df['Int/Ext_Rot_CF_GN'])-min(df['Int/Ext_Rot_CF_GN'])
            romYawKalmanGd = max(df['Int/Ext_Rot_Kalman_GD'])-min(df['Int/Ext_Rot_Kalman_GD'])
            romYawKalmanGn = max(df['Int/Ext_Rot_Kalman_GN'])-min(df['Int/Ext_Rot_Kalman_GN'])
            romYawMad = max(df['Int/Ext_Rot_Madgwick'])-min(df['Int/Ext_Rot_Madgwick'])
            minRawYaw = min(df['Int/Ext_Rot'])
            minYawCF = min(df['Int/Ext_Rot_CF'])
            minYawCfGd = min(df['Int/Ext_Rot_CF_GD'])
            minYawCfGn = min(df['Int/Ext_Rot_CF_GN'])
            minYawKalmanGd = min(df['Int/Ext_Rot_Kalman_GD'])
            minYawKalmanGn = min(df['Int/Ext_Rot_Kalman_GN'])
            minYawMad = max(df['Int/Ext_Rot_Madgwick'])
            maxRawYaw = max(df['Int/Ext_Rot'])
            maxYawCF = max(df['Int/Ext_Rot_CF'])
            maxYawCfGd = max(df['Int/Ext_Rot_CF_GD'])
            maxYawCfGn = max(df['Int/Ext_Rot_CF_GN'])
            maxYawKalmanGd = max(df['Int/Ext_Rot_Kalman_GD'])
            maxYawKalmanGn = max(df['Int/Ext_Rot_Kalman_GN'])
            maxYawMad = max(df['Int/Ext_Rot_Madgwick'])

            indexYaw = ['Int/Ext_Rot','Int/Ext_Rot_CF','Int/Ext_Rot_CF_GD','Int/Ext_Rot_CF_GN','Int/Ext_Rot_Kalman_GD','Int/Ext_Rot_Kalman_GN','Int/Ext_Rot_Madgwick']
            meanYaw = [rm_Y,cf_Y,cfgd_Y,cfgn_Y,kgd_Y,kgn_Y,mad_Y]
            stdYaw = [rawYaw[1],cfYaw[1],cfgdYaw[1],cfgnYaw[1],kgdYaw[1],kgnYaw[1],madYaw[1]]
            ciYaw = [rawYaw[0],cfYaw[0],cfgdYaw[0],cfgnYaw[0],kgdYaw[0],kgnYaw[0],madYaw[0]]
            varYaw = [rawYaw[7],cfYaw[7],cfgdYaw[7],cfgnYaw[7],kgdYaw[7],kgnYaw[7],madYaw[7]]
            minEstYaw = [rawYaw[4],cfYaw[4],cfgdYaw[4],cfgnYaw[4],kgdYaw[4],kgnYaw[4],madYaw[4]]
            maxEstYaw = [rawYaw[5],cfYaw[5],cfgdYaw[5],cfgnYaw[5],kgdYaw[5],kgnYaw[5],madYaw[5]]
            romYaw = [romRawYaw,romYawCF,romYawCfGd,romYawCfGn,romYawKalmanGd,romYawKalmanGn,romYawMad]
            minYaw = [minRawYaw,minYawCF,minYawCfGd,minYawCfGn,minYawKalmanGd,minYawKalmanGn,minYawMad]
            maxYaw = [maxRawYaw,maxYawCF,maxYawCfGd,maxYawCfGn,maxYawKalmanGd,maxYawKalmanGn,maxYawMad]
            index = np.concatenate((index,indexYaw ))
            Rom = np.concatenate((Rom,romYaw ))
            Mean = np.concatenate((Mean,meanYaw ))
            Std = np.concatenate((Std,stdYaw ))
            CI = np.concatenate((CI,ciYaw ))
            Var = np.concatenate((Var,varYaw ))
            Min = np.concatenate((Min,minYaw ))
            Max = np.concatenate((Max,maxYaw ))
            MinEst = np.concatenate((MinEst,minEstYaw ))
            MaxEst = np.concatenate((MaxEst,maxEstYaw ))

        df_metrics = pd.DataFrame({'Movement':index,
                        'Rom':Rom,
                        ' Mean':Mean,
                        'Std':Std,
                        'CI':CI,
                        'Var':Var,
                        'Min':Min,
                        'Max':Max,
                        'Min Est':MinEst,
                        'Max Est':MaxEst})

        return df ,df_metrics

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
        https://github.com/tuliofalmeida    
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
        https://github.com/tuliofalmeida    
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
        https://github.com/tuliofalmeida    
        """  
        return np.asarray([DataHandler.csvToFloat(dfX),DataHandler.csvToFloat(dfY),DataHandler.csvToFloat(dfZ)]).T
    
    def viconDataToList(dataPath):
        """Convert Vicon's orientation data into a Dictionary.
        Device probably used Vicon Blade.

        Parameters
        ----------
        dataPath:  string
            Data path.
        
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
        for ç in range(len(columns)-1):
            temp = []
            for i in df[columns[ç]]:
                t = i.split(' ')
                temp2 = [];
                for j in t:
                    temp2.append(float(j))
                temp.append(temp2)
            dataDict[names[ç]] = temp

        return dataDict
    
    def xSens2dict(dataPath):
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
        
        Return
        ------
        dataDict: dictionary
            Dictionary separating the data of 
            each part of the body and orientation 
            (Quaternion, Acelerometer, Gyroscope 
            and Magnetometer). The autocomplete 
            works.

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
        dataDict = {}
        for j in name:
            quaternion = []
            accel = []
            gyro = []
            mag = []
            for i in range(len(output)):
                if outputNames[i] == j:
                    quaternion.append(output[i][0:4])
                    accel.append(output[i][4:7])
                    gyro.append(output[i][7:10])
                    mag.append(output[i][10:13])
            dataDict[j] = (pd.DataFrame({'Quaternion':quaternion,
                                    'Accelerometer':accel,
                                    'Gyroscope':gyro,
                                    'Magnetometer':mag
                                    }))
        return dataDict