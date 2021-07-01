import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pyjamalib
import scipy.signal,scipy.stats

class DataProcessing:
    """Integrates all functions to perform data 
    processing to calculate the joint angle.

    See Also
    --------
    Developed by T.F Almeida in 25/03/2021
    
    For more information see:
    https://github.com/tuliofalmeida/pyjama    
    """
    def __init__(self, Quaternion = [1,0,0,0]): 
        """Pass the necessary variables to run the class.
         Parameters
        ----------
        Quaternion:  tuple optional
            Initial positioning in quaternion.
        
        See Also
        --------
        Developed by T.F Almeida in 25/03/2021

        For more information see:
        https://github.com/tuliofalmeida/pyjama   
        """
        self.Quaternion = np.asarray(Quaternion)
        self.qGyro_1 = np.zeros(4)
        self.qGyro_1[0] = 1
        self.qFilt_1 = np.zeros(4)
        self.qFilt_1[0] = 1
        self.qOsserv_1 = np.zeros(4)
        self.qOsserv_1[0] = 1
        self.accF_Length = 13
        self.i = 0

    def toDataframe(data,data_calib,low_pass=.1,freq=75,dt=1/75,alpha=.01,beta=.05,beta_mad=.9,beta_mad2=.01,conj=True,gyr_filt_cf=.1,gyr_filt_k=.05):
        """This function receives the data from JAMA 
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
        low_pass: float
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
        beta_mad: float
           There is a tradeoff in the beta parameter 
           between accuracy and response speed. Must
           be determined according with the Gyroscope
           error. Used for the first quaternion 
           orientation. For MadgwickAHRS filte.
        beta_mad2: float
           There is a tradeoff in the beta parameter 
           between accuracy and response speed. Must
           be determined according with the Gyroscope
           error. Used for the others orientations.
           For MadgwickAHRS filte.
        conj: bool
            Determine if the quaternion resulted will be
            conjugated or not.
        gyr_filt_cf: float
            Low-pass filter intensity specific to Gyroscope
            data of complementary filter.
        gyr_filt_k: float
            Low-pass filter intensity specific to Gyroscope
            data of kalman filter.

        Returns
        -------
        df: pandas dataframe
            A pandas dataframe with the euler angles computed
            using quaternions formulations.

        See Also
        --------
        Developed by T.F Almeida in 25/03/2021

        For more information see:
        https://github.com/tuliofalmeida/pyjama
        https://github.com/tuliofalmeida/jama     
        """         
        time,acc,gyr,mag= pyjamalib.DataHandler.get_imu_data(data)
        time_calib,acc_calib,gyr_calib,mag_calib = pyjamalib.DataHandler.get_imu_data(data_calib)
        time = np.arange(0, len(time)/freq, dt)
        
        end_calib = 5*freq
        kalamn_gyr = gyr[0:end_calib]
         
        acc, gyr, mag = pyjamalib.DataHandler.calibration_imu(acc,gyr,mag,mag_calib)
        accf = DataProcessing.low_pass_filter(acc,low_pass)
        gyrf = DataProcessing.low_pass_filter(gyr,low_pass)
        magf = DataProcessing.low_pass_filter(mag,low_pass)

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
        
        acc_df   = pyjamalib.DataHandler.csvFloatMerge(df['Acc_X'],df['Acc_Y'],df['Acc_Z'])
        gyr_df   = pyjamalib.DataHandler.csvFloatMerge(df['Gyr_X'],df['Gyr_Y'],df['Gyr_Z'])
        mag_df   = pyjamalib.DataHandler.csvFloatMerge(df['Mag_X'],df['Mag_Y'],df['Mag_Z'])
        acc_df_f = pyjamalib.DataHandler.csvFloatMerge(df['Acc_X_Filt'],df['Acc_Y_Filt'],df['Acc_Z_Filt'])
        gyr_df_f = pyjamalib.DataHandler.csvFloatMerge(df['Gyr_X_Filt'],df['Gyr_Y_Filt'],df['Gyr_Z_Filt'])
        mag_df_f = pyjamalib.DataHandler.csvFloatMerge(df['Mag_X_Filt'],df['Mag_Y_Filt'],df['Mag_Z_Filt'])

        Roll, Pitch, Yaw = DataProcessing.get_euler(q=[1,0,0,0],Acc=acc_df_f,Mag=mag_df_f,conj=conj)
        CF    = DataProcessing.complementaryFilter(Roll,Pitch,Yaw,
                                                    DataProcessing.low_pass_filter(gyr_df[:,0],gyr_filt_cf),
                                                    DataProcessing.low_pass_filter(gyr_df[:,1],gyr_filt_cf),
                                                    DataProcessing.low_pass_filter(gyr_df[:,2],gyr_filt_cf),
                                                   alpha=alpha,dt=dt)
        CF_GD = DataProcessing.ComplementaryFilterGD(acc_df, DataProcessing.low_pass_filter(gyr_df,gyr_filt_cf),mag_df,
                                                     dt=dt,alpha=alpha,beta=beta,conj=conj,low_pass=low_pass)
        CF_GN = DataProcessing.ComplementaryFilterGN(acc_df,DataProcessing.low_pass_filter(gyr_df,gyr_filt_cf),mag_df,
                                                     dt=dt,alpha=alpha,beta=beta,conj=conj,low_pass=low_pass)
        Kalman_GD = DataProcessing.KalmanGD(acc_df,DataProcessing.low_pass_filter(gyr_df,gyr_filt_k),mag_df,gyrcalib=kalamn_gyr,
                                            dt=dt,beta=beta,conj=conj,low_pass=low_pass)
        Kalman_GN = DataProcessing.KalmanGN(acc_df,DataProcessing.low_pass_filter(gyr_df,gyr_filt_k),mag_df,gyrcalib=kalamn_gyr,
                                            dt=dt,beta=beta,conj=conj,low_pass=low_pass)
        Madgwick  = DataProcessing.MadgwickAHRS(acc_df,gyr_df,mag_df,
                                                freq=freq,beta1=beta_mad,beta2=beta_mad2)

        df['Roll'],df['Pitch'],df['Yaw'] = Roll, Pitch, Yaw
        df['CF_Roll'],df['CF_Pitch'],df['CF_Yaw'] = CF[:,0],CF[:,1],CF[:,2]
        df['CF_GD_Roll'],df['CF_GD_Pitch'],df['CF_GD_Yaw'] = CF_GD[:,0],CF_GD[:,1],CF_GD[:,2]
        df['CF_GN_Roll'],df['CF_GN_Pitch'],df['CF_GN_Yaw'] = CF_GN[:,0],CF_GN[:,1],CF_GN[:,2]
        df['Kalman_GD_Roll'],df['Kalman_GD_Pitch'],df['Kalman_GD_Yaw'] = Kalman_GD[:,0],Kalman_GD[:,1],Kalman_GD[:,2]
        df['Kalman_GN_Roll'],df['Kalman_GN_Pitch'],df['Kalman_GN_Yaw'] = Kalman_GN[:,0],Kalman_GN[:,1],Kalman_GN[:,2]
        df['Madgwick_Roll'],df['Madgwick_Pitch'],df['Madgwick_Yaw'] = Madgwick[:,0],Madgwick[:,1],Madgwick[:,2]

        return df

    def joint_measures(df_first_joint,df_second_joint,patternRoll=False,patternPitch=False,patternYaw=False,init=0,end=None,freq=75,threshold=None,cycle=2,bias=0,poly_degree=9,CI=1.96,positive=True):
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
            Dataframe with JAMA data positioned 
            above the target joint returned by the 
            'toDataFrame' function.
        df_second_joint: pandas dataframe
            Dataframe with JAMA data positioned 
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
        threshold: float
            Point at which the data moves between 
            movements. Example: flexion and extension.
        cycle: int
            Number of points to be considered a pattern.
        bias: int optional
            Value to compensate the cycle adjust.
        poly_degree: int
            Degree of the polynomial to fit the data curve.
        CI: float
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
        https://github.com/tuliofalmeida/pyjama 
        https://github.com/tuliofalmeida/jama    
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
        
        #Joint angle
        column_name = df.columns.tolist()[1:]
        row_names = df_first_joint.columns.tolist()[19:]
        for ç in zip(column_name,row_names):            
                df[ç[0]] = 180-(df_first_joint[ç[1]]+df_second_joint[ç[1]])
                if positive:
                    df[ç[0]] = (df[ç[0]] + abs(min(df[ç[0]])))*-1
                    df[ç[0]] = df[ç[0]] + abs(min(df[ç[0]]))
                
        thre = np.zeros((len(column_name),1))      
        if type(threshold) == type(None):
            for i in range(len(thre)):
                  thre[i] = np.mean(df[column_name[i]])
        else:
            for i in range(len(thre)):
                  thre[i] = threshold                 
                  
        index = []
        Rom = []
        Mean = []
        Std = []
        CIlist = []
        Var = []
        Min = []
        Max = []
        MinEst = []
        MaxEst = []

        if patternRoll:
            rR,rawRrom = DataProcessing.pattern_extraction(df['Flex/Ext'],df['Time'],threshold=thre[0], bias=bias, cycle=cycle);
            rawRoll = DataProcessing.patternCI(rR[:,0],rR[:,1],poly_degree=poly_degree,CI=CI, df=False);
            rm_R = DataProcessing.rom_mean(rawRrom)
            CFR,CFRrom=DataProcessing.pattern_extraction(df['Flex/Ext_CF'],df['Time'],threshold=thre[3], bias=bias, cycle=cycle);
            cfRoll=DataProcessing.patternCI(CFR[:,0],CFR[:,1],poly_degree=poly_degree,CI=CI, df=False);
            cf_R = DataProcessing.rom_mean(CFRrom)
            CFGDR,CFGDRrom=DataProcessing.pattern_extraction(df['Flex/Ext_CF_GD'],df['Time'],threshold=thre[6], bias=bias, cycle=cycle);
            cfgdRoll=DataProcessing.patternCI(CFGDR[:,0],CFGDR[:,1],poly_degree=poly_degree,CI=CI, df=False);
            cfgd_R= DataProcessing.rom_mean(CFGDRrom)
            CFGNR,CFGNRrom=DataProcessing.pattern_extraction(df['Flex/Ext_CF_GN'],df['Time'],threshold=thre[9], bias=bias, cycle=cycle);
            cfgnRoll=DataProcessing.patternCI(CFGNR[:,0],CFGNR[:,1],poly_degree=poly_degree,CI=CI, df=False);
            cfgn_R = DataProcessing.rom_mean(CFGNRrom)
            KGDR,KGDRrom=DataProcessing.pattern_extraction(df['Flex/Ext_Kalman_GD'],df['Time'],threshold=thre[12], bias=bias, cycle=cycle);
            kgdRoll=DataProcessing.patternCI(KGDR[:,0],KGDR[:,1],poly_degree=poly_degree,CI=CI, df=False);
            kgd_R = DataProcessing.rom_mean(KGDRrom)
            KGNR,KGNRRrom=DataProcessing.pattern_extraction(df['Flex/Ext_Kalman_GN'],df['Time'],threshold=thre[15], bias=bias, cycle=cycle);
            kgnRoll=DataProcessing.patternCI(KGNR[:,0],KGNR[:,1],poly_degree=poly_degree,CI=CI, df=False);
            kgn_R = DataProcessing.rom_mean(KGNRRrom)
            MADR,MADRrom=DataProcessing.pattern_extraction(df['Flex/Ext_Madgwick'],df['Time'],threshold=thre[18], bias=bias, cycle=cycle);
            madRoll=DataProcessing.patternCI(MADR[:,0],MADR[:,1],poly_degree=poly_degree,CI=CI, df=False);
            mad_R = DataProcessing.rom_mean(MADRrom)
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
            minRollMad = min(df['Flex/Ext_Madgwick'])
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
            CIlist = np.concatenate((CIlist,ciRoll))
            Var = np.concatenate((Var,varRoll))
            Min = np.concatenate((Min,minRoll))
            Max = np.concatenate((Max,maxRoll))
            MinEst = np.concatenate((MinEst,minEstRoll))
            MaxEst = np.concatenate((MaxEst,maxEstRoll))

        if patternPitch:    
            rP,rawProm=DataProcessing.pattern_extraction(df['Adu/Abd'],df['Time'],threshold=thre[1], bias=bias, cycle=cycle);
            rawPitch=DataProcessing.patternCI(rP[:,0],rP[:,1],poly_degree=poly_degree,CI=CI,df=False);
            rm_P = DataProcessing.rom_mean(rawProm) 
            CFP,CFProm=DataProcessing.pattern_extraction(df['Adu/Abd_CF'],df['Time'],threshold=thre[4], bias=bias, cycle=cycle);
            cfPitch=DataProcessing.patternCI(CFP[:,0],CFP[:,1],poly_degree=poly_degree,CI=CI, df=False);
            cf_P = DataProcessing.rom_mean(CFProm)
            CFGDP,CFGDProm=DataProcessing.pattern_extraction(df['Adu/Abd_CF_GD'],df['Time'],threshold=thre[7], bias=bias, cycle=cycle);
            cfgdPitch=DataProcessing.patternCI(CFGDP[:,0],CFGDP[:,1],poly_degree=poly_degree,CI=CI, df=False);
            cfgd_P = DataProcessing.rom_mean(CFGDProm)
            CFGNP,CFGNProm=DataProcessing.pattern_extraction(df['Adu/Abd_CF_GN'],df['Time'],threshold=thre[10], bias=bias, cycle=cycle);
            cfgnPitch=DataProcessing.patternCI(CFGNP[:,0],CFGNP[:,1],poly_degree=poly_degree,CI=CI, df=False);
            cfgn_P = DataProcessing.rom_mean(CFGNProm)
            KGDP,KGDProm=DataProcessing.pattern_extraction(df['Adu/Abd_Kalman_GD'],df['Time'],threshold=thre[13], bias=bias, cycle=cycle);
            kgdPitch=DataProcessing.patternCI(KGDP[:,0],KGDP[:,1],poly_degree=poly_degree,CI=CI, df=False);
            kgd_P = DataProcessing.rom_mean(KGDProm)
            KGNP,KGNProm=DataProcessing.pattern_extraction(df['Adu/Abd_Kalman_GN'],df['Time'],threshold=thre[16], bias=bias, cycle=cycle);
            kgnPitch=DataProcessing.patternCI(KGNP[:,0],KGNP[:,1],poly_degree=poly_degree,CI=CI, df=False);
            kgn_P = DataProcessing.rom_mean(KGNProm)
            MADP,MADProm=DataProcessing.pattern_extraction(df['Adu/Abd_Madgwick'],df['Time'],threshold=thre[19], bias=bias, cycle=cycle);
            madPitch=DataProcessing.patternCI(MADP[:,0],MADP[:,1],poly_degree=poly_degree,CI=CI, df=False);
            mad_P = DataProcessing.rom_mean(MADProm)
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
            CIlist = np.concatenate((CIlist,ciPitch ))
            Var = np.concatenate((Var,varPitch ))
            Min = np.concatenate((Min,minPitch ))
            Max = np.concatenate((Max,maxPitch ))
            MinEst = np.concatenate((MinEst,minEstPitch ))
            MaxEst = np.concatenate((MaxEst,maxEstPitch ))

        if patternYaw:
            rY,rawYrom=DataProcessing.pattern_extraction(df['Int/Ext_Rot'],df['Time'],threshold=thre[2], bias=bias, cycle=cycle);
            rawYaw=DataProcessing.patternCI(rY[:,0],rY[:,1],poly_degree=poly_degree,CI=CI,df=False);
            rm_Y = DataProcessing.rom_mean(rawYrom)            
            CFY,CFYrom=DataProcessing.pattern_extraction(df['Int/Ext_Rot_CF'],df['Time'],threshold=thre[5], bias=bias, cycle=cycle);
            cfYaw=DataProcessing.patternCI(CFY[:,0],CFY[:,1],poly_degree=poly_degree,CI=CI, df=False);
            cf_Y = DataProcessing.rom_mean(CFYrom)
            CFGDY,CFGDYrom=DataProcessing.pattern_extraction(df['Int/Ext_Rot_CF_GD'],df['Time'],thre[8], bias=bias, cycle=cycle);
            cfgdYaw=DataProcessing.patternCI(CFGDY[:,0],CFGDY[:,1],poly_degree=poly_degree,CI=CI, df=False);
            cfgd_Y = DataProcessing.rom_mean(CFGDYrom)   
            CFGNY,CFGNYrom=DataProcessing.pattern_extraction(df['Int/Ext_Rot_CF_GN'],df['Time'],threshold=thre[11], bias=bias, cycle=cycle);
            cfgnYaw=DataProcessing.patternCI(CFGNY[:,0],CFGNY[:,1],poly_degree=poly_degree,CI=CI, df=False);
            cfgn_Y = DataProcessing.rom_mean(CFGNYrom)
            KGDY,KGDYrom=DataProcessing.pattern_extraction(df['Int/Ext_Rot_Kalman_GD'],df['Time'],threshold=thre[14], bias=bias, cycle=cycle);
            kgdYaw=DataProcessing.patternCI(KGDY[:,0],KGDY[:,1],poly_degree=poly_degree,CI=CI, df=False); 
            kgd_Y = DataProcessing.rom_mean(KGDYrom)
            KGNY,KGNYrom=DataProcessing.pattern_extraction(df['Int/Ext_Rot_Kalman_GN'],df['Time'],threshold=thre[17], bias=bias, cycle=cycle);
            kgnYaw=DataProcessing.patternCI(KGNY[:,0],KGNY[:,1],poly_degree=poly_degree,CI=CI, df=False); 
            kgn_Y = DataProcessing.rom_mean(KGNYrom)
            MADY,MADYrom=DataProcessing.pattern_extraction(df['Int/Ext_Rot_Madgwick'],df['Time'],threshold=thre[20], bias=bias, cycle=cycle);
            madYaw=DataProcessing.patternCI(MADY[:,0],MADY[:,1],poly_degree=poly_degree,CI=CI, df=False);
            mad_Y = DataProcessing.rom_mean(MADYrom)
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
            CIlist = np.concatenate((CIlist,ciYaw ))
            Var = np.concatenate((Var,varYaw ))
            Min = np.concatenate((Min,minYaw ))
            Max = np.concatenate((Max,maxYaw ))
            MinEst = np.concatenate((MinEst,minEstYaw ))
            MaxEst = np.concatenate((MaxEst,maxEstYaw ))

        df_metrics = pd.DataFrame({'Movement':index,
                        'Rom':Rom,
                        'Mean':Mean,
                        'Std':Std,
                        'CI':CIlist,
                        'Var':Var,
                        'Min':Min,
                        'Max':Max,
                        'Min Est':MinEst,
                        'Max Est':MaxEst})

        return df ,df_metrics  
     
    def joint_rom(df):
        """ Calculate the max range of motion (ROM) 
        of a joint.

        Parameters
        ----------
        df: data frame
            The output from 'joint_measures'
            functions.

        Returns
        -------
        df_rom: data frame
            Rom of joint
        
        See Also
        --------
        Developed by T.F Almeida in 25/03/2021

        For more information see:
        https://github.com/tuliofalmeida/pyjama"""
        
        romRawRoll = max(df['Flex/Ext'])-min(df['Flex/Ext'])
        romRollCF = max(df['Flex/Ext_CF'])-min(df['Flex/Ext_CF'])
        romRollCfGd = max(df['Flex/Ext_CF_GD'])-min(df['Flex/Ext_CF_GD'])
        romRollCfGn = max(df['Flex/Ext_CF_GN'])-min(df['Flex/Ext_CF_GN'])
        romRollKalmanGd = max(df['Flex/Ext_Kalman_GD'])-min(df['Flex/Ext_Kalman_GD'])
        romRollKalmanGn = max(df['Flex/Ext_Kalman_GN'])-min(df['Flex/Ext_Kalman_GN'])
        romRollMad = max(df['Flex/Ext_Madgwick'])-min(df['Flex/Ext_Madgwick']) 
        
        romRawPitch = max(df['Adu/Abd'])-min(df['Adu/Abd'])
        romPitchCF = max(df['Adu/Abd_CF'])-min(df['Adu/Abd_CF'])
        romPitchCfGd =max(df['Adu/Abd_CF_GD'])-min(df['Adu/Abd_CF_GD'])
        romPitchCfGn = max(df['Adu/Abd_CF_GN'])-min(df['Adu/Abd_CF_GN'])
        romPitchKalmanGd = max(df['Adu/Abd_Kalman_GD'])-min(df['Adu/Abd_Kalman_GD'])
        romPitchKalmanGn = max(df['Adu/Abd_Kalman_GN'])-min(df['Adu/Abd_Kalman_GN'])
        romPitchMad = max(df['Adu/Abd_Madgwick'])-min(df['Adu/Abd_Madgwick'])  
        
        romRawYaw = max(df['Int/Ext_Rot'])-min(df['Int/Ext_Rot'])
        romYawCF = max(df['Int/Ext_Rot_CF'])-min(df['Int/Ext_Rot_CF'])
        romYawCfGd =max(df['Int/Ext_Rot_CF_GD'])-min(df['Int/Ext_Rot_CF_GD'])
        romYawCfGn = max(df['Int/Ext_Rot_CF_GN'])-min(df['Int/Ext_Rot_CF_GN'])
        romYawKalmanGd = max(df['Int/Ext_Rot_Kalman_GD'])-min(df['Int/Ext_Rot_Kalman_GD'])
        romYawKalmanGn = max(df['Int/Ext_Rot_Kalman_GN'])-min(df['Int/Ext_Rot_Kalman_GN'])
        romYawMad = max(df['Int/Ext_Rot_Madgwick'])-min(df['Int/Ext_Rot_Madgwick'])
        
        row_names = df.columns.tolist()[1:]

        columns = [romRawRoll     ,romRawPitch     ,romRawYaw,
                   romRollCF      ,romPitchCF      ,romYawCF,
                   romRollCfGd    ,romPitchCfGd    ,romYawCfGd,
                   romRollCfGn    ,romPitchCfGn    ,romYawCfGn,
                   romRollKalmanGd,romPitchKalmanGd,romYawKalmanGd,
                   romRollKalmanGn,romPitchKalmanGn,romYawKalmanGn,
                   romRollMad     ,romPitchMad     ,romYawMad]
                  
        df_rom = pd.DataFrame({'Angles':columns},row_names )
        
        return df_rom
     
    def get_roll(acc):
        """ Calculate the euler roll angle using the
        Accelerometer data.

        Parameters
        ----------
        acc: ndarray
            Accelerometer data (XYZ) in rad/s.

        Returns
        -------
        Roll angle
        
        See Also
        --------
        Developed by T.F Almeida in 25/03/2021

        For more information see:
        https://github.com/tuliofalmeida/pyjama  """
        return math.atan2(-acc[0],math.sqrt((acc[1]*acc[1]) + (acc[2] * acc[2])))

    def get_pitch(acc):
        """ Calculate the euler pitch angle using the
        Accelerometer data.

        Parameters
        ----------
        acc: ndarray
            Accelerometer data (XYZ) in rad/s.

        Returns
        -------
        Pitch angle

        See Also
        --------
        Developed by T.F Almeida in 25/03/2021

        For more information see:
        https://github.com/tuliofalmeida/pyjama  """
        return math.atan2(acc[1],math.sqrt((acc[0] * acc[0]) + (acc[2] * acc[2])))

    def get_yaw(roll,pitch):
        """ Calculate the euler yaw angle using the
        Magnetometer data, roll and pitch angles.

        Parameters
        ----------
        mag: ndarray
            Accelerometer data (XYZ).
        roll: ndarray
            Roll angle.
        pitch: ndarray
            Pitch angle.

        Returns
        -------
        Yaw angle
        
        See Also
        --------
        Developed by T.F Almeida in 25/03/2021

        For more information see:
        https://github.com/tuliofalmeida/pyjama  """
        Yh = (mag[1] * math.cos(roll)) - (mag[2] * math.pi(roll))
        Xh = (mag[0] * math.cos(pitch))+ (mag[1] * math.sin(roll)*math.sin(pitch)) + (mag[2] * math.cos(roll) * math.sin(pitch))	
        return math.atan2(Yh, Xh)

    def get_euler(q,Acc,Mag,conj=True):
        """ Calculate the euler yaw angle using the
        Magnetometer data, roll and pitch angles.

        Parameters
        ----------
        q: ndarray
            orientation in quaternion.
        Acc: ndarray
            Accelerometer data (XYZ) in rad/s.
        Mag: ndarray
            Magnetometer data (XYZ) in mG.

        Returns
        -------
        Roll, Pitch and Yaw angles.

        See Also
        --------
        Developed by T.F Almeida in 25/03/2021

        For more information see:
        https://github.com/tuliofalmeida/pyjama  """
        quat = []
        euler = []
        quat.append(np.asarray(q))
        
        for i in range(len(Acc)):
            quat.append(DataProcessing.GaussNewton(quat[i-1],Acc[i],Mag[i]))
            if conj == True:
                euler.append(DataProcessing.GetAnglesFromQuaternion(DataProcessing.quaternConj(quat[i])))
            else:
                euler.append(DataProcessing.GetAnglesFromQuaternion(quat[i]))

        euler = np.asarray(euler)
        roll  = euler[:,0] #x
        pitch = euler[:,1] #y
        yaw   = euler[:,2] #z

        return roll,pitch,yaw

    def axisAngle2quatern(axis, angle):
        """ Converts and axis-angle orientation to a 
        quaternion where a 3D rotation is described 
        by an angular rotation around axis defined 
        by a vector.

        Parameters
        ----------
        axis: vector
            Data with 3 dimensions (XYZ).
        angle: ndarray
            angular rotation.

        Returns
        -------
        q: quaternion tuple

        See Also
        --------
        Developed by S.O.H Madgwick in 27/09/2011
        Adapted for python T.F Almeida in 25/03/2021
        
        For more information see:
        Test scripts
        http://www.x-io.co.uk/node/8#quaternions
        https://github.com/tuliofalmeida/pyjama
        """
        q0 = math.cos(np.divide(angle,2))
        q1 = -axis[0]*math.sin(np.divide(angle,2))
        q2 = -axis[1]*math.sin(np.divide(angle,2))
        q3 = -axis[2]*math.sin(np.divide(angle,2))
        q = [q0,q1,q2,q3]

        return q

    def axisAngle2rotMat(axis, angle):
        """ Converts and axis-angle orientation to a 
            rotation matrix where a 3D rotation is 
            described by an angular rotation around 
            axis defined by a vector.

        Parameters
        ----------
        axis: vector
            Data with 3 dimensions (XYZ).
        angle: ndarray
            angular rotation.

        Returns
        -------
        R: rotation matrix

        See Also
        --------
        Developed by S.O.H Madgwick in 27/09/2011
        Adapted for python T.F Almeida in 25/03/2021
        
        For more information see:
        Test scripts
        http://www.x-io.co.uk/node/8#quaternions
        https://github.com/tuliofalmeida/pyjama
        """
        kx = axis[0]
        ky = axis[1]
        kz = axis[2]
        cT = math.cos(angle)
        sT = math.sin(angle)
        vT = 1 - math.cos(angle)

        R = np.zeros((3,3))   

        R[0,0] = kx * kx * vT + cT
        R[0,1] = kx * ky * vT - kz * sT
        R[0,2] = kx * kz * vT + ky * sT

        R[1,0] = kx * ky *vT + kz * sT
        R[1,1] = ky * ky *vT + cT
        R[1,2]=  ky * kz *vT - kx * sT

        R[2,0] = kx * kz *vT - ky * sT
        R[2,1] = ky * kz *vT + kx * sT
        R[2,2] = kz * kz *vT + cT

        return R

    def euler2rotMat(phi, theta, psi):
        """ Converts ZYX Euler angle orientation to 
            a rotation matrix where phi is a rotation 
            around X, theta around Y and psi around Z.

        Parameters
        ----------
        phi: vector
            rotation around X.
        theta: vector
            rotation around Y.
        psi: vector
            rotation around Z.

        Returns
        -------
        R: rotation matrix

        See Also
        --------
        Developed by S.O.H Madgwick in 27/09/2011
        Adapted for python T.F Almeida in 25/03/2021
        
        For more information see:
        Test scripts
        http://www.x-io.co.uk/node/8#quaternions
        https://github.com/tuliofalmeida/pyjama
        """
        R = np.zeros((3,3))

        R[0,0] = math.cos(psi) * math.cos(theta) 
        R[0,1] = -math.sin(psi) * math.cos(phi) + math.cos(psi) * math.sin(theta) * math.sin(phi) 
        R[0,2] = math.sin(psi) * math.sin(phi) + math.cos(psi) * math.sin(theta) * math.cos(phi)

        R[1,0] = math.sin(psi) * math.cos(theta)
        R[1,1] = math.cos(psi) * math.cos(phi) + math.sin(psi) * math.sin(theta) * math.sin(phi)
        R[1,2] = -math.cos(psi) * math.sin(phi) + math.sin(psi) * math.sin(theta) * math.cos(phi)

        R[2,0] = -math.sin(theta)
        R[2,1] = math.cos(theta) * math.sin(phi)
        R[2,2] = math.cos(theta) * math.cos(phi)

        return R
    
    def euler2quat(euler):
        """ Converts a Euler angles to quaternions
        performing euler2rotmat and rotMat2quatern.

        Parameters
        ----------
        euler: ndarray
            quaternion orientation.

        Returns
        -------
        quat: ndarray
            quaternion orientation.

        See Also
        --------
        Developed T.F Almeida in 25/03/2021 using
        S.O.H Madgwick library.
        
        For more information see:
        Test scripts
        http://www.x-io.co.uk/node/8#quaternions
        https://github.com/tuliofalmeida/pyjama
        """
        rot = [DataProcessing.euler2rotMat(euler[i][0],euler[i][1],euler[i][2]) for i in range(len(euler))]
        quat = [DataProcessing.rotMat2quatern(rot[i]) for i in range(len(rot))]

        return np.asarray(quat)
    
    def quatern2euler(q):
        """ Converts a quaternion orientation to 
            XYZ Euler angles where phi is a rotation 
            around X, theta around Y and psi around Z.

        Parameters
        ----------
        q: tuple
            quaternion orientation.

        Returns
        -------
        euler: vector with phi(Z),theta(Y),psi(Z)

        See Also
        --------
        Developed by S.O.H Madgwick in 27/09/2011
        Adapted for python T.F Almeida in 25/03/2021
        
        For more information see:
        Test scripts
        http://www.x-io.co.uk/node/8#quaternions
        https://github.com/tuliofalmeida/pyjama
        """
        R = np.zeros((3,3))
        q = np.asarray(q)

        if q.shape == (4,):
            R[0,0] = 2*q[0]**2-1+2 * q[1]**2
            R[1,0] = 2*(q[1]*q[2]-q[0]*q[3])
            R[2,0] = 2*(q[1]*q[3]+q[0]*q[2])
            R[2,1] = 2*(q[2]*q[3]-q[0]*q[1])
            R[2,2] = 2*q[0]**2-1+2*q[3]**2
        else:
            R[0,0] = 2*q[0][0]**2-1+2 * q[0][1]**2
            R[1,0] = 2*(q[0][1]*q[0][2]-q[0][0]*q[0][3])
            R[2,0] = 2*(q[0][1]*q[0][3]+q[0][0]*q[0][2])
            R[2,1] = 2*(q[0][2]*q[0][3]-q[0][0]*q[0][1])
            R[2,2] = 2*q[0][0]**2-1+2*q[0][3]**2

        phi = math.atan2(R[2,1], R[2,2])
        theta = -math.atan(R[2,0]/math.sqrt(1-R[2,0]**2))
        psi = math.atan2(R[1,0], R[0,0] )

        euler = [phi,theta,psi]

        return euler

    def quatern2rotMat(q):
        """ Converts a quaternion orientation 
            to a rotation matrix.

        Parameters
        ----------
        q: tuple
            quaternion orientation.

        Returns
        -------
        R: Rotation matrix

        See Also
        --------
        Developed by S.O.H Madgwick in 27/09/2011
        Adapted for python T.F Almeida in 25/03/2021
        
        For more information see:
        Test scripts
        http://www.x-io.co.uk/node/8#quaternions
        https://github.com/tuliofalmeida/pyjama
        """
        R = np.zeros((3,3))

        R[0,0] = 2*q[0]**2-1+2*q[1]**2
        R[0,1] = 2*(q[1]*q[2]+q[0]*q[3])
        R[0,2] = 2*(q[1]*q[3]-q[0]*q[2])
        R[1,0] = 2*(q[1]*q[2]-q[0]*q[3])
        R[1,1] = 2*q[0]**2-1+2*q[2]**2
        R[1,2] = 2*(q[2]*q[3]+q[0]*q[1])
        R[2,0] = 2*(q[1]*q[3]+q[0]*q[2])
        R[2,1] = 2*(q[2]*q[3]-q[0]*q[1])
        R[2,2] = 2*q[0]**2-1+2*q[3]**2

        return R

    def quaternConj(q):
        """Converts a quaternion to its conjugate.

        Parameters
        ----------
        q: tuple
            quaternion orientation.

        Returns
        -------
        qConj: quaternions conjugated 

        See Also
        --------
        Developed by S.O.H Madgwick in 27/09/2011
        Adapted for python T.F Almeida in 25/03/2021
        
        For more information see:
        Test scripts
        http://www.x-io.co.uk/node/8#quaternions
        https://github.com/tuliofalmeida/pyjama
        """
        qConj = np.zeros((4))
        if q.shape == (4,):
            qConj[0] = q[0]
            qConj[1] = -q[1]
            qConj[2] = -q[2]
            qConj[3] = -q[3]
        else:
            qConj[0] = q[0][0]
            qConj[1] = -q[0][1]
            qConj[2] = -q[0][2]
            qConj[3] = -q[0][3]
        
        return qConj

    def quaternProd(a, b):
        """Calculates the quaternion product of 
        quaternion a and b.

        Parameters
        ----------
        a: quatenion tuple
            quaternion orientation.
        b: quatenion tuple
            quaternion orientation.

        Returns
        -------
        ab: quaternions conjugated 

        See Also
        --------
        Developed by S.O.H Madgwick in 27/09/2011
        Adapted for python T.F Almeida in 25/03/2021
        
        For more information see:
        Test scripts
        http://www.x-io.co.uk/node/8#quaternions
        https://github.com/tuliofalmeida/pyjama
        """
        ab = np.zeros(4)

        if np.size(np.asarray(a).shape) == 2:
            a = np.asarray(a[0])

        ab[0]     = a[0]  *b[0]   -a[1]   *b[1]  -a[2]   *b[2]  -a[3]   *b[3]
        ab[1]   = a[0]*   b[1]  +a[1]   *b[0]  +a[2]   *b[3]  -a[3]   *b[2]
        ab[2]   =  a[0]   *b[2]  -a[1]   *b[3]  +a[2]   *b[0]  +a[3]*   b[1]
        ab[3]     = a[0]   *b[3]  +a[1]   *b[2]  -a[2]   *b[1]  +a[3]   *b[0]

        return ab

    def rotMat2euler(R):
        """Converts a rotation matrix orientation 
            to ZYX Euler angles where phi is a 
            rotation around X, theta around Y and 
            psi around Z.

        Parameters
        ----------
        R: matrix
           rotation matrix.

        Returns
        -------
        euler: ndarray
               Angles in euler rotation 

        See Also
        --------
        Developed by S.O.H Madgwick in 27/09/2011
        Adapted for python T.F Almeida in 25/03/2021
        
        For more information see:
        Test scripts
        http://www.x-io.co.uk/node/8#quaternions
        https://github.com/tuliofalmeida/pyjama
        """
        if np.size(np.asarray(R).shape) == 2:
            row, col = np.asarray(R).shape
            numR = 1
            phi = math.atan2(R[2,1], R[2,2] )
            theta = -math.atan(R[2,0]/math.sqrt(1-R[2,0]**2))
            psi = math.atan2(R[1,0], R[0,0] )

            euler = [phi,theta,psi]

            return euler
            
        else:
            row, col, numR = np.asarray(R).shape
            phi = []
            theta = []
            psi =[]
            for i in range(numR):
                phi.append(math.atan2(R[2,1,i], R[2,2,i]))
                theta.append(-math.atan(R[2,0,i]/math.sqrt(1-R[2,0,i]**2)))
                psi.append(math.atan2(R[1,0,i], R[0,0,i]))

            euler = [np.asarray(phi),np.asarray(theta),np.asarray(psi)] 

            return euler

    def rotMat2quatern(R):
        """Converts a rotation matrix orientation 
           to ZYX Euler angles where phi is a 
           rotation around X, theta around Y and 
           psi around Z.

        Parameters
        ----------
        R: matrix
           rotation matrix.

        Returns
        -------
        q: tuple
           quaternion orientation

        See Also
        --------
        Developed by S.O.H Madgwick in 27/09/2011
        Adapted for python T.F Almeida in 25/03/2021
        
        For more information see:
        Test scripts
        http://www.x-io.co.uk/node/8#quaternions
        https://github.com/tuliofalmeida/pyjama
        """
        if np.size(np.asarray(R).shape) == 2:
            row, col = np.asarray(R).shape
            numR = 1
            K = np.zeros((4,4))
            K[0,0] = (1/3) * (R[0,0] - R[1,1] - R[2,2])
            K[0,1] = (1/3) * (R[1,0] + R[0,1])
            K[0,2] = (1/3) * (R[2,0] + R[0,2])
            K[0,3] = (1/3) * (R[1,2] - R[2,1])
            K[1,0] = (1/3) * (R[1,0] + R[0,1])
            K[1,1] = (1/3) * (R[1,1] - R[0,0] - R[2,2])
            K[1,2] = (1/3) * (R[2,1] + R[1,2])
            K[1,3] = (1/3) * (R[2,0] - R[0,2])
            K[2,0] = (1/3) * (R[2,0] + R[0,2])
            K[2,1] = (1/3) * (R[2,1] + R[1,2])
            K[2,2] = (1/3) * (R[2,2] - R[0,0] - R[1,1])
            K[2,3] = (1/3) * (R[0,1] - R[1,0])
            K[3,0] = (1/3) * (R[1,2] - R[2,1])
            K[3,1] = (1/3) * (R[2,0] - R[0,2])
            K[3,2] = (1/3) * (R[0,1] - R[1,0])
            K[3,3] = (1/3) * (R[0,0] + R[1,1] + R[2,2])
            D,V = np.linalg.eig(K)
            q = [-V[3][1],-V[0][1],-V[1][1],-V[2][1]]

        else:
            row, col, numR = np.asarray(R).shape
            q = np.zeros((numR, 4))
            K = np.zeros((4,4))
            for i in range(numR):
                K[0,0] = (1/3) * (R[0,0,i] - R[1,1,i] - R[2,2,i])
                K[0,1] = (1/3) * (R[1,0,i] + R[0,1,i])
                K[0,2] = (1/3) * (R[2,0,i] + R[0,2,i])
                K[0,3] = (1/3) * (R[1,2,i] - R[2,1,i])
                K[1,0] = (1/3) * (R[1,0,i] + R[0,1,i])
                K[1,1] = (1/3) * (R[1,1,i] - R[0,0,i] - R[2,2,i])
                K[1,2] = (1/3) * (R[2,1,i] + R[1,2,i])
                K[1,3] = (1/3) * (R[2,0,i] - R[0,2,i])
                K[2,0] = (1/3) * (R[2,0,i] + R[0,2,i])
                K[2,1] = (1/3) * (R[2,1,i] + R[1,2,i])
                K[2,2] = (1/3) * (R[2,2,i] - R[0,0,i] - R[1,1,i])
                K[2,3] = (1/3) * (R[0,1,i] - R[1,0,i])
                K[3,0] = (1/3) * (R[1,2,i] - R[2,1,i])
                K[3,1] = (1/3) * (R[2,0,i] - R[0,2,i])
                K[3,2] = (1/3) * (R[0,1,i] - R[1,0,i])
                K[3,3] = (1/3) * (R[0,0,i] + R[1,1,i] + R[2,2,i])
            
                D,V = np.linalg.eig(K)
                q[i,:] = -V[:,1]   

        return q

        #https://github.com/danicomo/9dof-orientation-estimation

    def computeJacobian(q,Acc,Mag):
        """Compute the Jacobian matrix using
           inital orientation in quaternion and
           Accelerometer and Magnetometer data.

        Parameters
        ----------
        q: tuple
           quaternion initial orientation.
        Acc: vector
           Accelerometer data (XYZ) in rad/s.
        Mag: vector
           Magnetometer data (XYZ) in mG.

        Returns
        -------
        q: tuple
           quaternion orientation.

        See Also
        --------
        Developed by Comotti & Ermidoro in 31/03/2017
        Adapted for python T.F Almeida in 25/03/2021
        
        For more information see:
        Test scripts
        https://github.com/danicomo/9dof-orientation-estimation
        https://github.com/tuliofalmeida/pyjama
        """

        # Compute the quaternion Jacobian
        a = q[1]
        b = q[2]
        c = q[3]
        d = q[0]

        Ax,Ay,Az = Acc[0],Acc[1],Acc[2]
        Mx,My,Mz = Mag[0],Mag[1],Mag[2]

        J_temp = np.zeros((6,4))

        J_temp[0][0] = (2*a*Ax+2*b*Ay+2*c*Az)
        J_temp[0][1] = (-2*b*Ax+2*a*Ay+2*d*Az)
        J_temp[0][2] = (-2*c*Ax-2*d*Ay+2*a*Az)
        J_temp[0][3] = (2*d*Ax-2*c*Ay+2*b*Az)

        J_temp[1][0] = (2*b*Ax-2*a*Ay-2*d*Az)
        J_temp[1][1] = (2*a*Ax+2*b*Ay+2*c*Az)
        J_temp[1][2] = (2*d*Ax-2*c*Ay+2*b*Az)
        J_temp[1][3] = (2*c*Ax+2*d*Ay-2*a*Az)

        J_temp[2][0] = (2*c*Ax+2*d*Ay-2*a*Az)
        J_temp[2][1] = (-2*d*Ax+2*c*Ay-2*b*Az)
        J_temp[2][2] = (2*a*Ax+2*b*Ay+2*c*Az)
        J_temp[2][3] = (-2*b*Ax+2*a*Ay+2*d*Az)

        J_temp[3][0] = (2*a*Mx+2*b*My+2*c*Mz)
        J_temp[3][1] = (-2*b*Mx+2*a*My+2*Mz*d)
        J_temp[3][2] = (-2*c*Mx-2*d*My+2*a*Mz)
        J_temp[3][3] = (2*d*Mx-2*c*My+2*b*Mz)

        J_temp[4][0] = (2*b*Mx-2*a*My-2*d*Mz)
        J_temp[4][1] = (2*a*Mx+2*b*My+2*c*Mz)
        J_temp[4][2] = (2*d*Mx-2*c*My+2*b*Mz)
        J_temp[4][3] = (2*c*Mx+2*d*My-2*a*Mz)

        J_temp[5][0] = (2*c*Mx+2*d*My-2*a*Mz)
        J_temp[5][1] = (-2*d*Mx+2*c*My-2*b*Mz)
        J_temp[5][2] = (2*a*Mx+2*b*My+2*c*Mz)
        J_temp[5][3] = (-2*b*Mx+2*a*My+2*d*Mz)

        J = [[J_temp[0][0], J_temp[0][1], J_temp[0][2], J_temp[0][3]],
            [J_temp[1][0], J_temp[1][1], J_temp[1][2], J_temp[1][3]],
            [J_temp[2][0], J_temp[2][1], J_temp[2][2], J_temp[2][3]],
            [J_temp[3][0], J_temp[3][1], J_temp[3][2], J_temp[3][3]],
            [J_temp[4][0], J_temp[4][1], J_temp[4][2], J_temp[4][3]],
            [J_temp[5][0], J_temp[5][1], J_temp[5][2], J_temp[5][3]]]
        
        return -np.asarray(J)

    def computeM_Matrix(q):
        """Compute the rotation transformation 
           matrix based on quaternions.

        Parameters
        ----------
        q: tuple
           quaternion orientation.

        Returns
        -------
        M: tuple
           rotation transformation matrix.

        See Also
        --------
        Developed by Comotti & Ermidoro in 31/03/2017
        Adapted for python T.F Almeida in 25/03/2021
        
        For more information see:
        Test scripts
        https://github.com/danicomo/9dof-orientation-estimation
        https://github.com/tuliofalmeida/pyjama
        """
        # Compute the rotation transformation matrix based on quaternions

        a = q[1]
        b = q[2]
        c = q[3]
        d = q[0]

        R_temp = np.zeros((3,3))

        R_temp[0][0] = d**2+a**2-b**2-c**2
        R_temp[0][1] = 2*(a*b-c*d)
        R_temp[0][2] = 2*(a*c+b*d)
        R_temp[1][0] = 2*(a*b+c*d)
        R_temp[1][1] = d**2+b**2-a**2-c**2
        R_temp[1][2] = 2*(b*c-a*d)
        R_temp[2][0] = 2*(a*c-b*d)
        R_temp[2][1] = 2*(b*c+a*d)
        R_temp[2][2] = d**2+c**2-b**2-a**2

        M = np.concatenate( (np.concatenate((R_temp         , np.zeros((3,3))), axis=1),
                            np.concatenate((np.zeros((3,3)), R_temp),          axis=1)), axis=0)

        return np.asarray(M)

    def GaussNewton(q,Acc,Mag):
        """Estimates the quaternion orientation
           from the accelerometer and magnetometer 
           data based on the Guass-Newton optimizer.

        Parameters
        ----------
        q: tuple
           quaternion orientation.
        Acc: ndarray
           Array with XYZ in rad/s
        Mag:
           Array wit XYZ in mG

        Returns
        -------
        q: tuple
           quaternion estimated with Gauss-Newton.

        See Also
        --------
        Developed by Comotti & Ermidoro in 31/03/2017
        Adapted for python T.F Almeida in 25/03/2021
        
        For more information see:
        Test scripts
        https://github.com/danicomo/9dof-orientation-estimation
        https://github.com/tuliofalmeida/pyjama
        """
        # Compute the new step quaternions by mean of the Gauss-Newton method
        i = 0
        n_k = np.asarray([q[1], q[2],q[3], q[0]]).T
        
        while(i < 3):
            # Magnetometer compensation
            m = Mag/np.linalg.norm(Mag)
            q_coniug = np.asarray([q[0],-q[1],-q[2],-q[3]])
            MAG = np.asarray([0,m[0],m[1],m[2]])
            hTemp = DataProcessing.QuaternionProduct(q,MAG)
            h = DataProcessing.QuaternionProduct(hTemp,q_coniug)
            bMagn = np.asarray([math.sqrt(h[1]**2+h[2]**2), 0, h[3]]).T
            bMagn = bMagn/np.linalg.norm(bMagn)
            # End magnetometer compensation
            
            J_nk = DataProcessing.computeJacobian(q,Acc,Mag)

            M = DataProcessing.computeM_Matrix(q)

            y_e=np.concatenate((np.asarray([0, 0, 1]).T,bMagn), axis=0)
            y_b=np.concatenate((Acc,Mag), axis=0)

            # Gauss-Newton step
            n = n_k-(np.linalg.inv(J_nk.T@J_nk))@J_nk.T@(y_e-M@y_b)
            n = n/np.linalg.norm(n)
            n_k = n
            q = np.asarray([n[3], n[0], n[1], n[2]])
            i += 1

        q = np.asarray([n[3], n[0], n[1], n[2]]).T
        return q

    def GetAnglesFromQuaternion(q):
        """Estimates the euler angles from
           quaternion orientation.

        Parameters
        ----------
        q: tuple
           quaternion orientation.

        Returns
        -------
        Angles: ndarray
           Euler angles (Roll,Pitch,Yaw) in
           ndarray.

        See Also
        --------
        Developed by Comotti & Ermidoro in 31/03/2017
        Adapted for python T.F Almeida in 25/03/2021
        
        For more information see:
        Test scripts
        https://github.com/danicomo/9dof-orientation-estimation
        https://github.com/tuliofalmeida/pyjama
        """
        q0=q[0]
        q1=q[1]
        q2=q[2]
        q3=q[3]
        
        AnglesX = math.atan2(2*(q2*q3)-2*q0*q1,2*q0**2+2*q3**2-1)*180/math.pi
        AnglesY = -math.asin(2*q1*q3+2*q0*q2)*180/math.pi
        AnglesZ = math.atan2(2*q1*q2-2*q0*q3,2*q0**2+2*q1**2-1)*180/math.pi

        Angles = np.asarray([AnglesX,AnglesY,AnglesZ])

        return Angles

    def GetQuaternionFromAngle(Angles):
        """Estimates the quaternion orientation
            from euler angles.

        Parameters
        ----------
        Angles: ndarray
           Euler angles (Roll,Pitch,Yaw)
           in degrees.

        Returns
        -------
        q: ndarray
           Orientation in quaternion.

        See Also
        --------
        Developed by Comotti & Ermidoro in 31/03/2017
        Adapted for python T.F Almeida in 25/03/2021
        
        For more information see:
        Test scripts
        https://github.com/danicomo/9dof-orientation-estimation
        https://github.com/tuliofalmeida/pyjama
        """       
        q = np.zeros((4))
        x = Angles[0]*math.pi/180
        y = Angles[1]*math.pi/180
        z = Angles[2]*math.pi/180

        q[0] = math.cos(x/2)*math.cos(y/2)*math.cos(z/2)+math.sin(x/2)*math.sin(y/2)*math.sin(z/2)
        q[1] = math.sin(x/2)*math.cos(y/2)*math.cos(z/2)-math.cos(x/2)*math.sin(y/2)*math.sin(z/2)
        q[2] = math.cos(x/2)*math.sin(y/2)*math.cos(z/2)+math.sin(x/2)*math.cos(y/2)*math.sin(z/2)
        q[3] = math.cos(x/2)*math.cos(y/2)*math.sin(z/2)-math.sin(x/2)*math.sin(y/2)*math.cos(z/2)

        return np.asarray(q)

    def GradientDescent(Acc,Mag,q,mu):
        """Estimates the quaternion orientation
           from the accelerometer and magnetometer 
           data based on the Gradient Descendent 
           optimizer.

        Parameters
        ----------
        Acc: ndarray
           Accelerometer array with XYZ in rad/s
        Mag: ndarray 
           Magnetometer array with XYZ in mG
        q: tuple
           quaternion orientation.
        mu: ndarray
           normalized quaternion data integrated
           with the Gyroscope data.

        Returns
        -------
        q: tuple
           quaternion estimated with Gradient-Descendent.

        See Also
        --------
        Developed by Comotti & Ermidoro in 31/03/2017
        Adapted for python T.F Almeida in 25/03/2021
        
        For more information see:
        Test scripts
        https://github.com/danicomo/9dof-orientation-estimation
        https://github.com/tuliofalmeida/pyjama
        """
        q1=q[0]
        q2=q[1]
        q3=q[2]
        q4=q[3]

        i = 1
        while(i <= 10):
            fg1 = 2*(q2*q4-q1*q3)-Acc[0]
            fg2 = 2*(q1*q2+q3*q4)-Acc[1]
            fg3 = 2*(0.5-q2**2-q3**2)-Acc[2]
            fg = np.asarray([fg1,fg2,fg3])

            Jg1 = [-2*q3, 2*q4, -2*q1, 2*q2]
            Jg2 = [2*q2, 2*q1, 2*q4, 2*q3]
            Jg3 = [0, -4*q2, -4*q3, 0]

            Jg = np.asarray([Jg1,Jg2,Jg3])

            m = Mag/np.linalg.norm(Mag)

            q_coniug = [q[0],-q[1],-q[2],-q[3]]

            M = [0,m[0],m[1],m[2]]
            hTemp = DataProcessing.QuaternionProduct(q,M)
            h = DataProcessing.QuaternionProduct(hTemp,q_coniug)

            b = [math.sqrt(h[1]**2 + h[2]**2),0,h[3]]
            b = b/np.linalg.norm(b)

            fb1 = 2*b[0]*(0.5-q3**2-q4**2)+2*b[2]*(q2*q4-q1*q3)-m[0]
            fb2 = 2*b[0]*(q2*q3-q1*q4)+2*b[2]*(q1*q2+q3*q4)-m[1]
            fb3 = 2*b[0]*(q1*q3+q2*q4)+2*b[2]*(0.5-q2**2-q3**2)-m[2]
            fb = np.asarray([fb1,fb2,fb3])

            Jb1 = [-2*b[2]*q3          , 2*b[2]*q4           , -4*b[0]*q3-2*b[2]*q1,-4*b[0]*q4+2*b[2]*q2]
            Jb2 = [-2*b[0]*q4+2*b[2]*q2, 2*b[0]*q3+2*b[2]*q1 , 2*b[0]*q2+2*b[2]*q4 , -2*b[0]*q1+2*b[2]*q3]
            Jb3 = [2*b[0]*q3           , 2*b[0]*q4-4*b[2]*q2 , 2*b[0]*q1-4*b[2]*q3 , 2*b[0]*q2]
            Jb = np.asarray([Jb1,Jb2,Jb3])

            fgb = np.concatenate((fg,fb), axis=0)
            Jgb = np.concatenate((Jg,Jb), axis=0)

            Df = Jgb.T @ fgb

            q_Temp = q-mu*Df/np.linalg.norm(Df)

            q_result = np.asarray([q_Temp[0],q_Temp[1],q_Temp[2],q_Temp[3]])
            
            q_result = q_result/np.linalg.norm(q_result)
            
            q = np.asarray([q_result[0],q_result[1],q_result[2],q_result[3]])
            q1 = q_result[0] 
            q2 = q_result[1]
            q3 = q_result[2]
            q4 = q_result[3]
            i += 1

        return q

    def QuaternionProduct(a,b):
        """Calculate the quaternion product
           from two quaterions.

        Parameters
        ----------
        a: tuple
           quaternion orientation
        b: tuple 
           quaternion orientation

        Returns
        -------
        P: ndarray
           quaternion conjugated.

        See Also
        --------
        Developed by Comotti & Ermidoro in 31/03/2017
        Adapted for python T.F Almeida in 25/03/2021
        
        For more information see:
        Test scripts
        https://github.com/danicomo/9dof-orientation-estimation
        https://github.com/tuliofalmeida/pyjama
        """
        a1 = a[0]
        a2 = a[1]
        a3 = a[2]
        a4 = a[3]
        
        b1 = b[0]
        b2 = b[1]
        b3 = b[2]
        b4 = b[3]
        
        P1 = a1*b1-a2*b2-a3*b3-a4*b4
        P2 = a1*b2+a2*b1+a3*b4-a4*b3
        P3 = a1*b3-a2*b4+a3*b1+a4*b2
        P4 = a1*b4+a2*b3-a3*b2+a4*b1

        P = [P1,P2,P3,P4]

        return np.asarray(P)

    def butter_lowpass(cutoff, freq = 75, order=2):
        """Low-pass butter filter parameters

        Parameters
        ----------
        cutoff: float
           Filter cutoff frequency.
        freq: int 
           Frequency of data acquisition.
        order: int
           Polynomials filter order

        Returns
        -------
        b: ndarray
           Polynomials of the IIR filter.

        a: ndarray
           Polynomials of the IIR filter.

        See Also
        --------
        Developed by T.F Almeida in 25/03/2021
        
        For more information see:
        https://github.com/tuliofalmeida/pyjama
        """
        nyq = 0.5 * freq
        normal_cutoff = cutoff / nyq
        b, a = scipy.signal.butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    def butter_lowpass_filter(cutoff = .1, freq = 75, order=2):
        """Low-pass butter filter

        Parameters
        ----------
        cutoff: float
           Filter cutoff frequency.
        freq: int 
           Frequency of data acquisition.
        order: int
           Polynomials filter order

        Returns
        -------
        y: ndarray
           Filtered data.

        See Also
        --------
        Developed by T.F Almeida in 25/03/2021
        
        For more information see:
        https://github.com/tuliofalmeida/pyjama
        """
        b, a = butter_lowpass(cutoff, freq, order=order)
        y = scipy.signal.lfilter(b, a, data)
        return y

    def low_pass_filter(data, T = 0.1):
        """Low-pass filter, T is the itensity of the
        filter. He must be implemented thinking in 1/freq
        and the analysis is visual,the closer to 0 the more 
        filtered the data will be.

        Parameters
        ----------
        data: ndarray
           Input data.

        T: float
           Filter intensity

        Returns
        -------
        out: ndarray
           Filtered data.

        See Also
        --------
        Developed by T.F Almeida in 25/03/2021
        
        For more information see:
        https://github.com/tuliofalmeida/pyjama
        """
        out = []
        for i in range(len(data)):
            if i == 0:
                out.append(data[i])
            else:
                out.append(out[i-1] + T*(data[i] - out[i-1]))

        return np.asarray(out)  

    def complementaryFilter(roll, pitch, yaw, gyroX, gyroY, gyroZ, alpha = .05,dt = 1/75):
        """Complementary fiter for data fusion

        Parameters
        ----------
        roll: ndarray
           Roll data (XYZ).
        pitch: ndarray
           Pitch data (XYZ).
        yaw: ndarray
           Yaw data (XYZ).
        gyroX: ndarray
           Gyroscope X data in rad/s.
        gyroZ: ndarray
           Gyroscope Y data in rad/s.
        yaw: ndarray
           Gyroscope Z data in rad/s.
        alpha: float
           Accelerometer contribution
        dt: float
           Sample time

        Returns
        -------
        out: ndarray
           Filtered data.

        See Also
        --------
        Developed by T.F Almeida in 25/03/2021
        
        For more information see:
        https://github.com/tuliofalmeida/pyjama
        """
        angle_x = roll[0]
        angle_y = pitch[0]
        angle_z = yaw[0]
        complementarF = list()

        for i in range(len(gyroX)):
            angle_x = (1-alpha)*(angle_x + gyroX[i] * dt) + (alpha)*(roll[i])
            angle_y = (1-alpha)*(angle_y + gyroY[i] * dt) + (alpha)*(pitch[i])
            angle_z = (1-alpha)*(angle_z + gyroZ[i] * dt) + (alpha)*(yaw[i])
            complementarF.append((angle_x,angle_y,angle_z))
        
        return np.asarray(complementarF)

    def ComplementaryFilterGNUpdate(self,acc,gyr,mag,dt,alpha=.01,beta=.01, conj = True):
        """Filters data in real time. Implemented 
           using the Gauss-Newton optimizer. 

        Parameters
        ----------
        acc: ndarray
           Accelerometer array with XYZ in rad/s
        gyr: ndarray
           Gyroscope array with XYZ in rad/s
        mag: ndarray 
           Magnetometer array with XYZ in mG
        dt: float
           Sample time.
        alpha: float
           Accelerometer data contribution to angle.
        beta: float
           Factor to improve the effectiveness of 
           integrating the accelerometer with gyroscope.
           Must be determined between 0 and 1.
        conj: bool
           Determine if the quaternion resulted will be
           conjugated or not.

        Returns
        -------
        Angles: ndarray
           Fusioned data using the complementery filter.

        See Also
        --------
        Developed by Comotti & Ermidoro in 31/03/2017
        Adapted for python T.F Almeida in 25/03/2021
        
        For more information see:
        https://github.com/danicomo/9dof-orientation-estimation
        https://github.com/tuliofalmeida/pyjama
        """        
        AccF = acc/np.linalg.norm(acc)
        MagnF = mag/np.linalg.norm(mag)
        qOsserv = DataProcessing.GaussNewton(self.qFilt_1,AccF,MagnF)
        qOsserv = beta*qOsserv/np.linalg.norm(qOsserv)
        if self.i <= (self.accF_Length+6):
            # Filtered values initialized with observed values
            qGyroFilt = qOsserv
            qFilt = qOsserv
            qGyro = qOsserv
        else:
            # Complementary Filter
            dq = 0.5*(DataProcessing.QuaternionProduct(self.qFilt_1,np.asarray([0,gyr[0],gyr[1],gyr[2]]).T))
            qGyro = self.qGyro_1 + 0.5*(DataProcessing.QuaternionProduct(self.qGyro_1,np.asarray([0,gyr[0],gyr[1],gyr[2]]).T))*dt
            qGyro = qGyro/np.linalg.norm(qGyro)
            qGyroFilt = self.qFilt_1+dq*dt
            qGyroFilt = qGyroFilt/np.linalg.norm(qGyroFilt)
            
            qFilt = qGyroFilt*(1-alpha)+qOsserv*alpha
            qFilt = qFilt/np.linalg.norm(qFilt)

        if conj == True:    
            Angles = DataProcessing.GetAnglesFromQuaternion(DataProcessing.quaternConj(qFilt))
        else:
            Angles = DataProcessing.GetAnglesFromQuaternion(qFilt)

        self.qFilt_1 = qFilt
        self.qGyro_1 = qGyro
        return Angles

    def ComplementaryFilterGN(acc,gyr,mag,dt,alpha=.01,beta=.01,conj=True,low_pass=None):
        """Filters data offline. Implemented 
           using the Gauss-Newton optimizer. 

        Parameters
        ----------
        acc: ndarray
           Accelerometer array with XYZ in rad/s.
        gyr: ndarray
           Gyroscope array with XYZ in rad/s.
        mag: ndarray 
           Magnetometer array with XYZ in mG.
        dt: float
           Sample time.
        alpha: float
           Accelerometer data contribution to angle.
        beta: float
           Factor to improve the effectiveness of 
           integrating the accelerometer with gyroscope.
           Must be determined between 0 and 1.
        conj: bool
           Determine if the quaternion resulted will be
           conjugated or not.
        low_pass: None
            If this value is different from "None" the 
            Accelerometer and Magnetometer data will be 
            filtered using a low pass filter (function 
            "low_pass_filter") with the intensity being 
            equal to the value passed in low_pass.

        Returns
        -------
        Angles: ndarray
           Fusioned data using the complementery filter.

        See Also
        --------
        Developed by Comotti & Ermidoro in 31/03/2017
        Adapted for python T.F Almeida in 25/03/2021
        
        For more information see:
        https://github.com/danicomo/9dof-orientation-estimation
        https://github.com/tuliofalmeida/pyjama
        """ 
        CF = DataProcessing()
        acqSize = acc.shape[0]
        Angles = np.zeros((3,acqSize))

        if type(low_pass) != type(None):
            acc = DataProcessing.low_pass_filter(acc,low_pass)
            mag = DataProcessing.low_pass_filter(mag,low_pass)

        CF.i = 1
        while (CF.i < acqSize):
            Angles[:,CF.i] = CF.ComplementaryFilterGNUpdate(acc[CF.i,:],gyr[CF.i,:],mag[CF.i,:],dt,alpha,beta, conj)
            CF.i += 1

        return np.asarray(Angles).T

    def ComplementaryFilterGDUpdate(self,acc,gyr,mag,dt,alpha=.01,beta=.5, conj = True):
        """Filters data in real time. Implemented 
           using the Gradient Descendent optimizer. 

        Parameters
        ----------
        acc: ndarray
           Accelerometer array with XYZ in rad/s
        gyr: ndarray
           Gyroscope array with XYZ in rad/s
        mag: ndarray 
           Magnetometer array with XYZ in mG
        dt: float
           Sample time
        alpha: float
           Accelerometer data contribution to angle.
        beta: float
           Factor to improve the effectiveness of 
           integrating the accelerometer with gyroscope.
           Must be determined between 0 and 1.
        conj: bool
           Determine if the quaternion resulted will be
           conjugated or not.

        Returns
        -------
        Angles: ndarray
           Fusioned data using the complementery filter.

        See Also
        --------
        Developed by Comotti & Ermidoro in 31/03/2017
        Adapted for python T.F Almeida in 25/03/2021
        
        For more information see:
        https://github.com/danicomo/9dof-orientation-estimation
        https://github.com/tuliofalmeida/pyjama
        """ 
        AccF = acc/np.linalg.norm(acc)
        MagnF = mag/np.linalg.norm(mag)

        dq = 0.5*(DataProcessing.QuaternionProduct(self.qFilt_1,np.asarray([0,gyr[0],gyr[1],gyr[2]]).T))
        dqnorm = np.linalg.norm(dq)
        mu = 10*dqnorm*dt
        qOsserv = DataProcessing.GradientDescent(AccF,MagnF,self.qOsserv_1,mu)
        qOsserv = beta*qOsserv/np.linalg.norm(qOsserv)
        
        if (self.i <= self.accF_Length+9):
            qGyroFilt = qOsserv
            qFilt = qOsserv
        else:
            qGyroFilt = self.qFilt_1+dq*dt
            qGyroFilt = qGyroFilt/np.linalg.norm(qGyroFilt)

            dqnorm = np.linalg.norm(dq)
            mu = 10*dqnorm*dt
            qOsserv = DataProcessing.GradientDescent(AccF,MagnF,self.qOsserv_1,mu)
            qOsserv = beta*qOsserv/np.linalg.norm(qOsserv)
            
            qFilt = qGyroFilt*(1-alpha)+qOsserv*alpha
            qFilt = qFilt/np.linalg.norm(qFilt)

        if conj == True:    
            Angles = DataProcessing.GetAnglesFromQuaternion(DataProcessing.quaternConj(qFilt))
        else:
            Angles = DataProcessing.GetAnglesFromQuaternion(qFilt)

        self.qFilt_1 = qFilt
        self.qOsserv_1 = qOsserv
        return Angles



    def ComplementaryFilterGD(acc,gyr,mag,dt,alpha=.01,beta=.5,conj=True,low_pass=None):
        """Filters data offline. Implemented 
           using the Gradient Descendent optimizer. 

        Parameters
        ----------
        acc: ndarray
           Accelerometer array with XYZ in rad/s.
        gyr: ndarray
           Gyroscope array with XYZ in rad/s.
        mag: ndarray 
           Magnetometer array with XYZ in mG.
        dt: float
           Sample time.
        alpha: float
           Accelerometer data contribution to angle.
        beta: float
           Factor to improve the effectiveness of 
           integrating the accelerometer with gyroscope.
           Must be determined between 0 and 1.
        conj: bool
           Determine if the quaternion resulted will be
           conjugated or not.
        low_pass: None
            If this value is different from "None" the 
            Accelerometer and Magnetometer data will be 
            filtered using a low pass filter (function 
            "low_pass_filter") with the intensity being 
            equal to the value passed in low_pass.

        Returns
        -------
        Angles: ndarray
           Fusioned data using the complementery filter.

        See Also
        --------
        Developed by Comotti & Ermidoro in 31/03/2017
        Adapted for python T.F Almeida in 25/03/2021
        
        For more information see:
        https://github.com/danicomo/9dof-orientation-estimation
        https://github.com/tuliofalmeida/pyjama
        """ 
        CF = DataProcessing()
        acqSize = acc.shape[0]
        Angles = np.zeros((3,acqSize))

        if type(low_pass) != type(None):
            acc = DataProcessing.low_pass_filter(acc,low_pass)
            mag = DataProcessing.low_pass_filter(mag,low_pass)

        CF.i = 1
        while (CF.i < acqSize):
            Angles[:,CF.i] = CF.ComplementaryFilterGDUpdate(acc[CF.i,:],gyr[CF.i,:],mag[CF.i,:],dt,alpha,beta, conj)
            CF.i += 1

        return np.asarray(Angles).T

    def KalmanGD(acc,gyr,mag,gyrcalib=None,dt=1/75,R_In=[0.01,0.01,0.01,0.01],beta=.05,conj=True,low_pass=None):
        """Filters data offline. Kalman filter
        using the Gradient Descendent optimizer. 

        Parameters
        ----------
        acc: ndarray
           Accelerometer array with XYZ in rad/s.
        gyr: ndarray
           Gyroscope array with XYZ in rad/s.
        mag: ndarray 
           Magnetometer array with XYZ in mG.
        gyrcalib: ndarray
           Frist 5 seconds of static gyroscopedata 
           data in a array with XYZ in rad/s. 
        dt: float
           Sample time.
        R_In: tuple
           Sigma used to compute rotation matrix.          
        beta: float
           Factor to improve the effectiveness of 
           integrating the accelerometer with gyroscope.
           Must be determined between 0 and 1.
        conj: bool
           Determine if the quaternion resulted will be
           conjugated or not.
        low_pass: None
            If this value is different from "None" the 
            Accelerometer and Magnetometer data will be 
            filtered using a low pass filter (function 
            "low_pass_filter") with the intensity being 
            equal to the value passed in low_pass.

        Returns
        -------
        Angles: ndarray
           Fusioned data using the kalman filter.

        See Also
        --------
        Developed by Comotti & Ermidoro in 31/03/2017
        Adapted for python T.F Almeida in 25/03/2021
        
        For more information see:
        https://github.com/danicomo/9dof-orientation-estimation
        https://github.com/tuliofalmeida/pyjama
        """         
        acqSize = acc.shape[0]
                
        # Variance 
        if type(gyrcalib) == type(None):
            _,varG = DataProcessing.varianceEstimation(gyr)
            var = np.asarray([varG[0]**2,varG[1]**2,varG[2]**2]).T
        else:
            _,varG = DataProcessing.varianceEstimation(gyrcalib)
            var = np.asarray([varG[0]**2,varG[1]**2,varG[2]**2]).T            

        if type(low_pass) != type(None):
            acc = DataProcessing.low_pass_filter(acc,low_pass)
            mag = DataProcessing.low_pass_filter(mag,low_pass)
            
        # Acquisition variables
        mu       = np.zeros((1,acqSize))
        dq       = np.zeros((4,acqSize))
        AccF     = np.zeros((3,acqSize))
        MagnF    = np.zeros((3,acqSize))
        dqnorm   = np.zeros((3,acqSize))
        Angles   = np.zeros((3,acqSize))
        GyroRate = np.zeros((3,acqSize))
        

        qUpdate = np.zeros((4,acqSize))

        # Initial quaternion values
        qUpdate[:,0] = np.asarray([1,0,0,0]).T

        # Observation vector
        qOsserv = np.zeros((4,acqSize))
        qOsserv[:,0] = np.asarray([1,0,0,0]).T

        # Kalman Matrixes
        Q1 = [var[0]+var[1]+var[2] ,-var[0]+var[1]-var[2], -var[0]-var[1]+var[2], var[0]-var[1]-var[2]]
        Q2 = [-var[0]+var[1]-var[2],var[0]+var[1]+var[2] ,var[0]-var[1]-var[2]  ,-var[0]-var[1]+var[2]]
        Q3 = [-var[0]-var[1]+var[2],var[0]-var[1]-var[2] ,var[0]+var[1]+var[2]  ,-var[0]+var[1]-var[2]]
        Q4 = [var[0]-var[1]-var[2] ,-var[0]+var[1]-var[2],-var[0]+var[1]-var[2] ,var[0]+var[1]+var[2 ]]
        Qmatrix = np.asarray([Q1,Q2,Q3,Q4])

        H = np.eye(4,4)

        sigmaR = np.asarray(R_In).T
        R = [[sigmaR[0], 0, 0, 0],
            [0, sigmaR[1], 0, 0],
            [0, 0, sigmaR[2], 0],
            [0, 0, 0, sigmaR[3]]]

        qPredicted = np.zeros((4,acqSize))
        qPredicted[0][0] = 1
        qPredicted[1][0] = 0
        qPredicted[2][0] = 0
        qPredicted[3][0] = 0
        
        P_Update = np.eye(4,4)*2

        i = 1
        while (i < acqSize):
            
            GyroRate[0,i] = (gyr[i,0]+gyr[i-1,0])/2
            GyroRate[1,i] = (gyr[i,1]+gyr[i-1,1])/2
            GyroRate[2,i] = (gyr[i,2]+gyr[i-1,2])/2
            
            # Normalization and filtering
            AccF[:,i] = acc[i,:]/np.linalg.norm(acc[i,:])
            MagnF[:,i] = mag[i,:]/np.linalg.norm(mag[i,:])
            
            # Observation Computing
            # Gradient  step 
            G = np.asarray([0, GyroRate[0,i],GyroRate[1,i],GyroRate[0,i]]).T
            dq[:,i] = beta*(DataProcessing.QuaternionProduct(qUpdate[:,i-1],G))
            dqnorm[:,i] = np.linalg.norm(dq[:,i])
            mu[0,i] = 10*dqnorm[0,i]*dt
            qOsserv[:,i] = DataProcessing.GradientDescent(AccF[:,i],MagnF[:,i],qOsserv[:,i-1],mu[0,i])
            qOsserv[:,i] = beta*qOsserv[:,i]/np.linalg.norm(qOsserv[:,i])
            # End Observation Computing
            
            # Kalman  
            const = dt/2

            # F matrix computing
            F1 = [1                   , -const*GyroRate[0,i],-const*GyroRate[1,i], -const*GyroRate[2,i]]
            F2 = [const*GyroRate[0,i] ,1                    ,const*GyroRate[2,i] , -const*GyroRate[1,i]]
            F3 = [const*GyroRate[1,i] ,-const*GyroRate[2,i] ,1                   ,const*GyroRate[0,i]  ]
            F4 = [-const*GyroRate[2,i], const*GyroRate[1,i] ,-const*GyroRate[0,i], 1                   ]
            F = np.asarray([F1,F2,F3,F4])

            qPredicted[:,i] = F@qUpdate[:,i-1]
            Q = Qmatrix
            
            P_Predicted = F@P_Update@F.T+Q
            
            K = P_Predicted@H.T @np.linalg.inv((H@P_Predicted@H.T+R))
            qUpdate[:,i] = qPredicted[:,i]+K@(qOsserv[:,i]-H@qPredicted[:,i])
            qUpdate[:,i] = qUpdate[:,i]/np.linalg.norm(qUpdate[:,i])
            
            P_Update = (np.eye(4,4)-K*H)@P_Predicted

            if conj == True:
                Angles[:,i] = DataProcessing.GetAnglesFromQuaternion(DataProcessing.quaternConj(qUpdate[:,i]))
            else:
                Angles[:,i] = DataProcessing.GetAnglesFromQuaternion(qUpdate[:,i])
            i += 1

        return np.asarray(Angles).T

    def KalmanGN(acc,gyr,mag,gyrcalib=None,dt=1/75,R_In=[0.01,0.01,0.01,0.01],beta=.05,conj=True,low_pass=None):
        """Filters data offline. Kalman filter
        using the Gauss-Newton optimizer. 

        Parameters
        ----------
        acc: ndarray
           Accelerometer array with XYZ in rad/s.
        gyr: ndarray
           Gyroscope array with XYZ in rad/s.
        mag: ndarray 
           Magnetometer array with XYZ in mG.
        gyrcalib: ndarray
           Frist 5 seconds of static gyroscopedata 
           data in a array with XYZ in rad/s.        
        dt: float
           Sample time.
        R_In: tuple
           Sigma used to compute rotation matrix.   
        beta: float
           Factor to improve the effectiveness of 
           integrating the accelerometer with gyroscope.
           Must be determined between 0 and 1.
        conj: bool
           Determine if the quaternion resulted will be
           conjugated or not.
        low_pass: None
            If this value is different from "None" the 
            Accelerometer and Magnetometer data will be 
            filtered using a low pass filter (function 
            "low_pass_filter") with the intensity being 
            equal to the value passed in low_pass.
            
        Returns
        -------
        Angles: ndarray
           Fusioned data using the kalman filter.

        See Also
        --------
        Developed by Comotti & Ermidoro in 31/03/2017
        Adapted for python T.F Almeida in 25/03/2021
        
        For more information see:
        https://github.com/danicomo/9dof-orientation-estimation
        https://github.com/tuliofalmeida/pyjama
        """                 
        acqSize = acc.shape[0]

        # Variance 
        if type(gyrcalib) == type(None):
            _,varG = DataProcessing.varianceEstimation(gyr)
            var = np.asarray([varG[0]**2,varG[1]**2,varG[2]**2]).T
        else:
            _,varG = DataProcessing.varianceEstimation(gyrcalib)
            var = np.asarray([varG[0]**2,varG[1]**2,varG[2]**2]).T 

        if type(low_pass) != type(None):
            acc = DataProcessing.low_pass_filter(acc,low_pass)
            mag = DataProcessing.low_pass_filter(mag,low_pass)

        # Acquisition variables
        AccF     = np.zeros((3,acqSize))
        MagnF    = np.zeros((3,acqSize))
        dqnorm   = np.zeros((3,acqSize))
        Angles   = np.zeros((3,acqSize))
        GyroRate = np.zeros((3,acqSize))
        

        qUpdate = np.zeros((4,acqSize))

        # Initial quaternion values
        qUpdate[:,0] = np.asarray([1,0,0,0]).T

        # Observation vector
        qOsserv = np.zeros((4,acqSize))
        qOsserv[:,0] = np.asarray([1,0,0,0]).T

        # Kalman Matrixes
        Q1 = [var[0]+var[1]+var[2] ,-var[0]+var[1]-var[2], -var[0]-var[1]+var[2], var[0]-var[1]-var[2]]
        Q2 = [-var[0]+var[1]-var[2],var[0]+var[1]+var[2] ,var[0]-var[1]-var[2]  ,-var[0]-var[1]+var[2]]
        Q3 = [-var[0]-var[1]+var[2],var[0]-var[1]-var[2] ,var[0]+var[1]+var[2]  ,-var[0]+var[1]-var[2]]
        Q4 = [var[0]-var[1]-var[2] ,-var[0]+var[1]-var[2],-var[0]+var[1]-var[2] ,var[0]+var[1]+var[2 ]]
        Qmatrix = np.asarray([Q1,Q2,Q3,Q4])

        H = np.eye(4,4)

        sigmaR = np.asarray(R_In).T
        R = [[sigmaR[0], 0, 0, 0],
            [0, sigmaR[1], 0, 0],
            [0, 0, sigmaR[2], 0],
            [0, 0, 0, sigmaR[3]]]

        qPredicted = np.zeros((4,acqSize))
        qPredicted[0][0] = 1
        qPredicted[1][0] = 0
        qPredicted[2][0] = 0
        qPredicted[3][0] = 0
        
        P_Update = np.eye(4,4)*2

        i = 1
        while (i < acqSize):
            
            GyroRate[0,i] = (gyr[i,0]+gyr[i-1,0])/2
            GyroRate[1,i] = (gyr[i,1]+gyr[i-1,1])/2
            GyroRate[2,i] = (gyr[i,2]+gyr[i-1,2])/2
            
            # Normalization and filtering
            AccF[:,i] = acc[i,:]/np.linalg.norm(acc[i,:])
            MagnF[:,i] = mag[i,:]/np.linalg.norm(mag[i,:])
            
            
            # Observation Computing
            # Gauss Newton step 
            
            qOsserv[:,i] = DataProcessing.GaussNewton(qOsserv[:,i-1],AccF[:,i],MagnF[:,i])
            qOsserv[:,i] = beta*qOsserv[:,i]/np.linalg.norm(qOsserv[:,i])
            
            # End Observation Computing
            
            # Kalman
            const = dt/2

            # F matrix computing
            F1 = [1                   , -const*GyroRate[0,i],-const*GyroRate[1,i], -const*GyroRate[2,i]]
            F2 = [const*GyroRate[0,i] ,1                    ,const*GyroRate[2,i] , -const*GyroRate[1,i]]
            F3 = [const*GyroRate[1,i] ,-const*GyroRate[2,i] ,1                   ,const*GyroRate[0,i]  ]
            F4 = [-const*GyroRate[2,i], const*GyroRate[1,i] ,-const*GyroRate[0,i], 1                   ]
            F = np.asarray([F1,F2,F3,F4])

            qPredicted[:,i] = F@qUpdate[:,i-1]
            Q = Qmatrix
            
            P_Predicted = F@P_Update@F.T+Q
            
            K = P_Predicted@H.T @np.linalg.inv((H@P_Predicted@H.T+R))
            qUpdate[:,i] = qPredicted[:,i]+K@(qOsserv[:,i]-H@qPredicted[:,i])
            qUpdate[:,i] = qUpdate[:,i]/np.linalg.norm(qUpdate[:,i])
            
            P_Update = (np.eye(4,4)-K*H)@P_Predicted

            if conj == True:
                Angles[:,i] = DataProcessing.GetAnglesFromQuaternion(DataProcessing.quaternConj(qUpdate[:,i]))
            else:
                Angles[:,i] = DataProcessing.GetAnglesFromQuaternion(qUpdate[:,i])
            i += 1

        return np.asarray(Angles).T

    def Madgwick9DOFUpdate(self,Gyroscope, Accelerometer, Magnetometer, SamplePeriod = 1/75, Beta = 1):
        """Filters data in real time. 9 degrees of
          freedom MadgwickAHRS filter. 

        Parameters
        ----------
        Gyroscope: ndarray
           Gyroscope array with XYZ in rad/s.
        Accelerometer: ndarray
           Accelerometer array with XYZ in rad/s.
        Magnetometer: ndarray 
           Magnetometer array with XYZ in mG.
        SamplePeriod: float
           Sample time.
        Beta: float
           There is a tradeoff in the beta parameter 
           between accuracy and response speed. Must
           be determined according with the Gyroscope
           error.

        Returns
        -------
        Quaternion: ndarray
           Fusioned data using the Madgwick filter
           in radians/s.

        See Also
        --------
        Developed by S.O.H Madgwick in 27/09/2011
        Adapted for python T.F Almeida in 25/03/2021
        
        For more information see:
        Test scripts
        http://www.x-io.co.uk/node/8#quaternions
        https://github.com/tuliofalmeida/pyjama
        """     
        q = self.Quaternion # short name local variable for readability

        # Normalise accelerometer measurement
        if (np.linalg.norm(Accelerometer) == 0): 
            return 
        Accelerometer = Accelerometer/np.linalg.norm(Accelerometer)	# normalise accelerometer

        # Normalise magnetometer measurement
        if (np.linalg.norm(Magnetometer) == 0):
            return	# handle NaN
        Magnetometer = Magnetometer/np.linalg.norm(Magnetometer)	# normalise magnitude

        M = [0,Magnetometer[0],Magnetometer[1],Magnetometer[2]]
        # Reference direction of Earth's magnetic feild
        h = DataProcessing.quaternProd(q, DataProcessing.quaternProd(M,DataProcessing.quaternConj(q)))
        b = [0,np.linalg.norm([h[1],h[2]]),0,h[3]]

        # Gradient decent algorithm corrective step
        if np.size(np.asarray(q).shape) == 2:
            q = np.asarray(q[0])
        F = [[2*(q[1]*q[3] - q[0]*q[2]) - Accelerometer[0]],
            [2*(q[0]*q[1] + q[2]*q[3]) - Accelerometer[1]],
            [2*(0.5 - q[1]**2 - q[2]**2) - Accelerometer[2]],
            [2*b[1]*(0.5 - q[2]**2 - q[3]**2) + 2*b[3]*(q[1]*q[3] - q[0]*q[2]) - Magnetometer[0]],
            [2*b[1]*(q[1]*q[2] - q[0]*q[3]) + 2*b[3]*(q[0]*q[1] + q[2]*q[3]) - Magnetometer[1]],
            [2*b[1]*(q[0]*q[2] + q[1]*q[3]) + 2*b[3]*(0.5 - q[1]**2 - q[2]**2) - Magnetometer[2]]]
            
        J = [[-2*q[2],2*q[3],-2*q[0],2*q[1]],
            [2*q[1],2*q[0],2*q[3],2*q[2]],
            [0,-4*q[1],-4*q[2],0],
            [-2*b[3]*q[2],2*b[3]*q[3],-4*b[1]*q[2]-2*b[3]*q[0],-4*b[1]*q[3]+2*b[3]*q[1]],
            [-2*b[1]*q[3]+2*b[3]*q[1],2*b[1]*q[2]+2*b[3]*q[0],2*b[1]*q[1]+2*b[3]*q[3],-2*b[1]*q[0]+2*b[3]*q[2]],
            [2*b[1]*q[2],    2*b[1]*q[3]-4*b[3]*q[1], 2*b[1]*q[0]-4*b[3]*q[2],        2*b[1]*q[1]]]

        F = np.asarray(F)
        J = np.asarray(J)
        step = (J.T@F)
        step = step/np.linalg.norm(step)	# normalise step magnitude
        G = [0,Gyroscope[0],Gyroscope[1],Gyroscope[2]]
        # Compute rate of change of quaternion
        qDot = 0.5 * DataProcessing.quaternProd(q, G) - Beta * step.T
        # Integrate to yield quaternion
        q = q + qDot * SamplePeriod
        self.Quaternion = q/np.linalg.norm(q) # normalise quaternion

        return self.Quaternion
        
    def Madgwick6DOFUpdate(self, Gyroscope, Accelerometer,SamplePeriod = 1/75, Beta = 1):
        """Filters data in real time. 6 degrees of
          freedom MadgwickAHRS filter. 

        Parameters
        ----------
        Gyroscope: ndarray
           Gyroscope array with XYZ in rad/s.
        Accelerometer: ndarray
           Accelerometer array with XYZ in rad/s.
        SamplePeriod: float
           Sample time.
        Beta: float
           There is a tradeoff in the beta parameter 
           between accuracy and response speed. Must
           be determined according with the Gyroscope
           error.

        Returns
        -------
        Quaternion: ndarray
           Fusioned data using the Madgwick filter.

        See Also
        --------
        Developed by S.O.H Madgwick in 27/09/2011
        Adapted for python T.F Almeida in 25/03/2021
        
        For more information see:
        Test scripts
        http://www.x-io.co.uk/node/8#quaternions
        https://github.com/tuliofalmeida/pyjama
        """
        q = self.Quaternion # short name local variable for readability

        # Normalise accelerometer measurement
        if (np.linalg.norm(Accelerometer) == 0):
            return	# handle NaN
        Accelerometer = Accelerometer/np.linalg.norm(Accelerometer)	# normalise magnitude

        # Gradient decent algorithm corrective step
        F = [[2*(q[1]*q[3] - q[0]*q[2]) - Accelerometer[0]],
            [2*(q[0]*q[1] + q[2]*q[3]) - Accelerometer[1]],
            [2*(0.5- q[1]**2- q[2]**2) - Accelerometer[2]]]
        J = [[-2*q[2],	2*q[3],    -2*q[0],	2*q[1]],
            [ 2*q[1],   2*q[0],     2*q[3],	2*q[2]],
            [ 0,       -4*q[1],    -4*q[2],	    0]]
        
        F = np.asarray(F)
        J = np.asarray(J)

        step = (J.T@F)
        step = step/np.linalg.norm(step)	# normalise step magnitude

        # Compute rate of change of quaternion
        qDot = 0.5 * quaternProd(q, [0,Gyroscope[0],Gyroscope[1],Gyroscope[2]]) - Beta * step.T
        # Integrate to yield quaternion
        q = q + qDot * SamplePeriod
        self.Quaternion = q/np.linalg.norm(q) # normalise quaternion

        return np.asarray(self.Quaternion)

    def MadgwickAHRS(acc,gyr,mag,freq,beta1=.9,beta2=.01,conj = True):
        """Filters data in real time. 9 degrees of
          freedom MadgwickAHRS filter. 

        Parameters
        ----------
        acc: ndarray
           Accelerometer array with XYZ in rad/s.
        gyr: ndarray
           Gyroscope array with XYZ in rad/s.
        mag: ndarray 
           Magnetometer array with XYZ in mG.
        freq: float
           Frequency of data acquisition.
        beta1: float
           There is a tradeoff in the beta parameter 
           between accuracy and response speed. Must
           be determined according with the Gyroscope
           error. Used for the first quaternion 
           orientation.
        beta2: float
           There is a tradeoff in the beta parameter 
           between accuracy and response speed. Must
           be determined according with the Gyroscope
           error. Used for the others orientations.
        conj: bool
           Determine if the quaternion resulted will be
           conjugated or not.           

        Returns
        -------
        Angles: ndarray
           Fusioned data using the Madgwick filter.

        See Also
        --------
        Developed by S.O.H Madgwick in 27/09/2011
        Adapted for python T.F Almeida in 25/03/2021
        
        For more information see:
        Test scripts
        http://www.x-io.co.uk/node/8#quaternions
        https://github.com/tuliofalmeida/pyjama
        """
        madgwick = []
        madgFilter = DataProcessing(Quaternion=np.asarray([[1,0,0,0]]))
        for i in range(len(gyr)):
            q = madgFilter.Madgwick9DOFUpdate(gyr[i],acc[i],mag[i], 1/freq, Beta = beta1)
            if conj:
                madgwick.append(DataProcessing.quatern2euler(DataProcessing.quaternConj(q[0])))
            else:
                madgwick.append(DataProcessing.quatern2euler(q[0]))
        madgwick = []
        for i in range(len(gyr)):
            q = madgFilter.Madgwick9DOFUpdate(gyr[i],acc[i],mag[i], 1/freq, Beta = beta2)
            if conj:
                madgwick.append(DataProcessing.quatern2euler(DataProcessing.quaternConj(q[0])))
            else:
                madgwick.append(DataProcessing.quatern2euler(q[0]))
                
        return np.asarray(madgwick)*180/math.pi

    def rsquared(x, y):
        """ Calculate the r² for two datas

        Parameters
        ----------
        x: ndarray
           Array of calculated angle.
        y: ndarray
           Array of estimated angle.  

        Returns
        -------
        r_value: float
           r²
        
        See Also
        --------
        Developed by T.F Almeida in 25/03/2021

        For more information see:
        https://github.com/tuliofalmeida/pyjama  """
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
        return r_value**2

    def varianceEstimation(data):
        """ Calculate the variance of the data

        Parameters
        ----------
        data: ndarray
           IMU XYZ data.

        Returns
        -------
        mean: float
           X mean,Y mean,Z mean
        std: float
           X std,Y std,Z std

        See Also
        --------
        Developed by T.F Almeida in 25/03/2021

        For more information see:
        https://github.com/tuliofalmeida/pyjama  """
        acqSize = data.shape[0]
        mean = np.asarray([np.mean(data[0,:]),np.mean(data[1,:]),np.mean(data[2,:])])
        std =  np.asarray([np.std(data[0,:]),np.std(data[1,:]),np.std(data[2,:])])

        return mean,std

    def pattern_extraction(jointAngle,time,threshold,cycle=2,df=True,plot=False,bias=0,
                           flip = False ,save=False, save_name='Data',fsize=30, 
                           title='Pattern Extraction', yaxis='Angle (°)',xaxis='Cycle (%)'):
        """ Find and extract patterns from the data. 
           Able to plot all overlapping pattern data 
           for easy understanding. This function was 
           developed specifically to extract gait patterns,
           but if the data has a similar pattern it 
           can be used. Em caso de dificuldade de identificar 
           o threshold é possível utilizar a função (threshold_ajust) 
           para visualisar o impacto dos diferentes thresholds

        Parameters
        ----------
        jointAngle: pandas dataframe
           Column of a pandas data frame with the 
           data of joint angle.
        time: pandas dataframe
           Column of a pandas data frame with the 
           time stamp.
        threshold: float
            Point at which the data moves between 
            movements. Example: flexion and extension.
        cycle: int
            Number of points to be considered a pattern.
        df: bool
            Determines whether the input data (jointAngle 
            and time) is on dataframe pandas. If you are 
            not using 'False',then the expected input data
            will be an array.
        plot: bool
            Determines if the function will return the plot.
        bias: int optional
            Value to compensate the cycle adjust.
        flip: bool
            Variable to compensate the cycle representation.       
        save: bool
            If true save a fig as pdf
        save_name: str
            The fig name.
        fsize: int
            Font size.
        title: str
            Title name.
        yaxis: str
            Y axis name.
        xaxis: str
            X axis name.

        Returns
        -------
        Time_Data_Array: ndarray
           Array with the first column being the time 
           vector and the second being the joint angle.
        rom: ndarray
           Array with data patterns in tuples.
           
        See Also
        --------
        Developed by T.F Almeida in 25/03/2021

        For more information see:
        https://github.com/tuliofalmeida/pyjama  """ 
        if df == True:
            jointAngle = pyjamalib.DataHandler.csvToFloat(jointAngle)
            time = pyjamalib.DataHandler.csvToFloat(time)
        if plot == True:
            plt.figure(figsize=(12,9))
        
        diff = pyjamalib.DataProcessing.low_pass_filter((jointAngle[1:] - jointAngle[:-1]))
        zero_crossings = np.where(np.diff(np.signbit(jointAngle-threshold)))[0]
        Time_Data_Array = []
        mCicloqtd = cycle
        rom = []

        for i in range(int(len(zero_crossings)/(mCicloqtd+1))-1): 
            ciclo = i *(mCicloqtd)+bias
            tempo = 100*((time[zero_crossings[0+ciclo]:zero_crossings[mCicloqtd+ciclo]] - time[zero_crossings[0+ciclo]])/(time[zero_crossings[mCicloqtd+ciclo]]- time[zero_crossings[0+ciclo]]))
            data = jointAngle[zero_crossings[0+ciclo]:zero_crossings[mCicloqtd+ciclo]]
            rom.append(data)

            for ç in range(len(tempo)):
                Time_Data_Array.append((tempo[ç],data[ç]));
    
            if plot == True:
                if flip == True:
                     plt.plot(tempo,pyjamalib.DataProcessing.low_pass_filter(np.flip(data)))                   
                else:
                    plt.plot(tempo,pyjamalib.DataProcessing.low_pass_filter(data))
        if plot == True:
            plt.title(title,fontsize = fsize);
            plt.ylabel(yaxis,fontsize = fsize);
            plt.xlabel(xaxis,fontsize = fsize);
            plt.yticks(fontsize = fsize)
            plt.xticks(fontsize = fsize)
            plt.grid();
            if save == True:
                plt.savefig(save_name + '.pdf')
            plt.show();
        
        return np.asarray(Time_Data_Array),np.asarray(rom)

    def patternCI(all_x, all_y,poly_degree = 1,CI = 1.96,df = True,plot=False,
                  figuresize=(12,9),title='Trajectory CI-95',x_label='Cycle (%)',
                  y_label='Angle (°)',label1='Data 1',label2='Data 2',label3='CI',
                  save=False, save_name='Data', fsize=30, lsize=20):
        """Calculates the confidence interval of the 
           patterns extracted by the 'pattern_extraction' 
           function.

        Parameters
        ----------
        all_x: ndarray
           Time data.
        all_y: ndarray
           Joint angle data.
        poly_degree: int
           Degree of the polynomial to fit the data curve.
        CI: float
           Reference value for calculating the 95% 
           confidence interval.
        df: bool
            Determines whether the input data (jointAngle 
            and time) is on dataframe pandas. If you are 
            not using 'False',then the expected input data
            will be an array.
        plot: bool
            Determines if the function will return the plot.
        figuresize: tuple optional
           Determine the size of the figure.
        title: string optional
           Determine the title of the plot.
        x_label: string optional
           Determine the label of the X axis.
        y_label: string optional
           Determine the label of the Y axis.
        label1: str
           Label for data 1.
        label2: str
           Label for data 2.   
        label3: str
           Label for CI.           
        save: bool
            If true save a fig as pdf
        save_name: str
            The fig name.
        fsize: int
            Font size.
        lsize: int
            Labels font size.              
        Returns
        -------
        statistics: ndarray
           An array containing the confidence 
           interval[0], standard deviation[1],
           Polynomial coefficient[2], r²[3], 
           estimated minimum[4], estimated maximum[5] 
           and data variance[6].
       
        See Also
        --------
        Developed by T.F Almeida in 25/03/2021

        For more information see:
        https://github.com/tuliofalmeida/pyjama  """             
        if df == True:
            all_x = pyjamalib.DataHandler.csvToFloat(all_x)
            all_y = pyjamalib.DataHandler.csvToFloat(all_y)

        coef = np.polyfit(all_x,all_y,poly_degree)
        yest = np.polyval(coef,all_x)

        stdev = np.sqrt(sum((yest - all_y)**2)/(len(all_y)-2))
        confidence_interval = CI*stdev

        r2 = DataProcessing.rsquared(all_y,yest)

        yup = []
        ydown = []
        for i in range(len(all_y)):
            yup.append(yest[i] + confidence_interval)
            ydown.append(yest[i] - confidence_interval)

        xgeralind = np.argsort(all_x)
        yupsorted = []
        ydownsorted = []
        for i in range(len(yup)):
            ydownsorted.append(ydown[xgeralind[i]])
            yupsorted.append(yup[xgeralind[i]])

        if plot == True:
            plt.figure(figsize=figuresize);
            plt.plot(all_x,all_y,'.',color= 'black');
            plt.plot(all_x,yest,'.',color= 'red');
            plt.fill_between(DataProcessing.low_pass_filter(np.sort(all_x)),yupsorted, ydownsorted,color='r',alpha=.1);
            plt.title(title).set_size(fsize);
            plt.xlabel(x_label).set_size(fsize);
            plt.ylabel(y_label).set_size(fsize);
            plt.yticks(fontsize = fsize)
            plt.xticks(fontsize = fsize)
            plt.legend([label1, label2, label3], loc='upper right',fontsize = lsize)
            plt.grid();
            if save == True:
                plt.savefig(save_name + '.pdf')
            plt.show();

        var = stdev**2
        statistics = confidence_interval,stdev,coef,r2,min(yest),max(yest),yest,var

        return statistics
    
    def data_comparison(Time_Data_Array,statistics_1,statistics_2, plot=True, 
                        label1='Data 1',label2='Data 2', save=False,
                        save_name = 'Data', fsize = 30, lsize = 20,
                        title= 'Data Comparison', yaxis='Angle (°)', xaxis='Cycle (%)'):
         """This function is made for visual
         comparison of two data, using the data 
         average. Using the output of the 
         'patter_extraction' and 'patternCI' 
         function.

         Parameters
         ----------
         Time_Data_Array: ndarray
           First output from patter_extraction .
         statistics_1: ndarray
           Data 1 output from 'patternCI'.
         statistics_2: float
           Data 2 output from 'patternCI'.
         plot: bool optional
           Plot the comparsion graph.
         label1: str
           Label for data 1.
         label1: str
           Label for data 2.    
        save: bool
            If true save a fig as pdf
        save_name: str
            The fig name.
        fsize: int
            Font size.
        lsize: int
            Labels font size.    
        title: str
            Title name.
        yaxis: str
            Y axis name.
        xaxis: str
            X axis name.
                                   
         Returns
         -------
         plot: if True
               plots the data as a function 
               of time and marks the threshold
               points in the data
         poly1: ndarray
               Polynomial from data 1
         poly2: ndarray
               Polynomial from data 1
         
         See Also
         --------
         Developed by T.F Almeida in 25/03/2021

         For more information see:
         https://github.com/tuliofalmeida/pyjama"""           
         poly1 = np.polyval(statistics_1[2],Time_Data_Array[:,0])
         poly2 = np.polyval(statistics_2[2],Time_Data_Array[:,0])
         if plot:
               plt.figure(figsize=(12,9))
               plt.plot(Time_Data_Array[:,0],poly1,'.',color='black', markersize=12)
               plt.plot(Time_Data_Array[:,0],poly2,'.',color='red', markersize=12)
               plt.legend([label1, label2], loc='upper right',fontsize = lsize)
               plt.title(title,fontsize = fsize);
               plt.ylabel(yaxis,fontsize = fsize);
               plt.xlabel(xaxis,fontsize = fsize);
               plt.yticks(fontsize = fsize);
               plt.xticks(fontsize = fsize);
               plt.grid();
               if save:
                    plt.savefig(save_name + '.pdf')    
               plt.show();

               return poly1,poly2;

         else:
               return poly1,poly2
            
    def threshold_ajust(data,time,threshold,x_adjust=0, df =True):  
        """This function plots the data 
        as a function of time and marks 
        the threshold points in the data 
        to facilitate decision making.

        Parameters
        ----------
        data: ndarray or dataframe
           Joint angle data.
        time: ndarray or dataframe
           Time data.
        threshold: float
            Point at which the data moves 
            between movements. Example: 
            flexion and extension.
       
        Returns
        -------
        plot: 
            plots the data as a function 
            of time and marks the threshold
            points in the data
       
        See Also
        --------
        Developed by T.F Almeida in 25/03/2021

        For more information see:
        https://github.com/tuliofalmeida/pyjama  """ 
        if df == True:
            data = pyjamalib.DataHandler.csvToFloat(data)
            time = pyjamalib.DataHandler.csvToFloat(time) 
        crossings = np.where(np.diff(np.signbit(data-threshold)))[0]
        plt.figure(figsize=(12,9))
        plt.plot(time,data)
        for i in range(len(crossings)):
            plt.plot(crossings[i]+x_adjust,data[crossings[i]], '.', color = 'red')
        plt.title('Threshold Adjust',fontsize = 30)
        plt.ylabel('Angle (°)',fontsize = 30)
        plt.xlabel('Time (bins)',fontsize = 30)
        plt.grid();
        plt.show();

    def range_of_motion(data, df = True):
        """Calculates the range of motion 
        data with some statistical metrics.

        Parameters
        ----------
        data: pandas dataframe
           Column with joint angle data.
        df: bool
           Determine if the data is pandas
           dataframe. If false, the data is 
           expected to be an ndarray.

        Returns
        -------
        metrics: ndarray
           An array containing the range 
           of motion (ROM) in each moment, 
           the average ROM, ROM standard 
           deviation, ROM variance and 
           ROM lenght.
       
        See Also
        --------
        Developed by T.F Almeida in 25/03/2021

        For more information see:
        https://github.com/tuliofalmeida/pyjama  """ 
        rom = []
        if df == True:
            data = pyjamalib.DataHandler.csvToFloat(data)
            rom = (max(data) - min(data))
            
            metrics = rom,np.mean(data),np.std(data),np.var(data),len(data)

            return metrics
        else:
            for i in range(len(data)):
                rom.append(max(data[i]) - min(data[i]))

            metrics = np.asarray(rom),np.mean(rom),np.std(rom),np.var(rom),len(rom)

            return metrics

    def rom_mean(data):
        """Calculates the average range of 
        motion of each pattern found and 
        then the average of all averagess.

        Parameters
        ----------
        data: pandas dataframe
           Column with joint angle data.

        Returns
        -------
        general_mean: ndarray
           The general average of ROM
       
        See Also
        --------
        Developed by T.F Almeida in 25/03/2021

        For more information see:
        https://github.com/tuliofalmeida/pyjama  """ 
        x=0
        for i in range(len(data)):
            x = x + max(data[i])-min(data[i])
        general_mean = x/len(data)
        return general_mean
    
    def mean_absolute_percentage_error(y_true, y_pred,sample_weight=None):  
        """Mean absolute percentage error regression loss.
        Note here that we do not represent the output as a percentage in range
        [0, 100]. Instead, we represent it in range [0, 1/eps].
        
        Parameters
        ----------
        y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Ground truth (correct) target values.
        y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Estimated target values.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.
            
        Returns
        -------
        loss : float or ndarray of floats in the range [0, 1/eps]
            MAPE output is non-negative floating point. The best value is 0.0.
            But note the fact that bad predictions can lead to arbitarily large
            MAPE values, especially if some y_true values are very close to zero.
            Note that we return a large value instead of `inf` when y_true is zero.
            
        Examples
        --------
        'from sklearn.metrics import mean_absolute_percentage_error
        y_true = [3, -0.5, 2, 7]
        y_pred = [2.5, 0.0, 2, 8]
        mean_absolute_percentage_error(y_true, y_pred)
        0.3273...
        y_true = [[0.5, 1], [-1, 1], [7, -6]]
        y_pred = [[0, 2], [-1, 2], [8, -5]]
        mean_absolute_percentage_error(y_true, y_pred)
        0.5515...
        mean_absolute_percentage_error(y_true, y_pred, multioutput=[0.3, 0.7])
        0.6198...'
        
        See Also
        --------
        Adapted by T.F Almeida in 25/03/2021
        Original code of scikit-learn v.0.24

        For more information see:
        https://github.com/scikit-learn/scikit-learn/blob/15a949460/sklearn/metrics/_regression.py#L197
        https://github.com/tuliofalmeida/pyjama 
        """          
        epsilon = np.finfo(np.float64).eps
        mape = np.abs(y_pred - y_true)/np.max(np.abs(y_true))
        output_errors = np.mean(mape)
            
        return output_errors*100