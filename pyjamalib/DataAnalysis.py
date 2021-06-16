import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pyjamalib
import scipy.signal,scipy.stats

class DataAnalysis:
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

    def all_filters(data,data_calib,low_pass=.1,freq=75,dt=1/75,alpha=.01,beta=.05,beta_mad=.9,beta_mad2=.01,
                    conj=True,gyr_filt_cf=.1,gyr_filt_k=.05,euler=True,cf=True,cf_gd=True,cf_gn=True,k_gd=True,
                    k_gn=True,mad=True):
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
            euler: bool
                If true it will apply Euler angles.
            cf: bool
                If true it will apply basic Complementary Filter.
            cf_gd: bool
                If true it will apply Complementary Filter with
                Gradient Descent.
            cf_gn: bool
                If true it will apply Complementary Filter with
                Gauss-Newton.
            k_gd: bool
                If true it will apply Kalman filter with Gradient
                Descent.
            k_gn: bool
                If true it will apply Kalman filter with Gauss-Newton.
            mad: bool 
                If true it will apply MadgwickAHRS.    

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
            accf = pyjamalib.DataProcessing.low_pass_filter(acc,low_pass)
            gyrf = pyjamalib.DataProcessing.low_pass_filter(gyr,low_pass)
            magf = pyjamalib.DataProcessing.low_pass_filter(mag,low_pass)

            df = pd.DataFrame({'Time':time[:]                                                            ,
                            'Acc_X':acc[:,0]         ,'Acc_Y': acc[:,1]         ,'Acc_Z': acc[:,2]       ,
                            'Gyr_X':gyr[:,0]         ,'Gyr_Y': gyr[:,1]         ,'Gyr_Z': gyr[:,2]       ,
                            'Mag_X':mag[:,0]         ,'Mag_Y': mag[:,1]         ,'Mag_Z': mag[:,2]       ,
                            'Acc_X_Filt':accf[:,0]   ,'Acc_Y_Filt':accf[:,1]    ,'Acc_Z_Filt':accf[:,2]  ,
                            'Gyr_X_Filt':gyrf[:,0]   ,'Gyr_Y_Filt':gyrf[:,1]    ,'Gyr_Z_Filt':gyrf[:,2]  ,
                            'Mag_X_Filt':magf[:,0]   ,'Mag_Y_Filt':magf[:,1]    ,'Mag_Z_Filt':magf[:,2] })

            acc_df   = pyjamalib.DataHandler.csvFloatMerge(df['Acc_X'],df['Acc_Y'],df['Acc_Z'])
            gyr_df   = pyjamalib.DataHandler.csvFloatMerge(df['Gyr_X'],df['Gyr_Y'],df['Gyr_Z'])
            mag_df   = pyjamalib.DataHandler.csvFloatMerge(df['Mag_X'],df['Mag_Y'],df['Mag_Z'])
            acc_df_f = pyjamalib.DataHandler.csvFloatMerge(df['Acc_X_Filt'],df['Acc_Y_Filt'],df['Acc_Z_Filt'])
            gyr_df_f = pyjamalib.DataHandler.csvFloatMerge(df['Gyr_X_Filt'],df['Gyr_Y_Filt'],df['Gyr_Z_Filt'])
            mag_df_f = pyjamalib.DataHandler.csvFloatMerge(df['Mag_X_Filt'],df['Mag_Y_Filt'],df['Mag_Z_Filt'])

            if euler:
                Roll, Pitch, Yaw = pyjamalib.DataProcessing.get_euler(q=[1,0,0,0],Acc=acc_df_f,Mag=mag_df_f,conj=conj)
                df['Roll'],df['Pitch'],df['Yaw'] = Roll, Pitch, Yaw 
                
            if cf:            
                CF    = pyjamalib.DataProcessing.complementaryFilter(Roll,Pitch,Yaw,
                                                                    pyjamalib.DataProcessing.low_pass_filter(gyr_df[:,0],gyr_filt_cf),
                                                                    pyjamalib.DataProcessing.low_pass_filter(gyr_df[:,1],gyr_filt_cf),
                                                                    pyjamalib.DataProcessing.low_pass_filter(gyr_df[:,2],gyr_filt_cf),
                                                                    alpha=alpha,dt=dt)   
                df['CF_Roll'],df['CF_Pitch'],df['CF_Yaw'] = CF[:,0],CF[:,1],CF[:,2]

            if cf_gd:
                CF_GD = pyjamalib.DataProcessing.ComplementaryFilterGD(acc_df, pyjamalib.DataProcessing.low_pass_filter(gyr_df,gyr_filt_cf),mag_df,
                                                                    dt=dt,alpha=alpha,beta=beta,conj=conj,low_pass=low_pass)
                df['CF_GD_Roll'],df['CF_GD_Pitch'],df['CF_GD_Yaw'] = CF_GD[:,0],CF_GD[:,1],CF_GD[:,2]

            if cf_gn:
                CF_GN = pyjamalib.DataProcessing.ComplementaryFilterGN(acc_df,pyjamalib.DataProcessing.low_pass_filter(gyr_df,gyr_filt_cf),mag_df,
                                                                    dt=dt,alpha=alpha,beta=beta,conj=conj,low_pass=low_pass)
                df['CF_GN_Roll'],df['CF_GN_Pitch'],df['CF_GN_Yaw'] = CF_GN[:,0],CF_GN[:,1],CF_GN[:,2]   

            if k_gd:     
                Kalman_GD = pyjamalib.DataProcessing.KalmanGD(acc_df,pyjamalib.DataProcessing.low_pass_filter(gyr_df,gyr_filt_k),mag_df,gyrcalib=kalamn_gyr,
                                                            dt=dt,beta=beta,conj=conj,low_pass=low_pass)
                df['Kalman_GD_Roll'],df['Kalman_GD_Pitch'],df['Kalman_GD_Yaw'] = Kalman_GD[:,0],Kalman_GD[:,1],Kalman_GD[:,2]

            if k_gn:          
                Kalman_GN = pyjamalib.DataProcessing.KalmanGN(acc_df,pyjamalib.DataProcessing.low_pass_filter(gyr_df,gyr_filt_k),mag_df,gyrcalib=kalamn_gyr,
                                                            dt=dt,beta=beta,conj=conj,low_pass=low_pass)
                df['Kalman_GN_Roll'],df['Kalman_GN_Pitch'],df['Kalman_GN_Yaw'] = Kalman_GN[:,0],Kalman_GN[:,1],Kalman_GN[:,2]
            if mad:
                Madgwick  = pyjamalib.DataProcessing.MadgwickAHRS(acc_df,gyr_df,mag_df,
                                                                freq=freq,beta1=beta_mad,beta2=beta_mad2)
                df['Madgwick_Roll'],df['Madgwick_Pitch'],df['Madgwick_Yaw'] = Madgwick[:,0],Madgwick[:,1],Madgwick[:,2]

            return df

    def joint_measures(df_first_joint,df_second_joint,patternRoll=False,patternPitch=False,patternYaw=False,
                       init=0,end=None,freq=75,threshold=None,cycle=2,bias=0,poly_degree=9,CI=1.96,absolute=True,
                       euler=True,cf=True,cf_gd=True,cf_gn=True,k_gd=True,k_gn=True,mad=True):
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
            euler: bool
                If true it will calculate Euler angles metrics.
            cf: bool
                If true it will calculate basic Complementary Filter
                metrics.
            cf_gd: bool
                If true it will calculate Complementary Filter with
                Gradient Descent metrics.
            cf_gn: bool
                If true it will calculate Complementary Filter with
                Gauss-Newton metrics.
            k_gd: bool
                If true it will calculate Kalman filter with Gradient
                Descent metrics.
            k_gn: bool
                If true it will calculate Kalman filter with Gauss-Newton
                metrics.
            mad: bool 
                If true it will calculate MadgwickAHRS metrics.   

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

            if patternRoll == False: 
                df = df.drop(['Flex/Ext','Flex/Ext_CF','Flex/Ext_CF_GD',
                            'Flex/Ext_CF_GN','Flex/Ext_Kalman_GD',
                            'Flex/Ext_Kalman_GN','Flex/Ext_Madgwick'], axis=1)                
            if patternPitch == False: 
                df = df.drop(['Adu/Abd','Adu/Abd_CF','Adu/Abd_CF_GD',
                            'Adu/Abd_CF_GN','Adu/Abd_Kalman_GD',
                            'Adu/Abd_Kalman_GN','Adu/Abd_Madgwick'], axis=1)   
            if patternYaw == False: 
                df = df.drop(['Int/Ext_Rot','Int/Ext_Rot_CF','Int/Ext_Rot_CF_GD',
                            'Int/Ext_Rot_CF_GN','Int/Ext_Rot_Kalman_GD',
                            'Int/Ext_Rot_Kalman_GN','Int/Ext_Rot_Madgwick'], axis=1)     

            if patternRoll:
                if euler == False:
                    df = df.drop(['Flex/Ext'], axis=1)                
                if cf == False:        
                    df = df.drop(['Flex/Ext_CF'], axis=1)
                if cf_gd == False:        
                    df = df.drop(['Flex/Ext_CF_GD'], axis=1)
                if cf_gn == False:        
                    df = df.drop(['Flex/Ext_CF_GN'], axis=1)
                if k_gd == False:        
                    df = df.drop(['Flex/Ext_Kalman_GD'], axis=1)
                if k_gn == False:        
                    df = df.drop(['Flex/Ext_Kalman_GN'], axis=1)
                if mad == False:        
                    df = df.drop(['Flex/Ext_Madgwick'], axis=1)
            if patternPitch:
                if euler == False:
                    df = df.drop(['Adu/Abd'], axis=1)                
                if cf == False:        
                    df = df.drop(['Adu/Abd_CF'], axis=1)
                if cf_gd == False:        
                    df = df.drop(['Adu/Abd_CF_GD'], axis=1)
                if cf_gn == False:        
                    df = df.drop(['Adu/Abd_CF_GN'], axis=1)
                if k_gd == False:        
                    df = df.drop(['Adu/Abd_Kalman_GD'], axis=1)
                if k_gn == False:        
                    df = df.drop(['Adu/Abd_Kalman_GN'], axis=1)
                if mad == False:        
                    df = df.drop(['Adu/Abd_Madgwick'], axis=1)
            if patternYaw:
                if euler == False:
                    df = df.drop(['Int/Ext_Rot'], axis=1)                
                if cf == False:        
                    df = df.drop(['Int/Ext_Rot_CF'], axis=1)
                if cf_gd == False:        
                    df = df.drop(['Int/Ext_Rot_CF_GD'], axis=1)
                if cf_gn == False:        
                    df = df.drop(['Int/Ext_Rot_CF_GN'], axis=1)
                if k_gd == False:        
                    df = df.drop(['Int/Ext_Rot_Kalman_GD'], axis=1)
                if k_gn == False:        
                    df = df.drop(['Int/Ext_Rot_Kalman_GN'], axis=1)
                if mad == False:        
                    df = df.drop(['Int/Ext_Rot_Madgwick'], axis=1)

            column_name = df.columns.tolist()[1:]
            row_names = df_first_joint.columns.tolist()[19:]
            cut_r = []
            cut_p = []
            cut_y = []
            for i in range(len(row_names)):
                if patternRoll == False:
                    if '_Roll' in row_names[i]:
                        cut_r.append(row_names[i])
                if patternPitch == False:
                    if '_Pitch' in row_names[i]:
                        cut_p.append(row_names[i])
                if patternYaw == False:
                    if '_Yaw' in row_names[i]:
                        cut_y.append(row_names[i])
            if len(cut_r) != 0:
                for i in range(len(cut_r)):
                    row_names.remove(cut_r[i])
            if len(cut_p) != 0:
                for i in range(len(cut_p)):
                    row_names.remove(cut_p[i])
            if len(cut_y) != 0:
                for i in range(len(cut_y)):
                    row_names.remove(cut_y[i])
                        
            for ç in zip(column_name,row_names):            
                    df[ç[0]] = 180-(df_first_joint[ç[1]]+df_second_joint[ç[1]])
                    if absolute:
                        df[ç[0]] = (df[ç[0]] + abs(min(df[ç[0]])))*-1
                        df[ç[0]] = df[ç[0]] + abs(min(df[ç[0]]))
                    
            thre = np.zeros((len(column_name),1))      
            if type(threshold) == type(None):
                for i in range(len(thre)):
                    thre[i] = np.mean(df[column_name[i]])
                thre_list = list(zip(thre,column_name))
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
                if euler:
                    pr = 0
                    for i in range(len(thre_list)):
                        if 'Flex/Ext' == thre_list[i][1]:
                            pr = thre_list[i][0]
                    rR,rawRrom = pyjamalib.DataProcessing.pattern_extraction(df['Flex/Ext'],df['Time'],threshold=pr, bias=bias, cycle=cycle);
                    rawRoll = pyjamalib.DataProcessing.patternCI(rR[:,0],rR[:,1],poly_degree=poly_degree,CI=CI, df=False);
                    rm_R = pyjamalib.DataProcessing.rom_mean(rawRrom)
                    index = np.concatenate((index,['Flex/Ext']))
                    Rom = np.concatenate((Rom,[max(df['Flex/Ext'])-min(df['Flex/Ext'])]))
                    Mean = np.concatenate((Mean,[rm_R])) 
                    Std = np.concatenate((Std,[rawRoll[1]])) 
                    CIlist = np.concatenate((CIlist,[rawRoll[0]]))
                    Var = np.concatenate((Var,[rawRoll[7]]))
                    Min = np.concatenate((Min,[min(df['Flex/Ext'])]))
                    Max = np.concatenate((Max,[max(df['Flex/Ext'])]))
                    MinEst = np.concatenate((MinEst,[rawRoll[4]]))
                    MaxEst = np.concatenate((MaxEst,[rawRoll[5]]))
                if cf:
                    pr = 0
                    for i in range(len(thre_list)):
                        if 'Flex/Ext_CF' == thre_list[i][1]:
                            pr = thre_list[i][0]
                    CFR,CFRrom=pyjamalib.DataProcessing.pattern_extraction(df['Flex/Ext_CF'],df['Time'],threshold=pr, bias=bias, cycle=cycle);
                    cfRoll=pyjamalib.DataProcessing.patternCI(CFR[:,0],CFR[:,1],poly_degree=poly_degree,CI=CI, df=False);
                    cf_R = pyjamalib.DataProcessing.rom_mean(CFRrom)
                    index = np.concatenate((index,['Flex/Ext_CF']))
                    Rom = np.concatenate((Rom,[max(df['Flex/Ext_CF'])-min(df['Flex/Ext_CF'])]))
                    Mean = np.concatenate((Mean,[cf_R])) 
                    Std = np.concatenate((Std,[cfRoll[1]])) 
                    CIlist = np.concatenate((CIlist,[cfRoll[0]]))
                    Var = np.concatenate((Var,[cfRoll[7]]))
                    Min = np.concatenate((Min,[min(df['Flex/Ext_CF'])]))
                    Max = np.concatenate((Max,[max(df['Flex/Ext_CF'])]))
                    MinEst = np.concatenate((MinEst,[cfRoll[4]]))
                    MaxEst = np.concatenate((MaxEst,[cfRoll[5]]))
                if cf_gd:
                    pr = 0
                    for i in range(len(thre_list)):
                        if 'Flex/Ext_CF_GD' == thre_list[i][1]:
                            pr = thre_list[i][0]
                    CFGDR,CFGDRrom=pyjamalib.DataProcessing.pattern_extraction(df['Flex/Ext_CF_GD'],df['Time'],threshold=pr, bias=bias, cycle=cycle);
                    cfgdRoll=pyjamalib.DataProcessing.patternCI(CFGDR[:,0],CFGDR[:,1],poly_degree=poly_degree,CI=CI, df=False);
                    cfgd_R= pyjamalib.DataProcessing.rom_mean(CFGDRrom)
                    index = np.concatenate((index,['Flex/Ext_CF_GD']))
                    Rom = np.concatenate((Rom,[max(df['Flex/Ext_CF_GD'])-min(df['Flex/Ext_CF_GD'])]))
                    Mean = np.concatenate((Mean,[cfgd_R])) 
                    Std = np.concatenate((Std,[cfgdRoll[1]])) 
                    CIlist = np.concatenate((CIlist,[cfgdRoll[0]]))
                    Var = np.concatenate((Var,[cfgdRoll[7]]))
                    Min = np.concatenate((Min,[min(df['Flex/Ext_CF_GD'])]))
                    Max = np.concatenate((Max,[max(df['Flex/Ext_CF_GD'])]))
                    MinEst = np.concatenate((MinEst,[cfgdRoll[4]]))
                    MaxEst = np.concatenate((MaxEst,[cfgdRoll[5]]))
                if cf_gn:
                    pr = 0
                    for i in range(len(thre_list)):
                        if 'Flex/Ext_CF_GN' == thre_list[i][1]:
                            pr = thre_list[i][0]
                    CFGNR,CFGNRrom=pyjamalib.DataProcessing.pattern_extraction(df['Flex/Ext_CF_GN'],df['Time'],threshold=pr, bias=bias, cycle=cycle);
                    cfgnRoll=pyjamalib.DataProcessing.patternCI(CFGNR[:,0],CFGNR[:,1],poly_degree=poly_degree,CI=CI, df=False);
                    cfgn_R = pyjamalib.DataProcessing.rom_mean(CFGNRrom)
                    index = np.concatenate((index,['Flex/Ext_CF_GN']))
                    Rom = np.concatenate((Rom,[max(df['Flex/Ext_CF_GN'])-min(df['Flex/Ext_CF_GN'])]))
                    Mean = np.concatenate((Mean,[cfgn_R])) 
                    Std = np.concatenate((Std,[cfgnRoll[1]])) 
                    CIlist = np.concatenate((CIlist,[cfgnRoll[0]]))
                    Var = np.concatenate((Var,[cfgnRoll[7]]))
                    Min = np.concatenate((Min,[min(df['Flex/Ext_CF_GN'])]))
                    Max = np.concatenate((Max,[max(df['Flex/Ext_CF_GN'])]))
                    MinEst = np.concatenate((MinEst,[cfgnRoll[4]]))
                    MaxEst = np.concatenate((MaxEst,[cfgnRoll[5]]))
                if k_gd:
                    pr = 0
                    for i in range(len(thre_list)):
                        if 'Flex/Ext_Kalman_GD' == thre_list[i][1]:
                            pr = thre_list[i][0]
                    KGDR,KGDRrom=pyjamalib.DataProcessing.pattern_extraction(df['Flex/Ext_Kalman_GD'],df['Time'],threshold=pr, bias=bias, cycle=cycle);
                    kgdRoll=pyjamalib.DataProcessing.patternCI(KGDR[:,0],KGDR[:,1],poly_degree=poly_degree,CI=CI, df=False);
                    kgd_R = pyjamalib.DataProcessing.rom_mean(KGDRrom)
                    index = np.concatenate((index,['Flex/Ext_Kalman_GD']))
                    Rom = np.concatenate((Rom,[max(df['Flex/Ext_Kalman_GD'])-min(df['Flex/Ext_Kalman_GD'])]))
                    Mean = np.concatenate((Mean,[kgd_R])) 
                    Std = np.concatenate((Std,[kgdRoll[1]])) 
                    CIlist = np.concatenate((CIlist,[kgdRoll[0]]))
                    Var = np.concatenate((Var,[kgdRoll[7]]))
                    Min = np.concatenate((Min,[min(df['Flex/Ext_Kalman_GD'])]))
                    Max = np.concatenate((Max,[max(df['Flex/Ext_Kalman_GD'])]))
                    MinEst = np.concatenate((MinEst,[kgdRoll[4]]))
                    MaxEst = np.concatenate((MaxEst,[kgdRoll[5]]))
                if k_gn:
                    pr = 0
                    for i in range(len(thre_list)):
                        if 'Flex/Ext_Kalman_GN' == thre_list[i][1]:
                            pr = thre_list[i][0]
                    KGNR,KGNRRrom=pyjamalib.DataProcessing.pattern_extraction(df['Flex/Ext_Kalman_GN'],df['Time'],threshold=pr, bias=bias, cycle=cycle);
                    kgnRoll=pyjamalib.DataProcessing.patternCI(KGNR[:,0],KGNR[:,1],poly_degree=poly_degree,CI=CI, df=False);
                    kgn_R = pyjamalib.DataProcessing.rom_mean(KGNRRrom)
                    index = np.concatenate((index,['Flex/Ext_Kalman_GN']))
                    Rom = np.concatenate((Rom,[max(df['Flex/Ext_Kalman_GN'])-min(df['Flex/Ext_Kalman_GD'])]))
                    Mean = np.concatenate((Mean,[kgn_R])) 
                    Std = np.concatenate((Std,[kgnRoll[1]])) 
                    CIlist = np.concatenate((CIlist,[kgnRoll[0]]))
                    Var = np.concatenate((Var,[kgnRoll[7]]))
                    Min = np.concatenate((Min,[min(df['Flex/Ext_Kalman_GN'])]))
                    Max = np.concatenate((Max,[max(df['Flex/Ext_Kalman_GN'])]))
                    MinEst = np.concatenate((MinEst,[kgnRoll[4]]))
                    MaxEst = np.concatenate((MaxEst,[kgnRoll[5]]))
                if mad:
                    pr = 0
                    for i in range(len(thre_list)):
                        if 'Flex/Ext_Madgwick' == thre_list[i][1]:
                            pr = thre_list[i][0]
                    MADR,MADRrom=pyjamalib.DataProcessing.pattern_extraction(df['Flex/Ext_Madgwick'],df['Time'],threshold=pr, bias=bias, cycle=cycle);
                    madRoll=pyjamalib.DataProcessing.patternCI(MADR[:,0],MADR[:,1],poly_degree=poly_degree,CI=CI, df=False);
                    mad_R = pyjamalib.DataProcessing.rom_mean(MADRrom)
                    index = np.concatenate((index,['Flex/Ext_Madgwick']))
                    Rom = np.concatenate((Rom,[max(df['Flex/Ext_Madgwick'])-min(df['Flex/Ext_Madgwick'])]))
                    Mean = np.concatenate((Mean,[mad_R])) 
                    Std = np.concatenate((Std,[madRoll[1]])) 
                    CIlist = np.concatenate((CIlist,[madRoll[0]]))
                    Var = np.concatenate((Var,[madRoll[7]]))
                    Min = np.concatenate((Min,[min(df['Flex/Ext_Madgwick'])]))
                    Max = np.concatenate((Max,[max(df['Flex/Ext_Madgwick'])]))
                    MinEst = np.concatenate((MinEst,[madRoll[4]]))
                    MaxEst = np.concatenate((MaxEst,[madRoll[5]]))

            if patternPitch:
                if euler:
                    pp = 0
                    for i in range(len(thre_list)):
                        if 'Adu/Abd' == thre_list[i][1]:
                            pp = thre_list[i][0]    
                    rP,rawProm=pyjamalib.DataProcessing.pattern_extraction(df['Adu/Abd'],df['Time'],threshold=pp, bias=bias, cycle=cycle);
                    rawPitch=pyjamalib.DataProcessing.patternCI(rP[:,0],rP[:,1],poly_degree=poly_degree,CI=CI,df=False);
                    rm_P = pyjamalib.DataProcessing.rom_mean(rawProm)
                    index = np.concatenate((index,['Adu/Abd']))
                    Rom = np.concatenate((Rom,[max(df['Adu/Abd'])-min(df['Adu/Abd'])]))
                    Mean = np.concatenate((Mean,[rm_P])) 
                    Std = np.concatenate((Std,[rawPitch[1]])) 
                    CIlist = np.concatenate((CIlist,[rawPitch[0]]))
                    Var = np.concatenate((Var,[rawPitch[7]]))
                    Min = np.concatenate((Min,[min(df['Adu/Abd'])]))
                    Max = np.concatenate((Max,[max(df['Adu/Abd'])]))
                    MinEst = np.concatenate((MinEst,[rawPitch[4]]))
                    MaxEst = np.concatenate((MaxEst,[rawPitch[5]]))
                if cf:
                    pp = 0
                    for i in range(len(thre_list)):
                        if 'Adu/Abd_CF' == thre_list[i][1]:
                            pp = thre_list[i][0] 
                    CFP,CFProm=pyjamalib.DataProcessing.pattern_extraction(df['Adu/Abd_CF'],df['Time'],threshold=pp, bias=bias, cycle=cycle);
                    cfPitch=pyjamalib.DataProcessing.patternCI(CFP[:,0],CFP[:,1],poly_degree=poly_degree,CI=CI, df=False);
                    cf_P = pyjamalib.DataProcessing.rom_mean(CFProm)
                    index = np.concatenate((index,['Adu/Abd_CF']))
                    Rom = np.concatenate((Rom,[max(df['Adu/Abd_CF'])-min(df['Adu/Abd_CF'])]))
                    Mean = np.concatenate((Mean,[cf_P])) 
                    Std = np.concatenate((Std,[cfPitch[1]])) 
                    CIlist = np.concatenate((CIlist,[cfPitch[0]]))
                    Var = np.concatenate((Var,[cfPitch[7]]))
                    Min = np.concatenate((Min,[min(df['Adu/Abd_CF'])]))
                    Max = np.concatenate((Max,[max(df['Adu/Abd_CF'])]))
                    MinEst = np.concatenate((MinEst,[cfPitch[4]]))
                    MaxEst = np.concatenate((MaxEst,[cfPitch[5]]))
                if cf_gd:
                    pp = 0
                    for i in range(len(thre_list)):
                        if 'Adu/Abd_CF_GD' == thre_list[i][1]:
                            pp = thre_list[i][0]
                    CFGDP,CFGDProm=pyjamalib.DataProcessing.pattern_extraction(df['Adu/Abd_CF_GD'],df['Time'],threshold=pp, bias=bias, cycle=cycle);
                    cfgdPitch=pyjamalib.DataProcessing.patternCI(CFGDP[:,0],CFGDP[:,1],poly_degree=poly_degree,CI=CI, df=False);
                    cfgd_P = pyjamalib.DataProcessing.rom_mean(CFGDProm)
                    index = np.concatenate((index,['Adu/Abd_CF_GD']))
                    Rom = np.concatenate((Rom,[max(df['Adu/Abd_CF_GD'])-min(df['Adu/Abd_CF_GD'])]))
                    Mean = np.concatenate((Mean,[cfgd_P])) 
                    Std = np.concatenate((Std,[cfgdPitch[1]])) 
                    CIlist = np.concatenate((CIlist,[cfgdPitch[0]]))
                    Var = np.concatenate((Var,[cfgdPitch[7]]))
                    Min = np.concatenate((Min,[min(df['Adu/Abd_CF_GD'])]))
                    Max = np.concatenate((Max,[max(df['Adu/Abd_CF_GD'])]))
                    MinEst = np.concatenate((MinEst,[cfgdPitch[4]]))
                    MaxEst = np.concatenate((MaxEst,[cfgdPitch[5]]))
                if cf_gn:
                    pp = 0
                    for i in range(len(thre_list)):
                        if 'Adu/Abd_CF_GN' == thre_list[i][1]:
                            pp = thre_list[i][0]
                    CFGNP,CFGNProm=pyjamalib.DataProcessing.pattern_extraction(df['Adu/Abd_CF_GN'],df['Time'],threshold=pp, bias=bias, cycle=cycle);
                    cfgnPitch=pyjamalib.DataProcessing.patternCI(CFGNP[:,0],CFGNP[:,1],poly_degree=poly_degree,CI=CI, df=False);
                    cfgn_P = pyjamalib.DataProcessing.rom_mean(CFGNProm)
                    index = np.concatenate((index,['Adu/Abd_CF_GN']))
                    Rom = np.concatenate((Rom,[max(df['Adu/Abd_CF_GN'])-min(df['Adu/Abd_CF_GN'])]))
                    Mean = np.concatenate((Mean,[cfgn_P])) 
                    Std = np.concatenate((Std,[cfgnPitch[1]])) 
                    CIlist = np.concatenate((CIlist,[cfgnPitch[0]]))
                    Var = np.concatenate((Var,[cfgnPitch[7]]))
                    Min = np.concatenate((Min,[min(df['Adu/Abd_CF_GN'])]))
                    Max = np.concatenate((Max,[max(df['Adu/Abd_CF_GN'])]))
                    MinEst = np.concatenate((MinEst,[cfgnPitch[4]]))
                    MaxEst = np.concatenate((MaxEst,[cfgnPitch[5]]))
                if k_gd:
                    pp = 0
                    for i in range(len(thre_list)):
                        if 'Adu/Abd_Kalman_GD' == thre_list[i][1]:
                            pp = thre_list[i][0]
                    KGDP,KGDProm=pyjamalib.DataProcessing.pattern_extraction(df['Adu/Abd_Kalman_GD'],df['Time'],threshold=pp, bias=bias, cycle=cycle);
                    kgdPitch=pyjamalib.DataProcessing.patternCI(KGDP[:,0],KGDP[:,1],poly_degree=poly_degree,CI=CI, df=False);
                    kgd_P = pyjamalib.DataProcessing.rom_mean(KGDProm)
                    index = np.concatenate((index,['Adu/Abd_Kalman_GD']))
                    Rom = np.concatenate((Rom,[max(df['Adu/Abd_Kalman_GD'])-min(df['Adu/Abd_Kalman_GD'])]))
                    Mean = np.concatenate((Mean,[kgd_P])) 
                    Std = np.concatenate((Std,[kgdPitch[1]])) 
                    CIlist = np.concatenate((CIlist,[kgdPitch[0]]))
                    Var = np.concatenate((Var,[kgdPitch[7]]))
                    Min = np.concatenate((Min,[min(df['Adu/Abd_Kalman_GD'])]))
                    Max = np.concatenate((Max,[max(df['Adu/Abd_Kalman_GD'])]))
                    MinEst = np.concatenate((MinEst,[kgdPitch[4]]))
                    MaxEst = np.concatenate((MaxEst,[kgdPitch[5]]))
                if k_gn:
                    pp = 0
                    for i in range(len(thre_list)):
                        if 'Adu/Abd_Kalman_GN' == thre_list[i][1]:
                            pp = thre_list[i][0]
                    KGNP,KGNProm=pyjamalib.DataProcessing.pattern_extraction(df['Adu/Abd_Kalman_GN'],df['Time'],threshold=pp, bias=bias, cycle=cycle);
                    kgnPitch=pyjamalib.DataProcessing.patternCI(KGNP[:,0],KGNP[:,1],poly_degree=poly_degree,CI=CI, df=False);
                    kgn_P = pyjamalib.DataProcessing.rom_mean(KGNProm)
                    index = np.concatenate((index,['Adu/Abd_Kalman_GN']))
                    Rom = np.concatenate((Rom,[max(df['Adu/Abd_Kalman_GN'])-min(df['Adu/Abd_Kalman_GN'])]))
                    Mean = np.concatenate((Mean,[kgn_P])) 
                    Std = np.concatenate((Std,[kgnPitch[1]])) 
                    CIlist = np.concatenate((CIlist,[kgnPitch[0]]))
                    Var = np.concatenate((Var,[kgnPitch[7]]))
                    Min = np.concatenate((Min,[min(df['Adu/Abd_Kalman_GN'])]))
                    Max = np.concatenate((Max,[max(df['Adu/Abd_Kalman_GN'])]))
                    MinEst = np.concatenate((MinEst,[kgnPitch[4]]))
                    MaxEst = np.concatenate((MaxEst,[kgnPitch[5]]))
                if mad:
                    pp = 0
                    for i in range(len(thre_list)):
                        if 'Adu/Abd_Madgwick' == thre_list[i][1]:
                            pp = thre_list[i][0]
                    MADP,MADProm=pyjamalib.DataProcessing.pattern_extraction(df['Adu/Abd_Madgwick'],df['Time'],threshold=pp, bias=bias, cycle=cycle);
                    madPitch=pyjamalib.DataProcessing.patternCI(MADP[:,0],MADP[:,1],poly_degree=poly_degree,CI=CI, df=False);
                    mad_P = pyjamalib.DataProcessing.rom_mean(MADProm)
                    index = np.concatenate((index,['Adu/Abd_Madgwick']))
                    Rom = np.concatenate((Rom,[max(df['Adu/Abd_Madgwick'])-min(df['Adu/Abd_Madgwick'])]))
                    Mean = np.concatenate((Mean,[mad_P])) 
                    Std = np.concatenate((Std,[madPitch[1]])) 
                    CIlist = np.concatenate((CIlist,[madPitch[0]]))
                    Var = np.concatenate((Var,[madPitch[7]]))
                    Min = np.concatenate((Min,[min(df['Adu/Abd_Madgwick'])]))
                    Max = np.concatenate((Max,[max(df['Adu/Abd_Madgwick'])]))
                    MinEst = np.concatenate((MinEst,[madPitch[4]]))
                    MaxEst = np.concatenate((MaxEst,[madPitch[5]]))

            if patternYaw:
                if euler:
                    py = 0
                    for i in range(len(thre_list)):
                        if 'Int/Ext_Rot' == thre_list[i][1]:
                            py = thre_list[i][0]
                    rY,rawYrom=pyjamalib.DataProcessing.pattern_extraction(df['Int/Ext_Rot'],df['Time'],threshold=py, bias=bias, cycle=cycle);
                    rawYaw=pyjamalib.DataProcessing.patternCI(rY[:,0],rY[:,1],poly_degree=poly_degree,CI=CI,df=False);
                    rm_Y = pyjamalib.DataProcessing.rom_mean(rawYrom)
                    index = np.concatenate((index,['Int/Ext_Rot']))
                    Rom = np.concatenate((Rom,[max(df['Int/Ext_Rot'])-min(df['Int/Ext_Rot'])]))
                    Mean = np.concatenate((Mean,[rm_Y])) 
                    Std = np.concatenate((Std,[rawYaw[1]])) 
                    CIlist = np.concatenate((CIlist,[rawYaw[0]]))
                    Var = np.concatenate((Var,[rawYaw[7]]))
                    Min = np.concatenate((Min,[min(df['Int/Ext_Rot'])]))
                    Max = np.concatenate((Max,[max(df['Int/Ext_Rot'])]))
                    MinEst = np.concatenate((MinEst,[rawYaw[4]]))
                    MaxEst = np.concatenate((MaxEst,[rawYaw[5]]))
                if cf:
                    py = 0
                    for i in range(len(thre_list)):
                        if 'Int/Ext_Rot_CF' == thre_list[i][1]:
                            py = thre_list[i][0]            
                    CFY,CFYrom=pyjamalib.DataProcessing.pattern_extraction(df['Int/Ext_Rot_CF'],df['Time'],threshold=py, bias=bias, cycle=cycle);
                    cfYaw=pyjamalib.DataProcessing.patternCI(CFY[:,0],CFY[:,1],poly_degree=poly_degree,CI=CI, df=False);
                    cf_Y = pyjamalib.DataProcessing.rom_mean(CFYrom)
                    index = np.concatenate((index,['Int/Ext_Rot_CF']))
                    Rom = np.concatenate((Rom,[max(df['Int/Ext_Rot_CF'])-min(df['Int/Ext_Rot_CF'])]))
                    Mean = np.concatenate((Mean,[cf_Y])) 
                    Std = np.concatenate((Std,[cfYaw[1]])) 
                    CIlist = np.concatenate((CIlist,[cfYaw[0]]))
                    Var = np.concatenate((Var,[cfYaw[7]]))
                    Min = np.concatenate((Min,[min(df['Int/Ext_Rot_CF'])]))
                    Max = np.concatenate((Max,[max(df['Int/Ext_Rot_CF'])]))
                    MinEst = np.concatenate((MinEst,[cfYaw[4]]))
                    MaxEst = np.concatenate((MaxEst,[cfYaw[5]]))
                if cf_gd:
                    py = 0
                    for i in range(len(thre_list)):
                        if 'Int/Ext_Rot_CF_GD' == thre_list[i][1]:
                            py = thre_list[i][0]
                    CFGDY,CFGDYrom=pyjamalib.DataProcessing.pattern_extraction(df['Int/Ext_Rot_CF_GD'],df['Time'],threshold=py, bias=bias, cycle=cycle);
                    cfgdYaw=pyjamalib.DataProcessing.patternCI(CFGDY[:,0],CFGDY[:,1],poly_degree=poly_degree,CI=CI, df=False);
                    cfgd_Y = pyjamalib.DataProcessing.rom_mean(CFGDYrom)
                    index = np.concatenate((index,['Int/Ext_Rot_CF_GD']))
                    Rom = np.concatenate((Rom,[max(df['Int/Ext_Rot_CF_GD'])-min(df['Int/Ext_Rot_CF_GD'])]))
                    Mean = np.concatenate((Mean,[cfgd_Y])) 
                    Std = np.concatenate((Std,[cfgdYaw[1]])) 
                    CIlist = np.concatenate((CIlist,[cfgdYaw[0]]))
                    Var = np.concatenate((Var,[cfgdYaw[7]]))
                    Min = np.concatenate((Min,[min(df['Int/Ext_Rot_CF_GD'])]))
                    Max = np.concatenate((Max,[max(df['Int/Ext_Rot_CF_GD'])]))
                    MinEst = np.concatenate((MinEst,[cfgdYaw[4]]))
                    MaxEst = np.concatenate((MaxEst,[cfgdYaw[5]]))   
                if cf_gn:
                    py = 0
                    for i in range(len(thre_list)):
                        if 'Int/Ext_Rot_CF_GN' == thre_list[i][1]:
                            py = thre_list[i][0]
                    CFGNY,CFGNYrom=pyjamalib.DataProcessing.pattern_extraction(df['Int/Ext_Rot_CF_GN'],df['Time'],threshold=py, bias=bias, cycle=cycle);
                    cfgnYaw=pyjamalib.DataProcessing.patternCI(CFGNY[:,0],CFGNY[:,1],poly_degree=poly_degree,CI=CI, df=False);
                    cfgn_Y = pyjamalib.DataProcessing.rom_mean(CFGNYrom)
                    index = np.concatenate((index,['Int/Ext_Rot_CF_GN']))
                    Rom = np.concatenate((Rom,[max(df['Int/Ext_Rot_CF_GN'])-min(df['Int/Ext_Rot_CF_GN'])]))
                    Mean = np.concatenate((Mean,[cfgn_Y])) 
                    Std = np.concatenate((Std,[cfgnYaw[1]])) 
                    CIlist = np.concatenate((CIlist,[cfgnYaw[0]]))
                    Var = np.concatenate((Var,[cfgnYaw[7]]))
                    Min = np.concatenate((Min,[min(df['Int/Ext_Rot_CF_GN'])]))
                    Max = np.concatenate((Max,[max(df['Int/Ext_Rot_CF_GN'])]))
                    MinEst = np.concatenate((MinEst,[cfgnYaw[4]]))
                    MaxEst = np.concatenate((MaxEst,[cfgnYaw[5]]))
                if k_gd:
                    py = 0
                    for i in range(len(thre_list)):
                        if 'Int/Ext_Rot_Kalman_GD' == thre_list[i][1]:
                            py = thre_list[i][0]
                    KGDY,KGDYrom=pyjamalib.DataProcessing.pattern_extraction(df['Int/Ext_Rot_Kalman_GD'],df['Time'],threshold=py, bias=bias, cycle=cycle);
                    kgdYaw=pyjamalib.DataProcessing.patternCI(KGDY[:,0],KGDY[:,1],poly_degree=poly_degree,CI=CI, df=False); 
                    kgd_Y = pyjamalib.DataProcessing.rom_mean(KGDYrom)
                    index = np.concatenate((index,['Int/Ext_Rot_Kalman_GD']))
                    Rom = np.concatenate((Rom,[max(df['Int/Ext_Rot_Kalman_GD'])-min(df['Int/Ext_Rot_Kalman_GD'])]))
                    Mean = np.concatenate((Mean,[kgd_Y])) 
                    Std = np.concatenate((Std,[kgdYaw[1]])) 
                    CIlist = np.concatenate((CIlist,[kgdYaw[0]]))
                    Var = np.concatenate((Var,[kgdYaw[7]]))
                    Min = np.concatenate((Min,[min(df['Int/Ext_Rot_Kalman_GD'])]))
                    Max = np.concatenate((Max,[max(df['Int/Ext_Rot_Kalman_GD'])]))
                    MinEst = np.concatenate((MinEst,[kgdYaw[4]]))
                    MaxEst = np.concatenate((MaxEst,[kgdYaw[5]]))
                if k_gn:
                    py = 0
                    for i in range(len(thre_list)):
                        if 'Int/Ext_Rot_Kalman_GN' == thre_list[i][1]:
                            py = thre_list[i][0]
                    KGNY,KGNYrom=pyjamalib.DataProcessing.pattern_extraction(df['Int/Ext_Rot_Kalman_GN'],df['Time'],threshold=py, bias=bias, cycle=cycle);
                    kgnYaw=pyjamalib.DataProcessing.patternCI(KGNY[:,0],KGNY[:,1],poly_degree=poly_degree,CI=CI, df=False); 
                    kgn_Y = pyjamalib.DataProcessing.rom_mean(KGNYrom)
                    index = np.concatenate((index,['Int/Ext_Rot_Kalman_GN']))
                    Rom = np.concatenate((Rom,[max(df['Int/Ext_Rot_Kalman_GN'])-min(df['Int/Ext_Rot_Kalman_GN'])]))
                    Mean = np.concatenate((Mean,[kgn_Y])) 
                    Std = np.concatenate((Std,[kgnYaw[1]])) 
                    CIlist = np.concatenate((CIlist,[kgnYaw[0]]))
                    Var = np.concatenate((Var,[kgnYaw[7]]))
                    Min = np.concatenate((Min,[min(df['Int/Ext_Rot_Kalman_GN'])]))
                    Max = np.concatenate((Max,[max(df['Int/Ext_Rot_Kalman_GN'])]))
                    MinEst = np.concatenate((MinEst,[kgnYaw[4]]))
                    MaxEst = np.concatenate((MaxEst,[kgnYaw[5]]))
                if mad:
                    py = 0
                    for i in range(len(thre_list)):
                        if 'Int/Ext_Rot_Madgwick' == thre_list[i][1]:
                            py = thre_list[i][0]
                    MADY,MADYrom=pyjamalib.DataProcessing.pattern_extraction(df['Int/Ext_Rot_Madgwick'],df['Time'],threshold=py, bias=bias, cycle=cycle);
                    madYaw=pyjamalib.DataProcessing.patternCI(MADY[:,0],MADY[:,1],poly_degree=poly_degree,CI=CI, df=False);
                    mad_Y = pyjamalib.DataProcessing.rom_mean(MADYrom)
                    index = np.concatenate((index,['Int/Ext_Rot_Madgwick']))
                    Rom = np.concatenate((Rom,[max(df['Int/Ext_Rot_Madgwick'])-min(df['Int/Ext_Rot_Madgwick'])]))
                    Mean = np.concatenate((Mean,[mad_Y])) 
                    Std = np.concatenate((Std,[madYaw[1]])) 
                    CIlist = np.concatenate((CIlist,[madYaw[0]]))
                    Var = np.concatenate((Var,[madYaw[7]]))
                    Min = np.concatenate((Min,[min(df['Int/Ext_Rot_Madgwick'])]))
                    Max = np.concatenate((Max,[max(df['Int/Ext_Rot_Madgwick'])]))
                    MinEst = np.concatenate((MinEst,[madYaw[4]]))
                    MaxEst = np.concatenate((MaxEst,[madYaw[5]]))

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

        columns = []
        row_names = []
        
        #Roll

        if 'Flex/Ext' in df:
            romRawRoll = max(df['Flex/Ext'])-min(df['Flex/Ext'])
            columns = np.concatenate((columns,[romRawRoll]))
            row_names = np.concatenate((row_names,['Flex/Ext']))
        if 'Flex/Ext_CF' in df:
            romRollCF = max(df['Flex/Ext_CF'])-min(df['Flex/Ext_CF'])
            columns = np.concatenate((columns,[romRollCF]))
            row_names = np.concatenate((row_names,['Flex/Ext_CF']))
        if 'Flex/Ext_CF_GD' in df:
            romRollCfGd = max(df['Flex/Ext_CF_GD'])-min(df['Flex/Ext_CF_GD'])
            columns = np.concatenate((columns,[romRollCfGd]))
            row_names = np.concatenate((row_names,['Flex/Ext_CF_GD']))
        if 'Flex/Ext_CF_GN' in df:       
            romRollCfGn = max(df['Flex/Ext_CF_GN'])-min(df['Flex/Ext_CF_GN'])
            columns = np.concatenate((columns,[romRollCfGn]))
            row_names = np.concatenate((row_names,['Flex/Ext_CF_GN']))
        if 'Flex/Ext_Kalman_GD' in df: 
            romRollKalmanGd = max(df['Flex/Ext_Kalman_GD'])-min(df['Flex/Ext_Kalman_GD'])
            columns = np.concatenate((columns,[romRollKalmanGd]))
            row_names = np.concatenate((row_names,['Flex/Ext_Kalman_GD']))
        if 'Flex/Ext_Kalman_GN' in df: 
            romRollKalmanGn = max(df['Flex/Ext_Kalman_GN'])-min(df['Flex/Ext_Kalman_GN'])
            columns = np.concatenate((columns,[romRollKalmanGn]))
            row_names = np.concatenate((row_names,['Flex/Ext_Kalman_GN']))
        if 'Flex/Ext_Madgwick' in df:
            romRollMad = max(df['Flex/Ext_Madgwick'])-min(df['Flex/Ext_Madgwick'])
            columns = np.concatenate((columns,[romRollMad]))
            row_names = np.concatenate((row_names,['Flex/Ext_Madgwick']))
        
        #Pitch

        if 'Adu/Abd' in df:
            romRawPitch = max(df['Adu/Abd'])-min(df['Adu/Abd'])
            columns = np.concatenate((columns,[romRawPitch]))
            row_names = np.concatenate((row_names,['Adu/Abd']))
        if 'Adu/Abd_CF' in df:
            romPitchCF = max(df['Adu/Abd_CF'])-min(df['Adu/Abd_CF'])
            columns = np.concatenate((columns,[romPitchCF]))
            row_names = np.concatenate((row_names,['Adu/Abd_CF']))
        if 'Adu/Abd_CF_GD' in df:
            romPitchCfGd =max(df['Adu/Abd_CF_GD'])-min(df['Adu/Abd_CF_GD'])
            columns = np.concatenate((columns,[romPitchCfGd]))
            row_names = np.concatenate((row_names,['Adu/Abd_CF_GD']))
        if 'Adu/Abd_CF_GN' in df:       
            romPitchCfGn = max(df['Adu/Abd_CF_GN'])-min(df['Adu/Abd_CF_GN'])
            columns = np.concatenate((columns,[romPitchCfGn]))
            row_names = np.concatenate((row_names,['Adu/Abd_CF_GN']))
        if 'Adu/Abd_Kalman_GD' in df: 
            romPitchKalmanGd = max(df['Adu/Abd_Kalman_GD'])-min(df['Adu/Abd_Kalman_GD'])
            columns = np.concatenate((columns,[romPitchKalmanGd]))
            row_names = np.concatenate((row_names,['Adu/Abd_Kalman_GD']))
        if 'Adu/Abd_Kalman_GN' in df: 
            romPitchKalmanGn = max(df['Adu/Abd_Kalman_GN'])-min(df['Adu/Abd_Kalman_GN'])
            columns = np.concatenate((columns,[romPitchKalmanGn]))
            row_names = np.concatenate((row_names,['Adu/Abd_Kalman_GN']))
        if 'Adu/Abd_Madgwick' in df:
            romPitchMad = max(df['Adu/Abd_Madgwick'])-min(df['Adu/Abd_Madgwick'])
            columns = np.concatenate((columns,[romPitchMad]))
            row_names = np.concatenate((row_names,['Adu/Abd_Madgwick']))  

        #Yaw   
        
        if 'Int/Ext_Rot' in df:
            romRawYaw = max(df['Int/Ext_Rot'])-min(df['Int/Ext_Rot'])
            columns = np.concatenate((columns,[romRawYaw]))
            row_names = np.concatenate((row_names,['Int/Ext_Rot']))
        if 'Int/Ext_Rot_CF' in df:
            romYawCF = max(df['Int/Ext_Rot_CF'])-min(df['Int/Ext_Rot_CF'])
            columns = np.concatenate((columns,[romYawCF]))
            row_names = np.concatenate((row_names,['Int/Ext_Rot_CF']))
        if 'Int/Ext_Rot_CF_GD' in df:
            romYawCfGd =max(df['Int/Ext_Rot_CF_GD'])-min(df['Int/Ext_Rot_CF_GD'])
            columns = np.concatenate((columns,[romYawCfGd]))
            row_names = np.concatenate((row_names,['Int/Ext_Rot_CF_GD']))
        if 'Int/Ext_Rot_CF_GN' in df:       
            romYawCfGn = max(df['Int/Ext_Rot_CF_GN'])-min(df['Int/Ext_Rot_CF_GN'])
            columns = np.concatenate((columns,[romYawCfGn]))
            row_names = np.concatenate((row_names,['Int/Ext_Rot_CF_GN']))
        if 'Int/Ext_Rot_Kalman_GD' in df: 
            romYawKalmanGd = max(df['Int/Ext_Rot_Kalman_GD'])-min(df['Int/Ext_Rot_Kalman_GD'])
            columns = np.concatenate((columns,[romYawKalmanGd]))
            row_names = np.concatenate((row_names,['Int/Ext_Rot_Kalman_GD']))
        if 'Int/Ext_Rot_Kalman_GN' in df: 
            romYawKalmanGn = max(df['Int/Ext_Rot_Kalman_GN'])-min(df['Int/Ext_Rot_Kalman_GN'])
            columns = np.concatenate((columns,[romYawKalmanGn]))
            row_names = np.concatenate((row_names,['Int/Ext_Rot_Kalman_GN']))
        if 'Int/Ext_Rot_Madgwick' in df:
            romYawMad = max(df['Int/Ext_Rot_Madgwick'])-min(df['Int/Ext_Rot_Madgwick'])
            columns = np.concatenate((columns,[romYawMad]))
            row_names = np.concatenate((row_names,['Int/Ext_Rot_Madgwick']))        
                
        df_rom = pd.DataFrame({'Angles':columns},row_names )
        
        return df_rom