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

class IMUDataProcessing:
    """Integrates all functions to perform data 
    processing to calculate the joint angle.

    See Also
    --------
    Developed by T.F Almeida in 25/03/2021
    
    For more information see:
    https://github.com/tuliofalmeida    
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
        https://github.com/tuliofalmeida   
        """
        self.Quaternion = np.asarray(Quaternion)
        self.qGyro_1 = np.zeros(4)
        self.qGyro_1[0] = 1
        self.qFilt_1 = np.zeros(4)
        self.qFilt_1[0] = 1
        self.accF_Length = 13
        self.i = 0

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
        https://github.com/tuliofalmeida  """
        return atan2(-acc[0],sqrt((acc[1]*acc[1]) + (acc[2] * acc[2])))

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
        https://github.com/tuliofalmeida  """
        return atan2(acc[1],sqrt((acc[0] * acc[0]) + (acc[2] * acc[2])))

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
        https://github.com/tuliofalmeida  """
        Yh = (mag[1] * cos(roll)) - (mag[2] * sin(roll))
        Xh = (mag[0] * cos(pitch))+ (mag[1] * sin(roll)*sin(pitch)) + (mag[2] * cos(roll) * sin(pitch))	
        return atan2(Yh, Xh)

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
        https://github.com/tuliofalmeida  """
        quat = []
        euler = []
        quat.append(np.asarray(q))
        
        for i in range(len(Acc)):
            quat.append(IMUDataProcessing.GaussNewtonMethod(quat[i-1],Acc[i],Mag[i]))
            if conj == True:
                euler.append(IMUDataProcessing.GetAnglesFromQuaternion(IMUDataProcessing.quaternConj(quat[i])))
            else:
                euler.append(IMUDataProcessing.GetAnglesFromQuaternion(quat[i]))

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
        https://github.com/tuliofalmeida
        """
        q0 = cos(np.divide(angle,2))
        q1 = -axis[0]*sin(np.divide(angle,2))
        q2 = -axis[1]*sin(np.divide(angle,2))
        q3 = -axis[2]*sin(np.divide(angle,2))
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
        https://github.com/tuliofalmeida
        """
        kx = axis[0]
        ky = axis[1]
        kz = axis[2]
        cT = cos(angle)
        sT = sin(angle)
        vT = 1 - cos(angle)

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
        https://github.com/tuliofalmeida
        """
        R = np.zeros((3,3))

        R[0,0] = cos(psi) * cos(theta) 
        R[0,1] = -sin(psi) * cos(phi) + cos(psi) * sin(theta) * sin(phi) 
        R[0,2] = sin(psi) * sin(phi) + cos(psi) * sin(theta) * cos(phi)

        R[1,0] = sin(psi) * cos(theta)
        R[1,1] = cos(psi) * cos(phi) + sin(psi) * sin(theta) * sin(phi)
        R[1,2] = -cos(psi) * sin(phi) + sin(psi) * sin(theta) * cos(phi)

        R[2,0] = -sin(theta)
        R[2,1] = cos(theta) * sin(phi)
        R[2,2] = cos(theta) * cos(phi)

        return R

    def quatern2euler(q):
        """ Converts a quaternion orientation to 
            ZYX Euler angles where phi is a rotation 
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
        https://github.com/tuliofalmeida
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

        phi = atan2(R[2,1], R[2,2])
        theta = -atan(R[2,0]/sqrt(1-R[2,0]**2))
        psi = atan2(R[1,0], R[0,0] )

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
        https://github.com/tuliofalmeida
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
        https://github.com/tuliofalmeida
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
        https://github.com/tuliofalmeida
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
        https://github.com/tuliofalmeida
        """
        if np.size(np.asarray(R).shape) == 2:
            row, col = np.asarray(R).shape
            numR = 1
            phi = atan2(R[2,1], R[2,2] )
            theta = -atan(R[2,0]/sqrt(1-R[2,0]**2))
            psi = atan2(R[1,0], R[0,0] )

            euler = [phi,theta,psi]

            return euler
            
        else:
            row, col, numR = np.asarray(R).shape
            phi = []
            theta = []
            psi =[]
            for i in range(numR):
                phi.append(atan2(R[2,1,i], R[2,2,i]))
                theta.append(-atan(R[2,0,i]/sqrt(1-R[2,0,i]**2)))
                psi.append(atan2(R[1,0,i], R[0,0,i]))

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
        https://github.com/tuliofalmeida
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

    def ComputeJacobian(q,Acc,Mag):
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
        https://github.com/tuliofalmeida
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

    def ComputeM_Matrix(q):
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
        https://github.com/tuliofalmeida
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

    def GaussNewtonMethod(q,Acc,Mag):
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
        https://github.com/tuliofalmeida
        """
        # Compute the new step quaternions by mean of the Gauss-Newton method
        i = 0
        n_k = np.asarray([q[1], q[2],q[3], q[0]]).T
        
        while(i < 3):
            # Magnetometer compensation
            m = Mag/np.linalg.norm(Mag)
            q_coniug = np.asarray([q[0],-q[1],-q[2],-q[3]])
            MAG = np.asarray([0,m[0],m[1],m[2]])
            hTemp = IMUDataProcessing.QuaternionProduct(q,MAG)
            h = IMUDataProcessing.QuaternionProduct(hTemp,q_coniug)
            bMagn = np.asarray([sqrt(h[1]**2+h[2]**2), 0, h[3]]).T
            bMagn = bMagn/np.linalg.norm(bMagn)
            # End magnetometer compensation
            
            J_nk = IMUDataProcessing.ComputeJacobian(q,Acc,Mag)

            M = IMUDataProcessing.ComputeM_Matrix(q)

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
        https://github.com/tuliofalmeida
        """
        q0=q[0]
        q1=q[1]
        q2=q[2]
        q3=q[3]
        
        AnglesX = atan2(2*(q2*q3)-2*q0*q1,2*q0**2+2*q3**2-1)*180/pi
        AnglesY = -asin(2*q1*q3+2*q0*q2)*180/pi
        AnglesZ = atan2(2*q1*q2-2*q0*q3,2*q0**2+2*q1**2-1)*180/pi

        Angles = np.asarray([AnglesX,AnglesY,AnglesZ])

        return Angles

    def GetQuaternionFromAngle(Angles):
        """Estimates the quaternion orientation
            from euler angles.

        Parameters
        ----------
        Angles: ndarray
           Euler angles (Roll,Pitch,Yaw).

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
        https://github.com/tuliofalmeida
        """       
        q = np.zeros((4,1))
        x = Angles[:,0]*pi/180
        y = Angles[:,1]*pi/180
        z = Angles[:,2]*pi/180

        q[0] = cos(x/2)*cos(y/2)*cos(z/2)+sin(x/2)*sin(y/2)*sin(z/2)
        q[1] = sin(x/2)*cos(y/2)*cos(z/2)-cos(x/2)*sin(y/2)*sin(z/2)
        q[2] = cos(x/2)*sin(y/2)*cos(z/2)+sin(x/2)*cos(y/2)*sin(z/2)
        q[3] = cos(x/2)*cos(y/2)*sin(z/2)-sin(x/2)*sin(y/2)*cos(z/2)

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
        https://github.com/tuliofalmeida
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
            hTemp = IMUDataProcessing.QuaternionProduct(q,M)
            h = IMUDataProcessing.QuaternionProduct(hTemp,q_coniug)

            b = [sqrt(h[1]**2 + h[2]**2),0,h[3]]
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
        https://github.com/tuliofalmeida
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
        https://github.com/tuliofalmeida
        """
        nyq = 0.5 * freq
        normal_cutoff = cutoff / nyq
        b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
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
        https://github.com/tuliofalmeida
        """
        b, a = butter_lowpass(cutoff, freq, order=order)
        y = signal.lfilter(b, a, data)
        return y

    def low_pass_filter(IN, par = 0.1):
        """Low-pass butter filter

        Parameters
        ----------
        IN: ndarray
           Input data.

        par: float
           Filter intensity

        Returns
        -------
        out: ndarray
           Filtered data.

        See Also
        --------
        Developed by T.F Almeida in 25/03/2021
        
        For more information see:
        https://github.com/tuliofalmeida
        """
        out = []
        for i in range(len(IN)):
            if i == 0:
                out.append(IN[i])
            else:
                out.append(out[i-1] + par*(IN[i] - out[i-1]))

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
        https://github.com/tuliofalmeida
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

    def ComplementaryFilterGNUpdate(acc,gyr,mag,dt,alpha=.01,beta=.01, conj = True):
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
        alpha: float
           Acceleroeter data contribution to angle.
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
        https://github.com/tuliofalmeida
        """        
        AccF = acc/np.linalg.norm(acc)
        MagnF = mag/np.linalg.norm(mag)
        qOsserv = IMUDataProcessing.GaussNewtonMethod(self.qFilt_1,AccF,MagnF)
        qOsserv = beta*qOsserv/np.linalg.norm(qOsserv)
        if self.i <= (self.accF_Length+6):
            # Filtered values initialized with observed values
            qGyroFilt = qOsserv
            qFilt = qOsserv
            qGyro = qOsserv
        else:
            # Complementary Filter
            dq = 0.5*(IMUDataProcessing.QuaternionProduct(self.qFilt_1,np.asarray([0,gyr[0],gyr[1],gyr[2]]).T))
            qGyro = self.qGyro_1 + 0.5*(IMUDataProcessing.QuaternionProduct(self.qGyro_1,np.asarray([0,gyr[0],gyr[1],gyr[2]]).T))*dt
            qGyro = qGyro/np.linalg.norm(qGyro)
            qGyroFilt = self.qFilt_1+dq*dt
            qGyroFilt = qGyroFilt/np.linalg.norm(qGyroFilt)
            
            qFilt = qGyroFilt*(1-alpha)+qOsserv*alpha
            qFilt = qFilt/np.linalg.norm(qFilt)

        if conj == True:    
            Angles = IMUDataProcessing.GetAnglesFromQuaternion(IMUDataProcessing.quaternConj(qFilt))
        else:
            Angles = IMUDataProcessing.GetAnglesFromQuaternion(qFilt)

        self.qFilt_1 = qFilt
        self.qGyro_1 = qGyro
        return Angles

    def ComplementaryFilterGN(self,acc,gyr,mag,dt,alpha=.01,beta=.01, conj = True):
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
        alpha: float
           Acceleroeter data contribution to angle.
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
        https://github.com/tuliofalmeida
        """ 
        CF = IMUDataProcessing()
        acqSize = acc.shape[0]
        Angles = np.zeros((3,acqSize))

        CF.i = 1
        while (CF.i < acqSize):
            Angles[:,CF.i] = CF.ComplementaryFilterGNUpdate(acc[CF.i,:],gyr[CF.i,:],mag[CF.i,:],dt,alpha,beta, conj)
            CF.i += 1

        return np.asarray(Angles).T

    def ComplementaryFilterGDUpdate(acc,gyr,mag,dt,alpha=.01,beta=.5, conj = True):
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
           Acceleroeter data contribution to angle.
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
        https://github.com/tuliofalmeida
        """ 
        AccF = acc/np.linalg.norm(acc)
        MagnF = mag/np.linalg.norm(mag)

        dq = 0.5*(IMUDataProcessing.QuaternionProduct(self.qFilt_1,np.asarray([0,gyr[0],gyr[1],gyr[2]]).T))
        dqnorm = np.linalg.norm(dq)
        mu = 10*dqnorm[0,i]*dt
        qOsserv = IMUDataProcessing.GradientDescent(AccF,MagnF,qOsserv_1,mu)
        qOsserv = beta*qOsserv/np.linalg.norm(qOsserv)
        
        if (self.i <= self.accF_Length+9):
            qGyroFilt = qOsserv
            qFilt = qOsserv
        else:
            qGyroFilt = self.qFilt_1+dq*dt
            qGyroFilt = qGyroFilt/np.linalg.norm(qGyroFilt)

            dqnorm = np.linalg.norm(dq)
            mu = 10*dqnorm*dt
            qOsserv = IMUDataProcessing.GradientDescent(AccF,MagnF,self.qOsserv_1,mu)
            qOsserv = beta*qOsserv/np.linalg.norm(qOsserv)
            
            qFilt = qGyroFilt*(1-alpha)+qOsserv*alpha
            qFilt = qFilt/np.linalg.norm(qFilt)

        if conj == True:    
            Angles = IMUDataProcessing.GetAnglesFromQuaternion(IMUDataProcessing.quaternConj(qFilt))
        else:
            Angles = IMUDataProcessing.GetAnglesFromQuaternion(qFilt)

        self.qFilt_1 = qFilt
        self.qGyro_1 = qGyro
        return Angles

    def ComplementaryFilterGD(acc,gyr,mag,dt,alpha=.01,beta=.5, conj = True):
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
           Acceleroeter data contribution to angle.
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
        https://github.com/tuliofalmeida
        """ 
        CF = IMUDataProcessing()
        acqSize = acc.shape[0]
        Angles = np.zeros((3,acqSize))

        CF.i = 1
        while (CF.i < acqSize):
            Angles[:,CF.i] = CF.ComplementaryFilterGDUpdate(acc[CF.i,:],gyr[CF.i,:],mag[CF.i,:],dt,alpha,beta, conj)
            CF.i += 1

        return np.asarray(Angles).T

    def KalmanGD(acc,gyr,mag,dt = 1/75,R_In=[0.01,0.01,0.01,0.01],beta=.05,conj = True):
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
        https://github.com/tuliofalmeida
        """         
        acqSize = acc.shape[0]

        # Variance 
        _,varG = IMUDataProcessing.varianceEstimation(gyr)
        var = np.asarray([varG[0]**2,varG[1]**2,varG[2]**2]).T

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
            dq[:,i] = beta*(IMUDataProcessing.QuaternionProduct(qUpdate[:,i-1],G))
            dqnorm[:,i] = np.linalg.norm(dq[:,i])
            mu[0,i] = 10*dqnorm[0,i]*dt
            qOsserv[:,i] = IMUDataProcessing.GradientDescent(AccF[:,i],MagnF[:,i],qOsserv[:,i-1],mu[0,i])
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
            Angles[:,i] = IMUDataProcessing.GetAnglesFromQuaternion(qUpdate[:,i])

            if conj == True:
                Angles[:,i] = IMUDataProcessing.GetAnglesFromQuaternion(IMUDataProcessing.quaternConj(qUpdate[:,i]))
            else:
                Angles[:,i] = IMUDataProcessing.GetAnglesFromQuaternion(qUpdate[:,i])
            i += 1

        return np.asarray(Angles).T

    def KalmanGN(acc,gyr,mag,dt = 1/75,R_In=[0.01,0.01,0.01,0.01],beta=.05,conj = True):
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
        https://github.com/tuliofalmeida
        """                 
        acqSize = acc.shape[0]

        # Variance 
        _,varG = IMUDataProcessing.varianceEstimation(gyr)
        var = np.asarray([varG[0]**2,varG[1]**2,varG[2]**2]).T

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
            
            qOsserv[:,i] = IMUDataProcessing.GaussNewtonMethod(qOsserv[:,i-1],AccF[:,i],MagnF[:,i])
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
            # Angles[:,i] = IMUDataProcessing.GetAnglesFromQuaternion(qUpdate[:,i])

            if conj == True:
                Angles[:,i] = IMUDataProcessing.GetAnglesFromQuaternion(IMUDataProcessing.quaternConj(qUpdate[:,i]))
            else:
                Angles[:,i] = IMUDataProcessing.GetAnglesFromQuaternion(qUpdate[:,i])
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
           Fusioned data using the Madgwick filter.

        See Also
        --------
        Developed by S.O.H Madgwick in 27/09/2011
        Adapted for python T.F Almeida in 25/03/2021
        
        For more information see:
        Test scripts
        http://www.x-io.co.uk/node/8#quaternions
        https://github.com/tuliofalmeida
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
        h = IMUDataProcessing.quaternProd(q, IMUDataProcessing.quaternProd(M,IMUDataProcessing.quaternConj(q)))
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
        qDot = 0.5 * IMUDataProcessing.quaternProd(q, G) - Beta * step.T
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
        https://github.com/tuliofalmeida
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

    def MadgwickAHRS(acc,gyr,mag,freq,beta1=.9,beta2=.01):
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
        https://github.com/tuliofalmeida
        """
        madgwick = []
        madgFilter = IMUDataProcessing(Quaternion=np.asarray([[1,0,0,0]]))
        for i in range(len(gyr)):
            q = madgFilter.Madgwick9DOFUpdate(gyr[i],acc[i],mag[i], 1/freq, Beta = beta1)
            madgwick.append(IMUDataProcessing.quatern2euler(IMUDataProcessing.quaternConj(q[0])))

        madgwick = []
        for i in range(len(gyr)):
            q = madgFilter.Madgwick9DOFUpdate(gyr[i],acc[i],mag[i], 1/freq, Beta = beta2)
            madgwick.append(IMUDataProcessing.quatern2euler(IMUDataProcessing.quaternConj(q[0])))

        return np.asarray(madgwick)*180/pi

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
        https://github.com/tuliofalmeida  """
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
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
        https://github.com/tuliofalmeida  """
        acqSize = data.shape[0]
        mean = np.asarray([np.mean(data[0,:]),np.mean(data[1,:]),np.mean(data[2,:])])
        std =  np.asarray([np.std(data[0,:]),np.std(data[1,:]),np.std(data[2,:])])

        return mean,std

    def pattern_extraction(jointAngle,time,threshold,cicle=2,df=True,plot=False,bias=0):
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
           data of time stamp.
        threshold: float
            Point at which the data moves between 
            movements. Example: flexion and extension.
        cicle: int
            Number of points to be considered a pattern.
        df: bool
            Determines whether the input data (jointAngle 
            and time) is on dataframe pandas. If you are 
            not using 'False',then the expected input data
            will be an array.
        plot: bool
            Determines if the function will return the plot.
        bias: int optional
            Value to compensate the cicle adjust.

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
        https://github.com/tuliofalmeida  """ 
        if df == True:
            jointAngle = csvToFloat(jointAngle)
            time = csvToFloat(time)
        if plot == True:
            plt.figure(figsize=(12,9))
        
        diff = low_pass_filter((jointAngle[1:] - jointAngle[:-1]))
        zero_crossings = np.where(np.diff(np.signbit(jointAngle-threshold)))[0]
        Time_Data_Array = []
        mCicloqtd = cicle
        rom = []

        for i in range(int(len(zero_crossings)/(mCicloqtd+1))-1): 
            ciclo = i *(mCicloqtd)+bias
            tempo = 100*((time[zero_crossings[0+ciclo]:zero_crossings[mCicloqtd+ciclo]] - time[zero_crossings[0+ciclo]])/(time[zero_crossings[mCicloqtd+ciclo]]- time[zero_crossings[0+ciclo]]))
            data = jointAngle[zero_crossings[0+ciclo]:zero_crossings[mCicloqtd+ciclo]]
            rom.append(data)

            for j in range(len(tempo)):
                Time_Data_Array.append((tempo[j],data[j]))
    
            if plot == True:
                plt.plot(tempo, jointAngle[zero_crossings[0+ciclo]:zero_crossings[mCicloqtd+ciclo]])

        if plot == True:
            plt.title('Pattern Extraction',fontsize = 20)
            plt.ylabel('Angle (°)',fontsize = 18)
            plt.xlabel('Cicle (%)',fontsize = 18)
            plt.show();

        return np.asarray(Time_Data_Array),np.asarray(rom)

    def patternIC(all_x, all_y,poly_degree = 1,IC = 1.96,df = True,plot=False,figuresize = (12,9), title ='Trajectory IC-95',x_label = 'X Axis',y_label = 'Y Axis'):
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
        IC: float
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
           
        Returns
        -------
        statistics: ndarray
           An array containing the confidence 
           interval, standard deviation, r²,
           Polynomial coefficient, estimated 
           minimum, estimated maximum and 
           data variance.
       
        See Also
        --------
        Developed by T.F Almeida in 25/03/2021

        For more information see:
        https://github.com/tuliofalmeida  """             
        if df == True:
            all_x = csvToFloat(all_x)
            all_y = csvToFloat(all_y)

        coef = np.polyfit(all_x,all_y,poly_degree)
        yest = np.polyval(coef,all_x)

        stdev = np.sqrt(sum((yest - all_y)**2)/(len(all_y)-2))
        confidence_interval = IC*stdev

        r2 = rsquared(all_y,yest)

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
            plt.figure(figsize=figuresize)
            plt.plot(all_x,all_y,'.',color= 'black')
            plt.plot(all_x,yest,'.',color= 'red')
            plt.fill_between(np.sort(all_x),yupsorted, ydownsorted,color='r',alpha=.1)
            plt.title(title).set_size(30)
            plt.xlabel(x_label).set_size(30)
            plt.ylabel(y_label).set_size(30)

        var = stdev**2
        statistics = confidence_interval,stdev,coef,r2,min(yest),max(yest),yest,var

        return statistics
    
    def threshold_ajust(data,time,threshold,x_adjust):  
        """This function plots the data 
        as a function of time and marks 
        the threshold points in the data 
        to facilitate decision making.

        Parameters
        ----------
        data: ndarray
           Joint angle data.
        time: ndarray
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
        https://github.com/tuliofalmeida  """ 
        crossings = np.where(np.diff(np.signbit(data-treshold)))[0]
        plt.figure(figsize=(12,9))
        plt.plot(time,data)
        for i in range(len(crossings)):
            plt.plot(crossings[i]+x_adjust,data[crossings[i]], '.', color = 'red')
        plt.title('Threshold Adjust',fontsize = 20)
        plt.ylabel('Angle (°)',fontsize = 18)
        plt.xlabel('Time (bins)',fontsize = 18)
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
        https://github.com/tuliofalmeida  """ 
        rom = []
        if df == True:
            data = DataHandler.csvToFloat(data)
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
        https://github.com/tuliofalmeida  """ 
        x=0
        for i in range(len(data)):
            x = x + max(data[i])-min(data[i])
        general_mean = x/len(data)
        return general_mean