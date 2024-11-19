#!/usr/bin/env python3

"""
Compute state space kinematic matrices for xArm7 robot arm (5 links, 7 joints)
"""

import numpy as np
import math
from math import cos, sin, atan2, pow, sqrt

# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

class xArm7_kinematics:
    def __init__(self):
        self.l1 = 0.267
        self.l2 = 0.293
        self.l3 = 0.0525
        self.l4 = 0.3512
        self.l5 = 0.1232

        self.theta1 = 0.2225  # (rad) (=12.75deg)
        self.theta2 = 0.6646  # (rad) (=38.08deg)

    def forward_kinematics(self, joint_angles):
        # Compute the forward kinematics for the xArm7 robot
        # Placeholder: Replace with actual forward kinematics computation
        # For example, using transformation matrices based on DH parameters
        tf_matrix = self.tf_A07(joint_angles)
        # Ensure position is a 1D array
        position = np.array(tf_matrix[0:3, 3]).flatten()
        # Extract orientation and ensure it is a 1D array
        orientation = np.array(self.rotationMatrixToEulerAngles(tf_matrix[0:3, 0:3])).flatten()
        return np.concatenate((position, orientation))
    
    def compute_angles(self, ee_position):
        pos_x = ee_position[0]
        pos_y = ee_position[1]
        pos_z = ee_position[2]

        #These will be static for  inverse kinematics of the first part
        joint_3 = 0
        joint_5 = 0
        joint_6 = 0.75
        joint_7 = 0
        #Inverse Kinematics Solve
        joint_1 = atan2(pos_y, pos_x)
        #Opou M eixame L kai opou p eixame to k
        #We need to solve joint_4 kinematics first because it includes the joint_2
        M = (1/2)*(pow(sqrt(pow(ee_position[0],2) + pow(ee_position[1],2)),2) + pow(ee_position[2] - self.l1, 2) - pow(self.l2, 2) - pow(self.l3, 2)- pow(self.l4, 2)- pow(self.l5, 2) - 2*self.l4*self.l5*cos(joint_6 + self.theta1 - self.theta2))
        p1 = -self.l2*self.l5*cos(self.theta2 - joint_6) + self.l3*self.l5*sin(self.theta2 - joint_6) - self.l2*self.l4*cos(self.theta1) + self.l3*self.l4*sin(self.theta1)
        p2 =  self.l2*self.l5*sin(self.theta2 - joint_6) + self.l3*self.l5*cos(self.theta2 - joint_6) + self.l2*self.l4*sin(self.theta1) + self.l3*self.l4*cos(self.theta1)
        joint_4 = atan2(M, sqrt(pow((sqrt(pow(p1,2) + pow(p2, 2))), 2) - pow(M, 2))) - atan2(p1, p2)

        p3 = self.l2 - (self.l4*cos(self.theta1)+self.l5*cos(joint_6-self.theta2))*cos(joint_4) + (self.l4*sin(self.theta1)-self.l5*sin(joint_6-self.theta2))*sin(joint_4)
        p4 = self.l3 + (self.l4*sin(self.theta1)-self.l5*sin(joint_6-self.theta2))*cos(joint_4) + (self.l4*cos(self.theta1)+self.l5*cos(joint_6-self.theta2))*sin(joint_4)
        joint_2 = atan2(p3*sqrt(pow(ee_position[0],2) + pow(ee_position[1],2))-p4*(ee_position[2]-self.l1),p4*sqrt(pow(ee_position[0],2) + pow(ee_position[1],2))+p3*(ee_position[2]-self.l1))
        
        joint_angles = np.matrix([ joint_1, joint_2, joint_3, joint_4, joint_5, joint_6, joint_7 ])
        return joint_angles
    
    def compute_jacobian(self, r_joints_array):
        l1 = self.l1
        l2 = self.l2
        l3 = self.l3
        l4 = self.l4
        l5 = self.l5
        theta1 = self.theta1
        theta2 = self.theta2
        
        # Placeholder: Replace with actual Jacobian computation
        J_11 = -l3*(cos(r_joints_array[0])*sin(r_joints_array[2]) + cos(r_joints_array[1])*cos(r_joints_array[2])*sin(r_joints_array[0]))\
               -l2*sin(r_joints_array[0])*sin(r_joints_array[1]) \
               -l4*cos(theta1)*(sin(r_joints_array[3])*(cos(r_joints_array[0])*sin(r_joints_array[2]) + cos(r_joints_array[1])*cos(r_joints_array[2])*sin(r_joints_array[0])) - cos(r_joints_array[3])*sin(r_joints_array[0])*sin(r_joints_array[1]))\
               -l4*sin(theta1)*(cos(r_joints_array[3])*(cos(r_joints_array[0])*sin(r_joints_array[2]) + cos(r_joints_array[1])*cos(r_joints_array[2])*sin(r_joints_array[0])) + sin(r_joints_array[0])*sin(r_joints_array[1])*sin(r_joints_array[3]))\
               -l5*cos(theta2)*(cos(r_joints_array[5])*(sin(r_joints_array[3])*(cos(r_joints_array[0])*sin(r_joints_array[2]) + cos(r_joints_array[1])*cos(r_joints_array[2])*sin(r_joints_array[0])) - cos(r_joints_array[3])*sin(r_joints_array[0])*sin(r_joints_array[1])) - sin(r_joints_array[5])*(cos(r_joints_array[5])*(cos(r_joints_array[3])*(cos(r_joints_array[0])*sin(r_joints_array[2]) + cos(r_joints_array[1])*cos(r_joints_array[2])*sin(r_joints_array[0])) + sin(r_joints_array[0])*sin(r_joints_array[1])*sin(r_joints_array[3])) - sin(r_joints_array[4])*(cos(r_joints_array[0])*cos(r_joints_array[2]) - cos(r_joints_array[1])*sin(r_joints_array[0])*sin(r_joints_array[2]))))\
               -l5*sin(theta2)*(sin(r_joints_array[5])*(sin(r_joints_array[3])*(cos(r_joints_array[0])*sin(r_joints_array[2]) + cos(r_joints_array[1])*cos(r_joints_array[2])*sin(r_joints_array[0])) - cos(r_joints_array[3])*sin(r_joints_array[0])*sin(r_joints_array[1])) + cos(r_joints_array[5])*(cos(r_joints_array[4])*(cos(r_joints_array[3])*(cos(r_joints_array[0])*sin(r_joints_array[2]) + cos(r_joints_array[1])*cos(r_joints_array[2])*sin(r_joints_array[0])) + sin(r_joints_array[0])*sin(r_joints_array[1])*sin(r_joints_array[3])) - sin(r_joints_array[4])*(cos(r_joints_array[0])*cos(r_joints_array[2]) - cos(r_joints_array[1])*sin(r_joints_array[0])*sin(r_joints_array[2]))))
        J_12 = l2*cos(r_joints_array[0])*cos(r_joints_array[1])\
             - l5*sin(theta2)*(sin(r_joints_array[5])*(cos(r_joints_array[0])*cos(r_joints_array[1])*cos(r_joints_array[3]) + cos(r_joints_array[0])*cos(r_joints_array[2])*sin(r_joints_array[1])*sin(r_joints_array[3])) - cos(r_joints_array[5])*(cos(r_joints_array[4])*(cos(r_joints_array[0])*cos(r_joints_array[1])*sin(r_joints_array[3]) - cos(r_joints_array[0])*cos(r_joints_array[2])*cos(r_joints_array[3])*sin(r_joints_array[1])) - cos(r_joints_array[0])*sin(r_joints_array[1])*sin(r_joints_array[2])*sin(r_joints_array[4])))\
             - l5*cos(theta2)*(cos(r_joints_array[5])*(cos(r_joints_array[0])*cos(r_joints_array[1])*cos(r_joints_array[3]) + cos(r_joints_array[0])*cos(r_joints_array[2])*sin(r_joints_array[1])*sin(r_joints_array[3])) + sin(r_joints_array[5])*(cos(r_joints_array[4])*(cos(r_joints_array[0])*cos(r_joints_array[1])*sin(r_joints_array[3]) - cos(r_joints_array[0])*cos(r_joints_array[2])*cos(r_joints_array[3])*sin(r_joints_array[1])) - cos(r_joints_array[0])*sin(r_joints_array[1])*sin(r_joints_array[2])*sin(r_joints_array[4])))\
             - l4*cos(theta1)*(cos(r_joints_array[0])*cos(r_joints_array[1])*cos(r_joints_array[3]) + cos(r_joints_array[0])*cos(r_joints_array[2])*sin(r_joints_array[1])*sin(r_joints_array[3]))\
             + l4*sin(theta1)*(cos(r_joints_array[0])*cos(r_joints_array[1])*sin(r_joints_array[3]) - cos(r_joints_array[0])*cos(r_joints_array[3])*cos(r_joints_array[3])*sin(r_joints_array[1])) - l3*cos(r_joints_array[0])*cos(r_joints_array[2])*sin(r_joints_array[1])
        J_13 = l5*cos(theta2)*(sin(r_joints_array[5])*(sin(r_joints_array[4])*(sin(r_joints_array[0])*sin(r_joints_array[2]) - cos(r_joints_array[0])*cos(r_joints_array[1])*cos(r_joints_array[2])) + cos(r_joints_array[3])*cos(r_joints_array[4])*(cos(r_joints_array[2])*sin(r_joints_array[0]) + cos(r_joints_array[0])*cos(r_joints_array[1])*sin(r_joints_array[2]))) - cos(r_joints_array[5])*sin(r_joints_array[3])*(cos(r_joints_array[2])*sin(r_joints_array[0]) + cos(r_joints_array[0])*cos(r_joints_array[1])*sin(r_joints_array[2])))\
             - l3*(cos(r_joints_array[2])*sin(r_joints_array[0]) + cos(r_joints_array[0])*cos(r_joints_array[1])*sin(r_joints_array[2]))\
             - l5*sin(theta2)*(cos(r_joints_array[5])*(sin(r_joints_array[4])*(sin(r_joints_array[0])*sin(r_joints_array[2]) - cos(r_joints_array[0])*cos(r_joints_array[1])*cos(r_joints_array[2])) + cos(r_joints_array[3])*cos(r_joints_array[4])*(cos(r_joints_array[2])*sin(r_joints_array[0]) + cos(r_joints_array[0])*cos(r_joints_array[1])*sin(r_joints_array[2]))) + sin(r_joints_array[3])*sin(r_joints_array[5])*(cos(r_joints_array[2])*sin(r_joints_array[0]) + cos(r_joints_array[0])*cos(r_joints_array[1])*sin(r_joints_array[2])))\
             - l4*cos(r_joints_array[3])*sin(theta1)*(cos(r_joints_array[2])*sin(r_joints_array[0]) + cos(r_joints_array[0])*cos(r_joints_array[1])*sin(r_joints_array[2]))\
             - l4*cos(theta1)*sin(r_joints_array[3])*(cos(r_joints_array[2])*sin(r_joints_array[0]) + cos(r_joints_array[0])*cos(r_joints_array[1])*sin(r_joints_array[2]))
        J_14 = l4*sin(theta1)*(sin(r_joints_array[3])*(sin(r_joints_array[0])*sin(r_joints_array[2]) - cos(r_joints_array[0])*cos(r_joints_array[1])*cos(r_joints_array[2])) + cos(r_joints_array[0])*cos(r_joints_array[3])*sin(r_joints_array[1]))\
             - l5*sin(theta2)*(sin(r_joints_array[5])*(cos(r_joints_array[3])*(sin(r_joints_array[0])*sin(r_joints_array[2]) - cos(r_joints_array[0])*cos(r_joints_array[1])*cos(r_joints_array[2])) - cos(r_joints_array[0])*sin(r_joints_array[1])*sin(r_joints_array[3])) - cos(r_joints_array[4])*cos(r_joints_array[5])*(sin(r_joints_array[3])*(sin(r_joints_array[0])*sin(r_joints_array[2]) - cos(r_joints_array[0])*cos(r_joints_array[1])*cos(r_joints_array[2])) + cos(r_joints_array[0])*cos(r_joints_array[3])*sin(r_joints_array[1])))\
             - l4*cos(theta1)*(cos(r_joints_array[3])*(sin(r_joints_array[0])*sin(r_joints_array[2]) - cos(r_joints_array[0])*cos(r_joints_array[1])*cos(r_joints_array[2])) - cos(r_joints_array[0])*sin(r_joints_array[1])*sin(r_joints_array[3]))\
             - l5*cos(theta2)*(cos(r_joints_array[5])*(cos(r_joints_array[3])*(sin(r_joints_array[0])*sin(r_joints_array[2]) - cos(r_joints_array[0])*cos(r_joints_array[1])*cos(r_joints_array[2])) - cos(r_joints_array[0])*sin(r_joints_array[1])*sin(r_joints_array[3])) + cos(r_joints_array[4])*sin(r_joints_array[5])*(sin(r_joints_array[3])*(sin(r_joints_array[0])*sin(r_joints_array[2]) - cos(r_joints_array[0])*cos(r_joints_array[1])*cos(r_joints_array[2])) + cos(r_joints_array[0])*cos(r_joints_array[3])*sin(r_joints_array[1])))
        J_15 = -l5*sin(r_joints_array[5] - theta2)*(cos(r_joints_array[2])*cos(r_joints_array[4])*sin(r_joints_array[0]) + cos(r_joints_array[0])*cos(r_joints_array[1])*cos(r_joints_array[4])*sin(r_joints_array[2]) - cos(r_joints_array[0])*sin(r_joints_array[1])*sin(r_joints_array[3])*sin(r_joints_array[4]) + cos(r_joints_array[3])*sin(r_joints_array[0])*sin(r_joints_array[2])*sin(r_joints_array[4]) - cos(r_joints_array[0])*cos(r_joints_array[1])*cos(r_joints_array[2])*cos(r_joints_array[3])*sin(r_joints_array[4]))
        J_16 = l5*cos(theta2)*(sin(r_joints_array[5])*(sin(r_joints_array[3])*(sin(r_joints_array[0])*sin(r_joints_array[2]) - cos(r_joints_array[0])*cos(r_joints_array[1])*cos(r_joints_array[2])) + cos(r_joints_array[0])*cos(r_joints_array[3])*sin(r_joints_array[1])) + cos(r_joints_array[5])*(cos(r_joints_array[5])*(cos(r_joints_array[3])*(sin(r_joints_array[0])*sin(r_joints_array[2]) - cos(r_joints_array[0])*cos(r_joints_array[1])*cos(r_joints_array[2])) - cos(r_joints_array[0])*sin(r_joints_array[1])*sin(r_joints_array[3])) - sin(r_joints_array[4])*(cos(r_joints_array[2])*sin(r_joints_array[0]) + cos(r_joints_array[0])*cos(r_joints_array[1])*sin(r_joints_array[2]))))\
             - l5*sin(theta2)*(cos(r_joints_array[5])*(sin(r_joints_array[3])*(sin(r_joints_array[0])*sin(r_joints_array[2]) - cos(r_joints_array[0])*cos(r_joints_array[1])*cos(r_joints_array[2])) + cos(r_joints_array[0])*cos(r_joints_array[3])*sin(r_joints_array[1])) - sin(r_joints_array[5])*(cos(r_joints_array[4])*(cos(r_joints_array[3])*(sin(r_joints_array[0])*sin(r_joints_array[2]) - cos(r_joints_array[0])*cos(r_joints_array[1])*cos(r_joints_array[2])) - cos(r_joints_array[0])*sin(r_joints_array[2])*sin(r_joints_array[3])) - sin(r_joints_array[4])*(cos(r_joints_array[2])*sin(r_joints_array[0]) + cos(r_joints_array[0])*cos(r_joints_array[1])*sin(r_joints_array[2]))))
        J_17 = 0

        J_21 = l2*cos(r_joints_array[0])*sin(r_joints_array[1]) - l3*(sin(r_joints_array[0])*sin(r_joints_array[2]) - cos(r_joints_array[0])*cos(r_joints_array[1])*cos(r_joints_array[2])) - l4*cos(theta1)*(sin(r_joints_array[3])*(sin(r_joints_array[0])*sin(r_joints_array[2]) - cos(r_joints_array[0])*cos(r_joints_array[1])*cos(r_joints_array[2])) + cos(r_joints_array[0])*cos(r_joints_array[3])*sin(r_joints_array[1]))\
              - l5*cos(theta2)*(cos(r_joints_array[5])*(sin(r_joints_array[3])*(sin(r_joints_array[0])*sin(r_joints_array[2]) - cos(r_joints_array[0])*cos(r_joints_array[1])*cos(r_joints_array[2])) + cos(r_joints_array[0])*cos(r_joints_array[3])*sin(r_joints_array[1])) - sin(r_joints_array[5])*(cos(r_joints_array[4])*(cos(r_joints_array[3])*(sin(r_joints_array[0])*sin(r_joints_array[2]) - cos(r_joints_array[0])*cos(r_joints_array[1])*cos(r_joints_array[2])) - cos(r_joints_array[0])*sin(r_joints_array[1])*sin(r_joints_array[3])) - sin(r_joints_array[4])*(cos(r_joints_array[2])*sin(r_joints_array[0]) + cos(r_joints_array[0])*cos(r_joints_array[1])*sin(r_joints_array[2]))))\
              - l4*sin(theta1)*(cos(r_joints_array[3])*(sin(r_joints_array[0])*sin(r_joints_array[2]) - cos(r_joints_array[0])*cos(r_joints_array[1])*cos(r_joints_array[2])) - cos(r_joints_array[0])*sin(r_joints_array[1])*sin(r_joints_array[3]))\
              - l5*sin(theta2)*(sin(r_joints_array[5])*(sin(r_joints_array[3])*(sin(r_joints_array[0])*sin(r_joints_array[2]) - cos(r_joints_array[0])*cos(r_joints_array[1])*cos(r_joints_array[2])) + cos(r_joints_array[0])*cos(r_joints_array[3])*sin(r_joints_array[1])) + cos(r_joints_array[5])*(cos(r_joints_array[4])*(cos(r_joints_array[3])*(sin(r_joints_array[0])*sin(r_joints_array[2]) - cos(r_joints_array[0])*cos(r_joints_array[1])*cos(r_joints_array[2])) - cos(r_joints_array[0])*sin(r_joints_array[1])*sin(r_joints_array[3])) - sin(r_joints_array[4])*(cos(r_joints_array[2])*sin(r_joints_array[0]) + cos(r_joints_array[0])*cos(r_joints_array[1])*sin(r_joints_array[2])))) 
        J_22 = l2*cos(r_joints_array[1])*sin(r_joints_array[0]) - l5*sin(theta2)*(sin(r_joints_array[5])*(cos(r_joints_array[1])*cos(r_joints_array[3])*sin(r_joints_array[0]) + cos(r_joints_array[2])*sin(r_joints_array[0])*sin(r_joints_array[1])*sin(r_joints_array[3])) - cos(r_joints_array[5])*(cos(r_joints_array[4])*(cos(r_joints_array[1])*sin(r_joints_array[0])*sin(r_joints_array[3]) - cos(r_joints_array[2])*cos(r_joints_array[3])*sin(r_joints_array[0])*sin(r_joints_array[1])) - sin(r_joints_array[0])*sin(r_joints_array[1])*sin(r_joints_array[2])*sin(r_joints_array[4])))\
             - l5*cos(theta2)*(cos(r_joints_array[5])*(cos(r_joints_array[1])*cos(r_joints_array[3])*sin(r_joints_array[0]) + cos(r_joints_array[2])*sin(r_joints_array[0])*sin(r_joints_array[1])*sin(r_joints_array[3])) + sin(r_joints_array[5])*(cos(r_joints_array[4])*(cos(r_joints_array[1])*sin(r_joints_array[0])*sin(r_joints_array[3]) - cos(r_joints_array[2])*cos(r_joints_array[3])*sin(r_joints_array[0])*sin(r_joints_array[1])) - sin(r_joints_array[0])*sin(r_joints_array[1])*sin(r_joints_array[2])*sin(r_joints_array[4])))\
             - l4*cos(theta1)*(cos(r_joints_array[1])*cos(r_joints_array[3])*sin(r_joints_array[0]) + cos(r_joints_array[2])*sin(r_joints_array[0])*sin(r_joints_array[1])*sin(r_joints_array[3]))\
             + l4*sin(theta1)*(cos(r_joints_array[1])*sin(r_joints_array[0])*sin(r_joints_array[3]) - cos(r_joints_array[2])*cos(r_joints_array[3])*sin(r_joints_array[0])*sin(r_joints_array[1]))\
             - l3*cos(r_joints_array[2])*sin(r_joints_array[0])*sin(r_joints_array[1])
        J_23 = l3*(cos(r_joints_array[0])*cos(r_joints_array[2]) - cos(r_joints_array[1])*sin(r_joints_array[0])*sin(r_joints_array[2])) - l5*cos(theta2)*(sin(r_joints_array[5])*(sin(r_joints_array[4])*(cos(r_joints_array[0])*sin(r_joints_array[2]) + cos(r_joints_array[1])*cos(r_joints_array[2])*sin(r_joints_array[0])) + cos(r_joints_array[3])*cos(r_joints_array[4])*(cos(r_joints_array[0])*cos(r_joints_array[2]) - cos(r_joints_array[1])*sin(r_joints_array[0])*sin(r_joints_array[2]))) - cos(r_joints_array[5])*sin(r_joints_array[3])*(cos(r_joints_array[0])*cos(r_joints_array[2]) - cos(r_joints_array[1])*sin(r_joints_array[0])*sin(r_joints_array[2])))\
             + l5*sin(theta2)*(cos(r_joints_array[5])*(sin(r_joints_array[4])*(cos(r_joints_array[0])*sin(r_joints_array[2]) + cos(r_joints_array[1])*cos(r_joints_array[2])*sin(r_joints_array[0])) + cos(r_joints_array[3])*cos(r_joints_array[4])*(cos(r_joints_array[0])*cos(r_joints_array[2]) - cos(r_joints_array[1])*sin(r_joints_array[0])*sin(r_joints_array[2]))) + sin(r_joints_array[3])*sin(r_joints_array[5])*(cos(r_joints_array[0])*cos(r_joints_array[2]) - cos(r_joints_array[1])*sin(r_joints_array[0])*sin(r_joints_array[2])))\
             + l4*cos(r_joints_array[3])*sin(theta1)*(cos(r_joints_array[0])*cos(r_joints_array[2]) - cos(r_joints_array[1])*sin(r_joints_array[0])*sin(r_joints_array[2]))\
             + l4*cos(theta1)*sin(r_joints_array[3])*(cos(r_joints_array[0])*cos(r_joints_array[2]) - cos(r_joints_array[1])*sin(r_joints_array[0])*sin(r_joints_array[2]))
        J_24 = l5*cos(theta2)*(cos(r_joints_array[5])*(cos(r_joints_array[3])*(cos(r_joints_array[0])*sin(r_joints_array[2]) + cos(r_joints_array[1])*cos(r_joints_array[2])*sin(r_joints_array[0])) + sin(r_joints_array[0])*sin(r_joints_array[1])*sin(r_joints_array[3])) + cos(r_joints_array[4])*sin(r_joints_array[5])*(sin(r_joints_array[3])*(cos(r_joints_array[0])*sin(r_joints_array[2]) + cos(r_joints_array[1])*cos(r_joints_array[2])*sin(r_joints_array[0])) - cos(r_joints_array[3])*sin(r_joints_array[0])*sin(r_joints_array[1])))\
             + l5*sin(theta2)*(sin(r_joints_array[5])*(cos(r_joints_array[3])*(cos(r_joints_array[0])*sin(r_joints_array[2]) + cos(r_joints_array[1])*cos(r_joints_array[2])*sin(r_joints_array[0])) + sin(r_joints_array[0])*sin(r_joints_array[1])*sin(r_joints_array[3])) - cos(r_joints_array[4])*cos(r_joints_array[5])*(sin(r_joints_array[3])*(cos(r_joints_array[0])*sin(r_joints_array[2]) + cos(r_joints_array[1])*cos(r_joints_array[2])*sin(r_joints_array[0])) - cos(r_joints_array[3])*sin(r_joints_array[0])*sin(r_joints_array[1])))\
             + l4*cos(theta1)*(cos(r_joints_array[3])*(cos(r_joints_array[0])*sin(r_joints_array[2]) + cos(r_joints_array[1])*cos(r_joints_array[2])*sin(r_joints_array[0])) + sin(r_joints_array[0])*sin(r_joints_array[1])*sin(r_joints_array[3]))\
             - l4*sin(theta1)*(sin(r_joints_array[3])*(cos(r_joints_array[0])*sin(r_joints_array[2]) + cos(r_joints_array[1])*cos(r_joints_array[2])*sin(r_joints_array[0])) - cos(r_joints_array[3])*sin(r_joints_array[0])*sin(r_joints_array[1]))
        J_25 = l5*sin(r_joints_array[5] - theta2)*(cos(r_joints_array[0])*cos(r_joints_array[2])*cos(r_joints_array[4]) - cos(r_joints_array[1])*cos(r_joints_array[4])*sin(r_joints_array[0])*sin(r_joints_array[2]) + cos(r_joints_array[0])*cos(r_joints_array[3])*sin(r_joints_array[2])*sin(r_joints_array[4]) + sin(r_joints_array[0])*sin(r_joints_array[1])*sin(r_joints_array[3])*sin(r_joints_array[4]) + cos(r_joints_array[1])*cos(r_joints_array[2])*cos(r_joints_array[3])*sin(r_joints_array[0])*sin(r_joints_array[4]))
        J_26 = l5*sin(theta2)*(cos(r_joints_array[5])*(sin(r_joints_array[3])*(cos(r_joints_array[0])*sin(r_joints_array[2]) + cos(r_joints_array[1])*cos(r_joints_array[2])*sin(r_joints_array[0])) - cos(r_joints_array[3])*sin(r_joints_array[0])*sin(r_joints_array[1])) - sin(r_joints_array[5])*(cos(r_joints_array[4])*(cos(r_joints_array[3])*(cos(r_joints_array[0])*sin(r_joints_array[2]) + cos(r_joints_array[1])*cos(r_joints_array[2])*sin(r_joints_array[0])) + sin(r_joints_array[0])*sin(r_joints_array[1])*sin(r_joints_array[3])) - sin(r_joints_array[4])*(cos(r_joints_array[0])*cos(r_joints_array[2]) - cos(r_joints_array[1])*sin(r_joints_array[0])*sin(r_joints_array[2]))))\
             - l5*cos(theta2)*(sin(r_joints_array[5])*(sin(r_joints_array[3])*(cos(r_joints_array[0])*sin(r_joints_array[2]) + cos(r_joints_array[1])*cos(r_joints_array[2])*sin(r_joints_array[0])) - cos(r_joints_array[3])*sin(r_joints_array[0])*sin(r_joints_array[1])) + cos(r_joints_array[5])*(cos(r_joints_array[4])*(cos(r_joints_array[3])*(cos(r_joints_array[0])*sin(r_joints_array[2]) + cos(r_joints_array[1])*cos(r_joints_array[2])*sin(r_joints_array[0])) + sin(r_joints_array[0])*sin(r_joints_array[1])*sin(r_joints_array[3])) - sin(r_joints_array[4])*(cos(r_joints_array[0])*cos(r_joints_array[2]) - cos(r_joints_array[1])*sin(r_joints_array[0])*sin(r_joints_array[2]))))
        J_27 = 0

        J_31 = 0
        J_32 = l4*cos(theta1)*(cos(r_joints_array[3])*sin(r_joints_array[1]) - cos(r_joints_array[1])*cos(r_joints_array[2])*sin(r_joints_array[3])) - l2*sin(r_joints_array[1])\
             - l4*sin(theta1)*(sin(r_joints_array[1])*sin(r_joints_array[3]) + cos(r_joints_array[1])*cos(r_joints_array[2])*cos(r_joints_array[3])) - l3*cos(r_joints_array[1])*cos(r_joints_array[2])\
             + l5*cos(theta2)*(sin(r_joints_array[5])*(cos(r_joints_array[4])*(sin(r_joints_array[1])*sin(r_joints_array[3]) + cos(r_joints_array[1])*cos(r_joints_array[2])*cos(r_joints_array[3])) + cos(r_joints_array[1])*sin(r_joints_array[2])*sin(r_joints_array[4])) + cos(r_joints_array[5])*(cos(r_joints_array[3])*sin(r_joints_array[1]) - cos(r_joints_array[1])*cos(r_joints_array[2])*sin(r_joints_array[3])))\
             - l5*sin(theta2)*(cos(r_joints_array[5])*(cos(r_joints_array[4])*(sin(r_joints_array[1])*sin(r_joints_array[3]) + cos(r_joints_array[1])*cos(r_joints_array[2])*cos(r_joints_array[3])) + cos(r_joints_array[1])*sin(r_joints_array[2])*sin(r_joints_array[4])) - sin(r_joints_array[5])*(cos(r_joints_array[3])*sin(r_joints_array[1]) - cos(r_joints_array[1])*cos(r_joints_array[2])*sin(r_joints_array[3])))
        J_33 = l4*cos(theta1)*(cos(r_joints_array[3])*sin(r_joints_array[1]) - cos(r_joints_array[1])*cos(r_joints_array[2])*sin(r_joints_array[3])) - l2*sin(r_joints_array[1])\
             - l4*sin(theta1)*(sin(r_joints_array[1])*sin(r_joints_array[3]) + cos(r_joints_array[1])*cos(r_joints_array[2])*cos(r_joints_array[3])) - l3*cos(r_joints_array[1])*cos(r_joints_array[2])\
             + l5*cos(theta2)*(sin(r_joints_array[5])*(cos(r_joints_array[4])*(sin(r_joints_array[1])*sin(r_joints_array[3]) + cos(r_joints_array[1])*cos(r_joints_array[2])*cos(r_joints_array[3])) + cos(r_joints_array[1])*sin(r_joints_array[2])*sin(r_joints_array[4])) + cos(r_joints_array[5])*(cos(r_joints_array[3])*sin(r_joints_array[1]) - cos(r_joints_array[1])*cos(r_joints_array[2])*sin(r_joints_array[3])))\
             - l5*sin(theta2)*(cos(r_joints_array[5])*(cos(r_joints_array[4])*(sin(r_joints_array[1])*sin(r_joints_array[3]) + cos(r_joints_array[1])*cos(r_joints_array[2])*cos(r_joints_array[3])) + cos(r_joints_array[1])*sin(r_joints_array[2])*sin(r_joints_array[4])) - sin(r_joints_array[5])*(cos(r_joints_array[3])*sin(r_joints_array[1]) - cos(r_joints_array[1])*cos(r_joints_array[2])*sin(r_joints_array[3])))
        J_34 = l4*cos(theta1)*(cos(r_joints_array[1])*sin(r_joints_array[3]) - cos(r_joints_array[2])*cos(r_joints_array[3])*sin(r_joints_array[1]))\
             + l4*sin(theta1)*(cos(r_joints_array[1])*cos(r_joints_array[3]) + cos(r_joints_array[2])*sin(r_joints_array[1])*sin(r_joints_array[3]))\
             + l5*cos(theta2)*(cos(r_joints_array[5])*(cos(r_joints_array[1])*sin(r_joints_array[3]) - cos(r_joints_array[2])*cos(r_joints_array[3])*sin(r_joints_array[1])) - cos(r_joints_array[4])*sin(r_joints_array[5])*(cos(r_joints_array[1])*cos(r_joints_array[3]) + cos(r_joints_array[2])*sin(r_joints_array[1])*sin(r_joints_array[3])))\
             + l5*sin(theta2)*(sin(r_joints_array[5])*(cos(r_joints_array[1])*sin(r_joints_array[3]) - cos(r_joints_array[2])*cos(r_joints_array[3])*sin(r_joints_array[1])) + cos(r_joints_array[4])*cos(r_joints_array[5])*(cos(r_joints_array[1])*cos(r_joints_array[3]) + cos(r_joints_array[2])*sin(r_joints_array[1])*sin(r_joints_array[3])))
        J_35 = l5*sin(r_joints_array[5] - theta2)*(cos(r_joints_array[4])*sin(r_joints_array[1])*sin(r_joints_array[2]) + cos(r_joints_array[1])*sin(r_joints_array[3])*sin(r_joints_array[4]) - cos(r_joints_array[2])*cos(r_joints_array[3])*sin(r_joints_array[1])*sin(r_joints_array[4]))
        J_36 = -l5*cos(theta2)*(cos(r_joints_array[5])*(cos(r_joints_array[4])*(cos(r_joints_array[1])*sin(r_joints_array[3]) - cos(r_joints_array[2])*cos(r_joints_array[3])*sin(r_joints_array[1])) - sin(r_joints_array[1])*sin(r_joints_array[2])*sin(r_joints_array[4])) - sin(r_joints_array[5])*(cos(r_joints_array[1])*cos(r_joints_array[3]) + cos(r_joints_array[2])*sin(r_joints_array[1])*sin(r_joints_array[3])))\
               -l5*sin(theta2)*(sin(r_joints_array[5])*(cos(r_joints_array[4])*(cos(r_joints_array[1])*sin(r_joints_array[3]) - cos(r_joints_array[2])*cos(r_joints_array[3])*sin(r_joints_array[1])) - sin(r_joints_array[1])*sin(r_joints_array[2])*sin(r_joints_array[4])) + cos(r_joints_array[5])*(cos(r_joints_array[1])*cos(r_joints_array[3]) + cos(r_joints_array[2])*sin(r_joints_array[1])*sin(r_joints_array[3])))
        J_37 = 0

        J_41 = 0
        J_42 = -sin(r_joints_array[0])
        J_43 = cos(r_joints_array[0])*sin(r_joints_array[1])
        J_44 = cos(r_joints_array[2])*sin(r_joints_array[0])+ cos(r_joints_array[0])*cos(r_joints_array[1])*sin(r_joints_array[2])
        J_45 = -sin(r_joints_array[3])*(sin(r_joints_array[0])*sin(r_joints_array[2]) - cos(r_joints_array[0])*cos(r_joints_array[1])*cos(r_joints_array[2])) - cos(r_joints_array[0])*cos(r_joints_array[3])*sin(r_joints_array[1])
        J_46 = -sin(r_joints_array[4])*(cos(r_joints_array[3])*(sin(r_joints_array[0])*sin(r_joints_array[2]) - cos(r_joints_array[0])*cos(r_joints_array[1])*cos(r_joints_array[2])) - cos(r_joints_array[0])*sin(r_joints_array[1])*sin(r_joints_array[3])) - cos(r_joints_array[4])*(cos(r_joints_array[2])*sin(r_joints_array[0]) + cos(r_joints_array[0])*cos(r_joints_array[1])*sin(r_joints_array[2]))
        J_47 = sin(r_joints_array[5])*(cos(r_joints_array[4])*(cos(r_joints_array[3])*(sin(r_joints_array[0])*sin(r_joints_array[2]) - cos(r_joints_array[0])*cos(r_joints_array[1])*cos(r_joints_array[2])) - cos(r_joints_array[0])*sin(r_joints_array[1])*sin(r_joints_array[3])) - sin(r_joints_array[4])*(cos(r_joints_array[2])*sin(r_joints_array[0]) + cos(r_joints_array[0])*cos(r_joints_array[1])*sin(r_joints_array[2]))) - cos(r_joints_array[5])*(sin(r_joints_array[3])*(sin(r_joints_array[0])*sin(r_joints_array[2]) - cos(r_joints_array[0])*cos(r_joints_array[1])*cos(r_joints_array[2])) + cos(r_joints_array[0])*cos(r_joints_array[3])*sin(r_joints_array[1]))
        
        J_51 = 0
        J_52 = cos(r_joints_array[0])
        J_53 = sin(r_joints_array[0])*sin(r_joints_array[1])
        J_54 = cos(r_joints_array[1])*sin(r_joints_array[0])*sin(r_joints_array[2]) - cos(r_joints_array[0])*cos(r_joints_array[2])
        J_55 = sin(r_joints_array[3])*(cos(r_joints_array[0])*sin(r_joints_array[2]) + cos(r_joints_array[1])*cos(r_joints_array[2])*sin(r_joints_array[0])) - cos(r_joints_array[3])*sin(r_joints_array[0])*sin(r_joints_array[1])
        J_56 =  sin(r_joints_array[4])*(cos(r_joints_array[3])*(cos(r_joints_array[0])*sin(r_joints_array[2]) + cos(r_joints_array[1])*cos(r_joints_array[2])*sin(r_joints_array[0])) + sin(r_joints_array[0])*sin(r_joints_array[1])*sin(r_joints_array[3])) + cos(r_joints_array[4])*(cos(r_joints_array[0])*cos(r_joints_array[2]) - cos(r_joints_array[1])*sin(r_joints_array[0])*sin(r_joints_array[2]))
        J_57 = cos(r_joints_array[5])*(sin(r_joints_array[3])*(cos(r_joints_array[0])*sin(r_joints_array[2]) + cos(r_joints_array[1])*cos(r_joints_array[2])*sin(r_joints_array[0])) - cos(r_joints_array[3])*sin(r_joints_array[0])*sin(r_joints_array[1])) - sin(r_joints_array[5])*(cos(r_joints_array[4])*(cos(r_joints_array[3])*(cos(r_joints_array[0])*sin(r_joints_array[2]) + cos(r_joints_array[1])*cos(r_joints_array[2])*sin(r_joints_array[0])) + sin(r_joints_array[0])*sin(r_joints_array[1])*sin(r_joints_array[3])) - sin(r_joints_array[4])*(cos(r_joints_array[0])*cos(r_joints_array[2]) - cos(r_joints_array[1])*sin(r_joints_array[0])*sin(r_joints_array[2])))
        
        J_61 = 1 
        J_62 = 0 
        J_63 = cos(r_joints_array[1])
        J_64 = -sin(r_joints_array[1])*sin(r_joints_array[2])
        J_65 = - cos(r_joints_array[1])*cos(r_joints_array[3]) - cos(r_joints_array[2])*sin(r_joints_array[1])*sin(r_joints_array[3])
        J_66 =  sin(r_joints_array[4])*(cos(r_joints_array[1])*sin(r_joints_array[3]) - cos(r_joints_array[2])*cos(r_joints_array[3])*sin(r_joints_array[1])) + cos(r_joints_array[4])*sin(r_joints_array[1])*sin(r_joints_array[2])
        J_67 = - sin(r_joints_array[5])*(cos(r_joints_array[4])*(cos(r_joints_array[1])*sin(r_joints_array[3]) - cos(r_joints_array[2])*cos(r_joints_array[3])*sin(r_joints_array[1])) - sin(r_joints_array[1])*sin(r_joints_array[2])*sin(r_joints_array[4])) - cos(r_joints_array[5])*(cos(r_joints_array[1])*cos(r_joints_array[3]) + cos(r_joints_array[2])*sin(r_joints_array[1])*sin(r_joints_array[3]))
        

        J = np.matrix([ [ J_11 , J_12 , J_13 , J_14 , J_15 , J_16 , J_17 ],\
                        [ J_21 , J_22 , J_23 , J_24 , J_25 , J_26 , J_27 ],\
                        [ J_31 , J_32 , J_33 , J_34 , J_35 , J_36 , J_37 ],\
                        [ J_41 , J_42 , J_43 , J_44 , J_45 , J_46 , J_47 ],\
                        [ J_51 , J_52 , J_53 , J_54 , J_55 , J_56 , J_57 ],\
                        [ J_61 , J_62 , J_63 , J_64 , J_65 , J_66 , J_67 ]])
        return J

    def tf_A01(self, r_joints_array):
        tf = np.matrix([[cos(r_joints_array[0]), -sin(r_joints_array[0]), 0, 0],\
                        [sin(r_joints_array[0]),  cos(r_joints_array[0]), 0, 0],\
                        [0, 0, 1, self.l1],\
                        [0, 0, 0, 1]])
        return tf

    def tf_A02(self, r_joints_array):
        tf_A12 = np.matrix([[cos(r_joints_array[1]), -sin(r_joints_array[1]), 0, 0],\
                            [0, 0, 1, 0],\
                            [-sin(r_joints_array[1]), -cos(r_joints_array[1]), 0, 0],\
                            [0, 0 ,0, 1]])
        tf = np.dot( self.tf_A01(r_joints_array), tf_A12 )
        return tf

    def tf_A03(self, r_joints_array):
        tf_A23 = np.matrix([[cos(r_joints_array[2]), -sin(r_joints_array[2]), 0, 0],\
                            [0, 0, -1, -self.l2],\
                            [sin(r_joints_array[2]), cos(r_joints_array[2]), 0, 0],\
                            [0, 0, 0, 1]])
        tf = np.dot( self.tf_A02(r_joints_array), tf_A23 )
        return tf

    def tf_A04(self, r_joints_array):
        tf_A34 = np.matrix([[cos(r_joints_array[3]), -sin(r_joints_array[3]), 0, self.l3],\
                            [0, 0, -1, 0],\
                            [sin(r_joints_array[3]), cos(r_joints_array[3]), 0, 0],\
                            [0, 0, 0, 1]])
        tf = np.dot( self.tf_A03(r_joints_array), tf_A34 )
        return tf

    def tf_A05(self, r_joints_array):
        tf_A45 = np.matrix([[cos(r_joints_array[4]), -sin(r_joints_array[4]), 0, self.l4*sin(self.theta1)],\
                            [0, 0, -1, -self.l4*cos(self.theta1)],\
                            [sin(r_joints_array[4]), cos(r_joints_array[4]), 0, 0],\
                            [0, 0, 0, 1]])
        tf = np.dot( self.tf_A04(r_joints_array), tf_A45 )
        return tf

    def tf_A06(self, r_joints_array):
        tf_A56 = np.matrix([[cos(r_joints_array[5]), -sin(r_joints_array[5]), 0, 0],\
                            [0, 0, -1, 0],\
                            [sin(r_joints_array[5]), cos(r_joints_array[5]), 0, 0],\
                            [0, 0, 0, 1]])
        tf = np.dot( self.tf_A05(r_joints_array), tf_A56 )
        return tf

    def tf_A07(self, r_joints_array):
        tf_A67 = np.matrix([[cos(r_joints_array[6]), -sin(r_joints_array[6]), 0, self.l5*sin(self.theta2)],\
                            [0, 0, 1, self.l5*cos(self.theta2)],\
                            [-sin(r_joints_array[6]), -cos(r_joints_array[6]), 0, 0],\
                            [0, 0, 0, 1]])
        tf = np.dot( self.tf_A06(r_joints_array), tf_A67 )
        return tf

    def rotationMatrixToEulerAngles(self, R):
        assert (isRotationMatrix(R))
        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6
        if not singular:
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0
        return np.array([x, y, z])
