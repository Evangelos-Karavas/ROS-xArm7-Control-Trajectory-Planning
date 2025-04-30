#!/usr/bin/env python3

"""
Trajectory generation for xArm7 path planning
"""

# Ros handlers services and messages
import rospy, roslib
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
#Math imports
from math import pow
import numpy as np


class TrajectoryGen():

    def __init__(self, initialPosition, endingPosition):
        self.initialPosition = initialPosition
        self.endingPosition = endingPosition
        self.Delta = 1.0
        self.t_0 = 0.0
        self.t_f = 10.0
        self.T = 20.0
        self.s_0 = 0.0
        self.s_f = 1.0
        self.V = (self.s_f - self.s_0)/(self.t_f - self.Delta)
        self.a = self.V/self.Delta
        self.s = np.zeros(1, dtype = np.float64)
        self.s_dot = np.zeros(1, dtype = np.float64)
        self.traj_position = np.zeros((6,1), dtype = np.float64)
        self.traj_velocity = np.zeros((6,1), dtype = np.float64)
    
    def setStartEnd(self, initialPosition, endingPosition):
        self.initialPosition = initialPosition
        self.endingPosition = endingPosition
        
    def setPhaseTime(self, Delta):
        self.Delta = Delta
        self.V = (self.s_f - self.s_0)/(self.t_f-self.t_0 - self.Delta)
        self.a = self.V/self.Delta
        
    def setMovementPeriod(self, t_0, t_f):
        self.t_0 = t_0
        self.t_f = t_f
        self.V = (self.s_f - self.s_0)/(self.t_f-self.t_0 - self.Delta)
        self.a = self.V/self.Delta
    
    def polynomialTrajectory(self, t):
        
        if (t <= self.Delta + self.t_0):
           self.s = self.s_0 + (self.a/2)*pow(t - self.t_0,2)
           self.s_dot = self.a*(t - self.t_0)
        
        elif (t > self.t_0 + self.Delta and t <= (self.t_f - self.Delta)):
           self.s = (self.s_0 + self.s_f - self.V*(self.t_f - self.t_0))/2 + self.V*(t - self.t_0)
           self.s_dot = self.V
        
        elif (t <= self.t_f and t > (self.t_f - self.Delta)):
            self.s = self.s_f - self.a*pow(self.t_f - self.t_0,2)/2 + self.a*(self.t_f - self.t_0)*(t - self.t_0) - (self.a/2)*pow(t - self.t_0, 2)
            self.s_dot = self.a*(self.t_f - self.t_0) - self.a*(t - self.t_0)
            
        elif (t > self.t_f):
            print('PROBLEEEEM')
            pass
        
        for i in range(0, 6):
            self.traj_position[i] = self.initialPosition[i,0] + self.s*(self.endingPosition[i] - self.initialPosition[i,0])
            self.traj_velocity[i] = self.s_dot*(self.endingPosition[i,0] - self.initialPosition[i,0])

        
        return self.traj_position.ravel().tolist(), self.traj_velocity.ravel().tolist()
