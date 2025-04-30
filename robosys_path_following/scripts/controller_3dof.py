#!/usr/bin/env python3

"""
Start ROS node to publish angles for the position control of the xArm7.
"""

# Ros handlers services and messages
import rospy, roslib
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
#Math imports
from math import sin, cos, atan2, pi, sqrt
from numpy.linalg import inv, det, norm, pinv
import numpy as np
import time as t
import trajectory as tg
# Arm parameters
# xArm7 kinematics class
from kinematics import xArm7_kinematics

# from tf.transformations import quaternion_matrix
# matrix = quaternion_matrix([1, 0, 0, 0])

class xArm7_controller():
    """Class to compute and publish joints positions"""
    def __init__(self,rate):

        # Init xArm7 kinematics handler
        self.kinematics = xArm7_kinematics()

        # joints' angular positions
        self.joint_angpos = [0, 0, 0, 0, 0, 0, 0]
        # joints' states
        self.joint_states = JointState()
        # joints' transformation matrix wrt the robot's base frame
        self.A01 = self.kinematics.tf_A01(self.joint_angpos)
        self.A02 = self.kinematics.tf_A02(self.joint_angpos)
        self.A03 = self.kinematics.tf_A03(self.joint_angpos)
        self.A04 = self.kinematics.tf_A04(self.joint_angpos)
        self.A05 = self.kinematics.tf_A05(self.joint_angpos)
        self.A06 = self.kinematics.tf_A06(self.joint_angpos)
        self.A07 = self.kinematics.tf_A07(self.joint_angpos)

        # ROS SETUP
        # initialize subscribers for reading encoders and publishers for performing position control in the joint-space
        # Robot
        self.joint_states_sub = rospy.Subscriber('/xarm/joint_states', JointState, self.joint_states_callback, queue_size=1)
        self.joint1_pos_pub = rospy.Publisher('/xarm/joint1_position_controller/command', Float64, queue_size=1)
        self.joint2_pos_pub = rospy.Publisher('/xarm/joint2_position_controller/command', Float64, queue_size=1)
        self.joint3_pos_pub = rospy.Publisher('/xarm/joint3_position_controller/command', Float64, queue_size=1)
        self.joint4_pos_pub = rospy.Publisher('/xarm/joint4_position_controller/command', Float64, queue_size=1)
        self.joint5_pos_pub = rospy.Publisher('/xarm/joint5_position_controller/command', Float64, queue_size=1)
        self.joint6_pos_pub = rospy.Publisher('/xarm/joint6_position_controller/command', Float64, queue_size=1)
        self.joint7_pos_pub = rospy.Publisher('/xarm/joint7_position_controller/command', Float64, queue_size=1)
        
        self.end_effector_pos_x_pub = rospy.Publisher('/xarm/ee_position_x', Float64, queue_size=1)
        self.end_effector_pos_y_pub = rospy.Publisher('/xarm/ee_position_y', Float64, queue_size=1)
        self.end_effector_pos_z_pub = rospy.Publisher('/xarm/ee_position_z', Float64, queue_size=1)

        #Publishing rate
        self.period = 1.0/rate
        self.pub_rate = rospy.Rate(rate)

        self.publish()

    #SENSING CALLBACKS
    def joint_states_callback(self, msg):
        # ROS callback to get the joint_states

        self.joint_states = msg
        # (e.g. the angular position of joint 1 is stored in :: self.joint_states.position[0])

    def publish(self):

        # set configuration
        self.joint_angpos = [0, 0.75, 0, 1.5, 0, 0.75, 0]
        tmp_rate = rospy.Rate(1)
        tmp_rate.sleep()
        self.joint4_pos_pub.publish(self.joint_angpos[3])
        tmp_rate.sleep()
        self.joint2_pos_pub.publish(self.joint_angpos[1])
        self.joint6_pos_pub.publish(self.joint_angpos[5])
        tmp_rate.sleep()
        print("The system is ready to execute your algorithm...")

        P1 = np.matrix([0.6043, -0.2000, 0.1508, pi, 0, 0]).reshape(6,1)
        P2 = np.matrix([0.6043, 0.2000, 0.1508, pi, 0, 0]).reshape(6,1)        
        A07_real = self.kinematics.tf_A07(self.joint_states.position)
        
        print("Cobot starting position")
        start_pos = A07_real[0:3,3]
        print(start_pos)
        
        Ps = np.concatenate([start_pos, np.zeros((3,1), dtype = np.float64)])
        
        traj = tg.TrajectoryGen(P1, P2)
        rostime_now = rospy.Time.now()
        t0 = rostime_now.to_sec()
        start = True
        p_desired = Ps.ravel().tolist()[0]
        
        while not rospy.is_shutdown():
            
            # Compute each transformation matrix wrt the base frame from joints' angular positions
            self.A07 = self.kinematics.tf_A07(self.joint_angpos)
            A07_real = self.kinematics.tf_A07(self.joint_states.position)
            ee_pos = A07_real[0:3,3]
            
            # Starting the Trajectory All Over Again
            if start:
                print("Initial state")
                start = False
                Pf = P1
                T = 5.0
                t0 = rostime_now.to_sec() 
            
            elif (abs(p_desired[1] - P1[1,0]) < 1e-4 and Pf[1,0] == P1[1,0]):
                print("End Effector is in point A!")
                print("Inside path AB")
                Ps = P1
                Pf = P2
                T = 10.0
                t0 = time
            elif (abs(p_desired[1] - P2[1,0]) < 1e-4 and Pf[1,0] == P2[1,0]):
                print("End Effector is in point B!")
                print("Inside path BA")
                Ps = P2
                Pf = P1
                T = 10.0
                t0 = time

            # Setting Trajectory Parameters based on Starting and Final Position
            print("Set trajectory motion based on new points!")
            traj.setStartEnd(Ps, Pf)
            print("t0 = ", t0)
            print("T = ", T)
            traj.setMovementPeriod(t0,t0 + T)
            
            # Trajectory Execution
            print("time is: ", rostime_now.to_sec())
            p_desired, _ = traj.polynomialTrajectory(rostime_now.to_sec())
            p_desired = p_desired[0:3]
            print("Desired position trajectory: \n", p_desired)
            
            joint_angles = self.kinematics.compute_angles(p_desired)
            self.joint_angpos[0] = joint_angles[0,0]
            self.joint_angpos[1] = joint_angles[0,1]
            self.joint_angpos[3] = joint_angles[0,3]
            print("Angular Position: \n", self.joint_angpos)
            
            rostime_now = rospy.Time.now()
            time = rostime_now.to_sec()

            # Publish the new joint's angular positions
            self.joint1_pos_pub.publish(self.joint_angpos[0])
            self.joint2_pos_pub.publish(self.joint_angpos[1])
            #self.joint3_pos_pub.publish(self.joint_angpos[2])
            self.joint4_pos_pub.publish(self.joint_angpos[3])
            #self.joint5_pos_pub.publish(self.joint_angpos[4])
            #self.joint6_pos_pub.publish(self.joint_angpos[5])
            #self.joint7_pos_pub.publish(self.joint_angpos[6])
            
            self.end_effector_pos_x_pub.publish(ee_pos[0])
            self.end_effector_pos_y_pub.publish(ee_pos[1])
            self.end_effector_pos_z_pub.publish(ee_pos[2])
            
            self.pub_rate.sleep()

    def turn_off(self):
        pass

def controller_py():
    # Starts a new node
    rospy.init_node('controller_node', anonymous=True)
    # Reading parameters set in launch file
    rate = rospy.get_param("/rate")

    controller = xArm7_controller(rate)
    rospy.on_shutdown(controller.turn_off)
    rospy.spin()

if __name__ == '__main__':
    try:
        controller_py()
    except rospy.ROSInterruptException:
        pass
