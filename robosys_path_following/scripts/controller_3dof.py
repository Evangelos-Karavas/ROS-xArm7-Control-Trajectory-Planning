#!/usr/bin/env python

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

        # Topics for initialize end effector position and orientation
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

        point_A = [0.6043, 0.2, 0.1508]
        point_B = [0.6043, -0.2, 0.1508]

        # Define the total time for the movement
        total_time = 10.0  # in seconds, you can adjust this
        start_time = rospy.get_time()  # get start time
        while not rospy.is_shutdown():

            self.A01 = self.kinematics.tf_A01(self.joint_angpos)

            current_time = rospy.get_time()
            elapsed_time = current_time - start_time

            t_normalized = (elapsed_time % total_time) / total_time

            # Calculating the y position using a sine wave for smooth periodic movement
            if t_normalized <= 0.5:
                # Moving from A to B
                phase_normalized = t_normalized * 2  
                current_y = point_A[1] + (point_B[1] - point_A[1]) * 0.5 * (1 - np.cos(np.pi * phase_normalized))
            else:
                # Moving from B to A
                phase_normalized = (t_normalized - 0.5) * 2 
                current_y = point_B[1] + (point_A[1] - point_B[1]) * 0.5 * (1 - np.cos(np.pi * phase_normalized))

            # Linear interpolation for y
            ee_position = [point_A[0], current_y, point_A[2]]

            # Compute joint angles using inverse kinematics
            joint_angles = self.kinematics.compute_angles(ee_position)
            self.joint_angpos = joint_angles.tolist()[0]

            # Publish the new joint's angular positions
            self.joint1_pos_pub.publish(self.joint_angpos[0])
            self.joint2_pos_pub.publish(self.joint_angpos[1])
            self.joint4_pos_pub.publish(self.joint_angpos[3])

            self.end_effector_pos_x_pub.publish(ee_position[0])
            self.end_effector_pos_y_pub.publish(ee_position[1])
            self.end_effector_pos_z_pub.publish(ee_position[2])

            print("XYZ Position: ", ee_position)
            print("q1,q2,q4", self.joint_angpos[0], self.joint_angpos[1], self.joint_angpos[3])

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
