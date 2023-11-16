#!/usr/bin/env python3

import time
import numpy as np
from low_level_control.gripper_controller import GripperController
# TODO: change it so that we don't copy gripper_controller.py to this faive_control
# Do so by making faive_integration as a python package that is installed in the env, then can be called from anywhere
# faive_integration is NOT a ROS package, it is a python package
from low_level_control.gripper_controller_mujoco_sim import GripperControllerMujocoSim

import rospy
from std_msgs.msg import Float32MultiArray

import argparse


class GripperControlNode:
    def __init__(self, sim=False, sub_queue_size=1) -> None:
        if not sim:
            self.gripper_controller = GripperController("/dev/ttyUSB0")
        else:
            self.gripper_controller = GripperControllerMujocoSim()
        self.gripper_controller.connect_to_dynamixels()
        self.gripper_controller.init_joints(calibrate=False)

        self.commmand_subscriber = rospy.Subscriber(
            '/faive/policy_output', Float32MultiArray, self.write_gripper_angles, queue_size=sub_queue_size)
        
        self.last_received_gc = time.monotonic()
        

    def write_gripper_angles(self, msg):
        unpacked_msg = np.array(msg.data, dtype=np.float32).flatten()
        rospy.loginfo("Received message for GC")
        print(unpacked_msg.shape)
        self.gripper_controller.command_joint_angles(unpacked_msg)
        self.last_received_gc = time.monotonic()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sim', action='store_true')
    parser.add_argument('--sub_queue_size', type=int, default=1)

    args, _ = parser.parse_known_args()

    print(f"GC Subscriber queue size: {args.sub_queue_size}")

    rospy.init_node("gripper_control_node")
    gc_node = GripperControlNode(sim=args.sim, sub_queue_size=args.sub_queue_size)
    r = rospy.Rate(50)
    while not rospy.is_shutdown():
        if time.monotonic() - gc_node.last_received_gc > 3.0:
            gc_node.gripper_controller.command_joint_angles(np.zeros(11,))
        r.sleep()