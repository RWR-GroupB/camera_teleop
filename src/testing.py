#!/usr/bin/env python3
import time
import numpy as np
import torch
from torch.nn.functional import normalize
import os
import pytorch_kinematics as pk
import rospy
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from mjcf_urdf_simple_converter import convert


from utils import retarget_utils, gripper_utils

base_path = os.path.dirname(os.path.realpath(__file__))

urdf_path =  base_path + '/../../sim_control/src/'
urdf_filename = urdf_path + "robot-hand-v4"
# convert(self.urdf_filename+".xml", self.urdf_filename+".urdf", asset_file_prefix="../../sim_control/src/")
prev_cwd = os.getcwd()
os.chdir(urdf_path)
chain = pk.build_chain_from_urdf(open(urdf_filename+".xml").read()).to(device=self.device)
# print(self.chain)
os.chdir(prev_cwd)