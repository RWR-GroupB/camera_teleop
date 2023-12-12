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
device = 'cpu'
urdf_path =  base_path + '/../../sim_control/src/'
urdf_filename = urdf_path + "robot-hand-v4"
# convert( urdf_filename+".xml",  urdf_filename+".urdf", asset_file_prefix="../../sim_control/src/")
prev_cwd = os.getcwd()
os.chdir(urdf_path)
chain = pk.build_chain_from_urdf(open(urdf_filename+".xml").read()).to(device='cpuo')
# print( chain)
os.chdir(prev_cwd)

joint_map = torch.zeros(23, 9).to(device)

joint_parameter_names = retarget_utils.JOINT_PARAMETER_NAMES
gc_tendons = retarget_utils.GC_TENDONS
# print(enumerate(gc_tendons.items()))
for i, (name, tendons) in enumerate(gc_tendons.items()):
    joint_map[joint_parameter_names.index(name), i] = 1 if len(tendons) == 0 else 0.5

    for tendon, weight in tendons.items():
        joint_map[joint_parameter_names.index(tendon), i] = weight * 0.5



