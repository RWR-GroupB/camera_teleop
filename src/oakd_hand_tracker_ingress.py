#!/usr/bin/env python3

import argparse
import numpy as np
import cv2
import torch
from scipy.spatial.transform import Rotation as R

# from pydrake.math import RotationMatrix, RigidTransform

import rospy
from geometry_msgs.msg import Point, Transform
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
# from visualization_msgs.msg import Marker
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import tf

import threading
from copy import deepcopy
import time

from utils import retarget_utils

# from oakd_driver.oakd_rgbd_driver import OakDDriver

from depthai_hand_tracker.HandTrackerEdge import HandTracker
from depthai_hand_tracker.Filters import LandmarksSmoothingFilter



class HandPredictionOutput:
    def __init__(self, rh_joints=None, lh_joints=None, rh_reproj=None, lh_reproj=None, frame=None, wrist_pos=None) -> None:
        self.rh_joints = rh_joints
        self.lh_joints = lh_joints
        self.rh_reproj = rh_reproj
        self.lh_reproj = lh_reproj
        self.frame = frame
        self.wrist_pos = wrist_pos

    def add_hand_pred(self, joints, reproj, hand_type):
        if hand_type == 'left_hand':
            self.lh_joints = joints
            self.lh_reproj = reproj
        elif hand_type == 'right_hand':
            self.rh_joints = joints
            self.rh_reproj = reproj
        else:
            raise ValueError(
                'Hand type should be either left_hand or right_hand')

    def add_wrist_pos(self, wrist_pos):
        self.wrist_pos = wrist_pos


class OakDIngress:
    def __init__(self, track_wrist=False) -> None:

        self.track_wrist = track_wrist

        self.tracker = HandTracker(
            solo=True,
            stats=True,
            use_world_landmarks=True,
            xyz=True,
            DEVICE_MXID=None,
        )
        
        self.smoother = LandmarksSmoothingFilter(min_cutoff=1, beta=20, derivate_cutoff=10, disable_value_scaling=True)

        # set up ROS
        self.joint_pub = rospy.Publisher(
            '/ingress/mano', Float32MultiArray, queue_size=10)
        self.img_output_pub = rospy.Publisher(
            '/ingress/processed_img', Image, queue_size=10)
        
        if self.track_wrist:
            self.wrist_positions = np.full((5, 3), np.nan)
            self.wrist_pub = rospy.Publisher(
                '/ingress/wrist', Transform, queue_size=10)

        self.bridge = CvBridge()
        self.tf_broadcaster = tf.TransformBroadcaster()


        self.recorded_mano_joints = []

        self.start_time = time.monotonic()

        rospy.loginfo('OakD Ingress initialized')

    def point_to_pixel(self, pixel_pos, depth):
        if self.inv_intrinsics is not None:
            pixel_pos = np.array([pixel_pos[0], pixel_pos[1], 1])
            pos_3d = self.inv_intrinsics @ pixel_pos * depth / 1000
            return pos_3d

        return None

    def image_received_callback(self):


        res_img, hands, bag = self.tracker.next_frame()
        if len(hands) <= 0: return
        hand_pred = hands[0]

        try:
            output_img = self.bridge.cv2_to_imgmsg(res_img, "bgr8")
        except CvBridgeError as e:
            rospy.logwarn(e)

        self.img_output_pub.publish(output_img)
        
        landmarks = hand_pred.world_landmarks
        smooth_landmarks = self.smoother.apply(landmarks, object_scale=hands[0].rect_w_a)


        normalized_joint_pos = retarget_utils.normalize_points(
            smooth_landmarks, flip_x_axis=False, flip_y_axis=True, add_z_rotation=np.pi/16)

        time_now = time.monotonic()

        if self.track_wrist and time_now - self.start_time > 5.0:

            rotation_matrix = retarget_utils.get_hand_rotation_matrix(smooth_landmarks)
            wrist_position = hand_pred.xyz / 1000.0
            wrist_position[0] = -wrist_position[0]
            wrist_position_y = wrist_position[1]
            wrist_position[1] = -wrist_position[2]
            wrist_position[2] = wrist_position_y
            self.wrist_positions, wrist_position = retarget_utils.rolling_average_filter(self.wrist_positions, wrist_position)

            # if np.linalg.norm(wrist_position) < 0.05:
            #     rospy.logwarn('Wrist position is too close to camera')
            #     return

            # rot_mat = RotationMatrix(rotation_matrix)
            # wrist_transform = RigidTransform(R=rot_mat, p=wrist_position)

            print(f'Wrist position: {wrist_position}')
            # Publish wrist position as a points
            # self.publish_wrist(wrist_transform)
            # self.publish_tf('wrist_transform', wrist_transform)

        # Publish joints
        # self.recorded_mano_joints.append(normalized_joint_pos)

        # print(f'Poses recorded: {len(self.recorded_mano_joints)}')
        # if len(self.recorded_mano_joints) > 2000:
        #     arr = np.array(self.recorded_mano_joints)
        #     np.save('mano_joints_recorded.npy', arr)
        #     print('Saved recorded mano joints')

        self.publish_joints(normalized_joint_pos)

    # def publish_tf(self, frame: str, pose: RigidTransform, from_frame: str = "world"):
    #     '''
    #     Publish a RigidTransform as tf2 message.
    #     '''
    #     xyz = pose.translation().tolist()
    #     quat = pose.rotation().ToQuaternion()
    #     quat_list = [quat.x(), quat.y(), quat.z(), quat.w()]

    #     self.tf_broadcaster.sendTransform(
    #         xyz, quat_list, rospy.Time.now(), frame, from_frame
    #     )

    # def publish_wrist(self, wrist_tf: RigidTransform):
    #     '''
    #     Publish the transform of the wrist as Transform message.
    #     '''
    #     wrist_transform = Transform()
    #     wrist_transform.translation.x = wrist_tf.translation()[0]
    #     wrist_transform.translation.y = wrist_tf.translation()[1]
    #     wrist_transform.translation.z = wrist_tf.translation()[2]

    #     wrist_transform.rotation.x = -wrist_tf.rotation().ToQuaternion().x()
    #     wrist_transform.rotation.y = wrist_tf.rotation().ToQuaternion().y()
    #     wrist_transform.rotation.z = -wrist_tf.rotation().ToQuaternion().z()
    #     wrist_transform.rotation.w = wrist_tf.rotation().ToQuaternion().w()

    #     self.wrist_pub.publish(wrist_transform)

    def publish_joints(self, joint_pos):
        # Publish joints
        arr = joint_pos

        msg = Float32MultiArray()
        msg.data = arr.flatten().tolist()

        # Set the 'layout' field of the message to describe the shape of the original array
        rows_dim = MultiArrayDimension()
        rows_dim.label = 'rows'
        rows_dim.size = arr.shape[0]
        rows_dim.stride = 1

        cols_dim = MultiArrayDimension()
        cols_dim.label = 'cols'
        cols_dim.size = arr.shape[1]
        cols_dim.stride = 1

        msg.layout.dim = [rows_dim, cols_dim]

        self.joint_pub.publish(msg)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--track_wrist', action='store_true')
    args, _ = parser.parse_known_args()

    rospy.init_node('oakd_ingress', anonymous=True)
    oakd_ingress = OakDIngress(track_wrist=args.track_wrist)

    r = rospy.Rate(30)
    while not rospy.is_shutdown():
        oakd_ingress.image_received_callback()
        r.sleep()

    cv2.destroyAllWindows()
