#!/usr/bin/env python3

# Code copied from main depthai repo, depthai_helpers/projector_3d.py

import depthai as dai
import cv2
import time
import open3d as o3d
import threading
from oakd_driver.depth_image_utils import PointCloudVisualizer
# from depth_image_utils import PointCloudVisualizer
import numpy as np

#!/usr/bin/env python3

COLOR = True

lrcheck = True  # Better handling for occlusions
extended = False  # Closer-in minimum depth, disparity range is doubled
subpixel = True  # Better accuracy for longer distance, fractional disparity 32-levels
# Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7
median = dai.StereoDepthProperties.MedianFilter.KERNEL_7x7

print("StereoDepth config options:")
print("    Left-Right check:  ", lrcheck)
print("    Extended disparity:", extended)
print("    Subpixel:          ", subpixel)
print("    Median filtering:  ", median)

pipeline = dai.Pipeline()

monoLeft = pipeline.create(dai.node.MonoCamera)
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)

monoRight = pipeline.create(dai.node.MonoCamera)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

stereo = pipeline.createStereoDepth()
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo.initialConfig.setMedianFilter(median)
# stereo.initialConfig.setConfidenceThreshold(255)

stereo.setLeftRightCheck(lrcheck)
stereo.setExtendedDisparity(extended)
stereo.setSubpixel(subpixel)
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

config = stereo.initialConfig.get()
config.postProcessing.speckleFilter.enable = False
config.postProcessing.speckleFilter.speckleRange = 50
config.postProcessing.temporalFilter.enable = True
config.postProcessing.spatialFilter.enable = True
config.postProcessing.spatialFilter.holeFillingRadius = 2
config.postProcessing.spatialFilter.numIterations = 1
config.postProcessing.thresholdFilter.minRange = 400
config.postProcessing.thresholdFilter.maxRange = 200000
config.postProcessing.decimationFilter.decimationFactor = 1
stereo.initialConfig.set(config)

xout_depth = pipeline.createXLinkOut()
xout_depth.setStreamName("depth")
stereo.depth.link(xout_depth.input)

# xout_disparity = pipeline.createXLinkOut()
# xout_disparity.setStreamName('disparity')
# stereo.disparity.link(xout_disparity.input)

xout_colorize = pipeline.createXLinkOut()
xout_colorize.setStreamName("colorize")
xout_rect_left = pipeline.createXLinkOut()
xout_rect_left.setStreamName("rectified_left")
xout_rect_right = pipeline.createXLinkOut()
xout_rect_right.setStreamName("rectified_right")

if COLOR:
    camRgb = pipeline.create(dai.node.ColorCamera)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camRgb.setIspScale(1, 3)
    camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
    camRgb.initialControl.setManualFocus(130)
    stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
    camRgb.isp.link(xout_colorize.input)
else:
    stereo.rectifiedRight.link(xout_colorize.input)

stereo.rectifiedLeft.link(xout_rect_left.input)
stereo.rectifiedRight.link(xout_rect_right.input)


class HostSync:
    def __init__(self):
        self.arrays = {}

    def add_msg(self, name, msg):
        if not name in self.arrays:
            self.arrays[name] = []
        # Add msg to array
        self.arrays[name].append({"msg": msg, "seq": msg.getSequenceNum()})

        synced = {}
        for name, arr in self.arrays.items():
            for i, obj in enumerate(arr):
                if msg.getSequenceNum() == obj["seq"]:
                    synced[name] = obj["msg"]
                    break
        # If there are 5 (all) synced msgs, remove all old msgs
        # and return synced msgs
        if len(synced) == 4:  # color, left, right, depth, nn
            # Remove old msgs
            for name, arr in self.arrays.items():
                for i, obj in enumerate(arr):
                    if obj["seq"] < msg.getSequenceNum():
                        arr.remove(obj)
                    else:
                        break
            return synced
        return False


class OakDDriver():
    TOP_CAMERA = '1944301021F54C1300'
    FRANKMOCAP_CAMERA = '18443010310CC71200'

    def __init__(self, callback, visualize=True, device_mxid=None) -> None:
        print(f'Using OAK-D device {device_mxid}')
        print(f'Available devices: {dai.Device.getAllAvailableDevices()}')

        self.visualize = visualize
        self.callback = callback
        self.device_mxid = device_mxid
        self.thread = threading.Thread(target=self.run_thread)
        self.thread.daemon = True
        self.thread.start()
        self.intrinsics = None
        self.inv_intrinsics = None
        self.distortion_coeff = None

    def print_devices(self):
        for device in dai.Device.getAllAvailableDevices():
            print(f"{device.getMxId()} {device.state}")

    def run_thread(self):
        if self.device_mxid is None:
            device = dai.Device(pipeline)
        else:
            device_info = dai.DeviceInfo(self.device_mxid)
            device = dai.Device(pipeline, device_info)
        
        self.run(device)

    def run(self, device):
        with device:

            device.setIrLaserDotProjectorBrightness(1200)
            qs = []
            qs.append(device.getOutputQueue(
                "depth", maxSize=1, blocking=False))
            qs.append(device.getOutputQueue(
                "colorize", maxSize=1, blocking=False))
            qs.append(device.getOutputQueue(
                "rectified_left", maxSize=1, blocking=False))
            qs.append(device.getOutputQueue(
                "rectified_right", maxSize=1, blocking=False))

            calibData = device.readCalibration()
            if COLOR:
                w, h = camRgb.getIspSize()
                intrinsics = calibData.getCameraIntrinsics(
                    dai.CameraBoardSocket.RGB, dai.Size2f(w, h))
                
                distortion_coeff = np.array(calibData.getDistortionCoefficients(dai.CameraBoardSocket.RGB))
            
            else:
                w, h = monoRight.getResolutionSize()
                intrinsics = calibData.getCameraIntrinsics(
                    dai.CameraBoardSocket.RIGHT, dai.Size2f(w, h))
                distortion_coeff = np.array(calibData.getDistortionCoefficients(dai.CameraBoardSocket.RIGHT))
            

            self.intrinsics = intrinsics
            self.inv_intrinsics = np.linalg.inv(intrinsics)
            self.distortion_coeff = distortion_coeff

            pcl_converter = PointCloudVisualizer(
                intrinsics, w, h, self.visualize)

            serial_no = device.getMxId()
            sync = HostSync()
            depth_vis, color, rect_left, rect_right = None, None, None, None

            while True:
                for q in qs:
                    new_msg = q.tryGet()
                    if new_msg is not None:
                        msgs = sync.add_msg(q.getName(), new_msg)
                        if msgs:
                            depth = msgs["depth"].getFrame()
                            color = msgs["colorize"].getCvFrame()

                            if self.visualize:
                                rectified_left = msgs["rectified_left"].getCvFrame(
                                )
                                rectified_right = msgs["rectified_right"].getCvFrame(
                                )
                                depth_vis = cv2.normalize(
                                    depth, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
                                depth_vis = cv2.equalizeHist(depth_vis)
                                depth_vis = cv2.applyColorMap(
                                    depth_vis, cv2.COLORMAP_HOT)
                                cv2.imshow("depth", depth_vis)
                                cv2.imshow("color", color)
                                cv2.imshow("rectified_left", rectified_left)
                                cv2.imshow("rectified_right", rectified_right)

                            rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
                            pcd, rgbd = pcl_converter.rgbd_to_projection(
                                depth, rgb)
                            
                            if self.visualize:
                                pcl_converter.visualize_pcd()
                            
                            if self.callback is not None:
                                self.callback(color, depth)

                key = cv2.waitKey(1)
                if key == ord("s"):
                    timestamp = str(int(time.time()))
                    cv2.imwrite(
                        f"{serial_no}_{timestamp}_depth.png", depth_vis)
                    cv2.imwrite(f"{serial_no}_{timestamp}_color.png", color)
                    cv2.imwrite(
                        f"{serial_no}_{timestamp}_rectified_left.png", rectified_left)
                    cv2.imwrite(
                        f"{serial_no}_{timestamp}_rectified_right.png", rectified_right)
                    o3d.io.write_point_cloud(
                        f"{serial_no}_{timestamp}.pcd", pcl_converter.pcl, compressed=True)
                elif key == ord("q"):
                    break


if __name__ == "__main__":
    '''
    Hand: 18443010310CC71200 XLinkDeviceState.X_LINK_UNBOOTED
    Top: 1944301021F54C1300 XLinkDeviceState.X_LINK_UNBOOTED

    '''
    OakDDriver(None, visualize=True, device_mxid='1944301021F54C1300')
    while True:
        time.sleep(1)
