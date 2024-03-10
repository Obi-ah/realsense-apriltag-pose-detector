import cv2
from apriltag import Detector, DetectorOptions
import pyrealsense2 as rs
import numpy as np


# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

align_to = rs.stream.color
align = rs.align(align_to)

options = DetectorOptions()
detector = Detector(options)

while True:
    frames = pipeline.wait_for_frames()

    aligned_frames = align.process(frames)
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    # Get active profile from the pipeline
    profile = pipeline.get_active_profile()
    depth_profile = profile.get_stream(rs.stream.depth)

    # Extract intrinsics from the stream profile
    depth_intrinsics = depth_profile.as_video_stream_profile().get_intrinsics()

    # Obtain intrinsic values
    ppx = depth_intrinsics.ppx
    ppy = depth_intrinsics.ppy
    fx = depth_intrinsics.fx
    fy = depth_intrinsics.fy

    # Convert color data to OpenCV format
    color_image = np.asanyarray(color_frame.get_data())
    imggray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

    results = detector.detect(imggray)

    # Mark boundaries and center
    if results:
        tag = results[0]
        ptA, ptB, ptC, ptD  = tag.corners
        
        ptA = (int(ptA[0]), int(ptA[1]))
        ptB = (int(ptB[0]), int(ptB[1]))
        ptC = (int(ptC[0]), int(ptC[1]))
        ptD = (int(ptD[0]), int(ptD[1]))

        cv2.line(color_image, ptA, ptB, (0, 255, 0), 4)
        cv2.line(color_image, ptB, ptC, (0, 255, 0), 4)
        cv2.line(color_image, ptC, ptD, (0, 255, 0), 4)
        cv2.line(color_image, ptD, ptA, (0, 255, 0), 4)

        center = (int(tag.center[0]), int(tag.center[1]))
        cv2.circle(color_image, center, 5, (0, 0, 255), -1)

        # Get distance to center
        distance = depth_frame.get_distance(center[0], center[1])  # may be center[1], center[0]. Test.
        print(f'Position vector: \n')

        # Calculate world 3D cooordinates
        X = distance * (center[0] - ppx) / fx
        Y = distance * (center[1] - ppy) / fy
        Z = distance

        print(f'X: {X}\nY: {Y}\nZ: {Z}\n\n\n')


    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 1000, 800)
    cv2.moveWindow("image", 800, 100)

    color_image_flipped = cv2.flip(color_image, cv2.ROTATE_180)
    cv2.imshow("image", color_image_flipped)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
