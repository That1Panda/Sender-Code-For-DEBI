#!/usr/bin/env python3
import sys
# Import the necessary libraries
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.draw import circle_perimeter

#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from tf.transformations import euler_from_quaternion
from differential_robot import DifferentialRobot
from std_msgs.msg import Bool
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from tf.transformations import euler_from_quaternion
import sys
# Import the necessary libraries
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.draw import circle_perimeter
import math

# Define a callback function to convert the ROS message to an image and display it

detect_ball=False

angle_to_go=0
r=0
reached_ball=False
distance_threshold=0.1
point=[0,0]


        # Initialize variables
robot= DifferentialRobot(0.033, 0.287)  # create DifferentialRobot object with wheel radius and distance
goal_x = 0.973147+0.258880
goal_x=0
goal_y=0
goal_y = 0.196194
linear_velocity = 0.2
angular_velocity = 0.5
scan_range = 0.7
min_scan_distance = float('inf')
vel_msg = Twist()
color=""
count=0

def odom_callback( odom_msg):
        # Update current position and orientation of the robot
        position = odom_msg.pose.pose.position
        orientation = odom_msg.pose.pose.orientation
        orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
        _, _, yaw = euler_from_quaternion(orientation_list)
        robot.x = position.x
        robot.y = position.y
        robot.theta = yaw

def scan_callback( scan_msg):
        # Find the minimum distance in the scan range
        min_scan_distance = min(scan_msg.ranges[:len(scan_msg.ranges)//3])

def distance():
    return ((goal_x - robot.x)*2 + (goal_y - robot.y)*2)*0.5
def linear():
    linear_velocity = math.tanh(distance()) * linear_velocity
    return linear_velocity
def angle():
    return (math.atan2(goal_y - robot.y, goal_x - robot.x) - robot.theta)
def angular():
        angular_velocity = math.tanh(angle()) * angular_velocity
        return angular_velocity
        

# def robot_and_curve_par(threshed):
#     # Get x and y coordinates of the robot
#     x_pixel, y_pixel = robot_coords(threshed)
#     # Fit a polynomial curve to the robot's path
#     coefficients = np.polyfit(x_pixel, y_pixel, 1)
#     # Generate x values for the polynomial fit
#     x_fit = np.linspace(x_pixel.min(), x_pixel.max(), len(x_pixel))
#     # Predict y values using the polynomial fit
#     y_fit = np.polyval(coefficients, x_fit)
    
#     return coefficients[0], coefficients[1]

def robot_and_curve_par(threshed):
    # Get x and y coordinates of the robot
    x_pixel, y_pixel = robot_coords(threshed)
    # Fit a polynomial curve to the robot's path
    coefficients = np.polyfit(x_pixel, y_pixel, 1)
    # Generate x values for the polynomial fit
    x_fit = np.linspace(x_pixel.min(), x_pixel.max(), len(x_pixel))
    # Predict y values using the polynomial fit
    y_fit = np.polyval(coefficients, x_fit)
    # Return the coefficients of the polynomial and the line image
    return coefficients[0], coefficients[1]

def to_polar_coords(x_pixel, y_pixel):
    # Convert (x_pixel, y_pixel) to (distance, angle) 
    # in polar coordinates in rover space
    # Calculate distance to each pixel
    dist = np.sqrt(x_pixel*2 + y_pixel*2)
    # Calculate angle away from vertical for each pixel
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles

def robot_coords(binary_img):
    # Identify nonzero pixels
    ypos, xpos, _ = binary_img.nonzero()
    # Calculate pixel positions with reference to the rover position being at the
    # # center bottom of the image.
    x_pixel = -(ypos - binary_img.shape[0]/2).astype(np.float64)
    y_pixel = -(xpos - binary_img.shape[1]/2).astype(np.float64)
    return x_pixel, y_pixel
def perspect_transform(img, src, dst):
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))  # keep same size as input image
    # cv2.imshow("ggg",img)
    cv2.imshow('perspect',warped)
    return warped
def color_thresh_green(img):
## convert to hsv
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (36, 25, 25), (70, 255,255))
## slice the green
    imask = mask>0
    green = np.zeros_like(img, np.uint8)
    green[imask] = img[imask]
    return green
def count_pixel(img):
    # Initialize the counter
    count = 0
    
   
    green_lower = np.array([0, 100, 0])
    green_upper = np.array([100, 255, 100])
    # Create a mask for the red pixels
    green_mask = cv2.inRange(img, green_lower, green_upper)
    count = np.count_nonzero(blue_mask)
    # Count the number of red pixels
    count = np.count_nonzero(green_mask)
    return count
def locate_ball(a, b, r, main_img):
    dst_size = 40
    bottom_offset = 7
    scale=2*dst_size
    # Known ball size in meters
    known_ball_size = 0.055
    # Focal length of Raspberry Pi Camera V2 lens in meters
    focal_length = 0.00304
    #focal_length=0.00247
    # Pixel size of the camera sensor in meters
    pixel_size = 0.0000058
    #pixel_size=0.000004477
    # Use the mask to copy the circle pixels from the original image to the black image
    source = np.float32([[53,114], [576 ,112],[439,50], [182,54]])
    destination = np.float32([[main_img.shape[1]/2 - dst_size, main_img.shape[0] - bottom_offset],
                          [main_img.shape[1]/2 + dst_size, main_img.shape[0] - bottom_offset],
                          [main_img.shape[1]/2 + dst_size, main_img.shape[0] - 2*dst_size - bottom_offset], 
                          [main_img.shape[1]/2 - dst_size, main_img.shape[0] - 2*dst_size - bottom_offset],
                          ])
    warped = perspect_transform(main_img, source, destination)

    cirmap_green= color_thresh_green(warped)
    if cirmap_green.any() :   
            print('hi ball ')
            # rock_x_pixel,rock_y_pixel=robot_coords(cirmap_green)
            # dist, angles = to_polar_coords(rock_x_pixel, rock_y_pixel)
            angle,c=robot_and_curve_par(cirmap_green)
    else:
        angle=0
    angle=math.atan(angle)
    # Convert the radius from pixels to meters
    apparent_size = 2.0 * r * pixel_size
    # Calculate the distance between the camera and the ball
    distance = (known_ball_size * focal_length) / apparent_size
    # # Calculate the angle of the ball from the center of the image
    # angle = math.atan2(b - main_img.shape[0], a - (main_img.shape[1]/2))
    #angle=np.median(angles)
    # Calculate the distance in the x and y directions
    x_rb=distance*math.cos(angle)
    y_rb=distance*math.sin(angle)
    # distance_y = distance * math.sin(angle)+self.robot.y
    distance_x= robot.x+(x_rb)*math.cos(robot.theta)-(y_rb)*math.sin(robot.theta)
    distance_y= robot.y+x_rb*math.sin(robot.theta)+y_rb*math.cos(robot.theta)
    print(r)
    # Calculate the x and y positions of the ball relative to the center of the camera
    x = distance_x
    y = distance_y
    return x, y, distance
def image_callback(ros_image):

    # Convert the ROS message to an image
    try:
        
        main_img = CvBridge().imgmsg_to_cv2(ros_image, "bgr8")
        # main_img = main_img[230:480,180:(640-180)]
        main_img = main_img[290:,:]
        gray=cv2.cvtColor(main_img,cv2.COLOR_BGR2GRAY)
        # print(main_img.shape)
        # hsv to segment background
        hsv = cv2.cvtColor(main_img, cv2.COLOR_BGR2HSV)
        hsv_gray = cv2.cvtColor(hsv, cv2.COLOR_BGR2GRAY)
 
        
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
        eq_img = clahe.apply(hsv_gray)
        eq_img=cv2.cvtColor(cv2.merge([eq_img, eq_img, eq_img]), cv2.COLOR_BGR2GRAY)
        # eq_img = cv2.adaptiveThreshold(eq_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31,9)
        # Apply binary threshold with a value of 150
        thresh_value = 120
        max_value = 255
        ret, eq_img = cv2.threshold(eq_img, thresh_value, max_value, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

        # eq_img = cv2.morphologyEx(eq_img, cv2.MORPH_OPEN, kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
        eq_img = cv2.morphologyEx(eq_img, cv2.MORPH_OPEN, kernel)
        canny=cv2.Canny(eq_img, 0, 180, apertureSize=3)
        # eq_img=cv2.GaussianBlur(eq_img,(3,3),cv2.BORDER_DEFAULT)
        # edges = 
##############################################################################################################
        # WORKING CODE FOR HOUGH CIRCLE

        # try_radii = np.arange(10, 70)
        # hough_results = hough_circle(canny, np.arange(10,70 ,1),normalize=False)
        # ridx, r, c = np.unravel_index(np.argmax(hough_results), hough_results.shape)

        # cv2.circle(main_img,(c,r),try_radii[ridx],(255,255,0),2)
        # cv2.circle(main_img,(c,r),2,(0,255,255),3)
##############################################################################################################

        # # Detect two radii
        # hough_radii =np.arange(10, 150)
        # hough_res = hough_circle(canny, hough_radii)
        # total_num_peaks=1
        # # Select the most prominent 3 circles
        # accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,total_num_peaks=total_num_peaks)
        
        # for center_y, center_x, radius in zip(cy, cx, radii):
        #     cv2.circle(main_img, (center_x, center_y), radius, (0, 0, 255), 2)
        #     cv2.circle(main_img, (center_x, center_y), 2, (0, 255, 0), 2)
        cv2.imshow("jjj",eq_img)


        # preform hsv on eq_img 
        # hsv_eq = cv2.cvtColor(eq_img, cv2.COLOR_GRAY2HSV)

        # Blur the image using a median filter
        
        # declare the gray img as the equalized gray image

        
        # # Perform Hough Circle Transform
        circles=cv2.HoughCircles(eq_img,cv2.HOUGH_GRADIENT,1,20,param1=70,param2=12,minRadius=10,maxRadius=90)
        if circles is not None:
            circles=np.uint16(np.around(circles))
            mask = np.zeros_like(main_img)
            try_radii = np.arange(10, 90)
            hough_results = hough_circle(canny, np.arange(10,90 ,1),normalize=False)
            ridx, r, c = np.unravel_index(np.argmax(hough_results), hough_results.shape)
            # find the circle with the closest radius to the one found by hough
            
            cv2.circle(main_img,(c,r),try_radii[ridx],(255,255,0),2)
            cv2.circle(main_img,(c,r),2,(0,255,255),3)
            cv2.circle(mask, (c,r), try_radii[ridx], (0, 255, 0),-1)

            cv2.imshow("mask",mask)
            distance_x, distance_y ,distance= locate_ball(c, r, try_radii[ridx],mask)
            print("Distance_x: {:.2f} m, Distance_y: {:.2f} m".format(distance, distance_y))
            point=[distance_x,distance_y]
        # print(circles)
        # print(circles.shape)
        # print([circles[0][0],circles[0][1]])
        # print("---------------------------------------")
        # Detect edges using Canny
        # edges = cv2.Canny(hsv_gray, 0, 180, apertureSize=3)
        

        # imgy=gray
        # Loop over the detected circles and draw them on the original image
        # for i in circles[0,:]:
        #     cv2.circle(main_img,(i[0],i[1]),i[2],(0,255,0),2)
        #     cv2.circle(main_img,(i[0],i[1]),2,(0,0,255),3)


    except Exception as e:
        print(e)
    # Display the image
    cv2.imshow("Camera Stream",main_img)
    
    cv2.waitKey(3)

# Initialize the ROS node and the subscriber
rospy.init_node("ip_camera_subscriber", anonymous=True)
sub = rospy.Subscriber("/camera/image", Image, image_callback)
odom_sub = rospy.Subscriber('/odom', Odometry, odom_callback)
scan_sub = rospy.Subscriber('/scan', LaserScan, scan_callback)

# Keep the ROS node running
rospy.spin()