# --------------------------
# ~~ Agile Robotics Lab ~~~
# The University of Alabama
# --------------------------

from collections import deque
import numpy as np
import cv2
from picamera2 import Picamera2
import csv
import time

import StatsImage as SI
import PlotImage as PI

from libcamera import controls

#
#
#MAKE SURE TO CHANGE THESE
#
#
robot = "yobot" 		# yobot or robot
spines = "none" 		# none, dir, or in
angle = "3" 			# 0,2,3,5
surface = "black" 		# black, white, or wood
gait = "trans" 			# trans or rot
test_num = "Test406" 			# 1, 2, or 3

def getColorLimits(color,n):
    
    #'ColorCode':["ColorName", "LowerLimit", "UpperLimit"]
    if surface == "white":
        colors_dict = {
            'G': ["Green",(50,50,30),(80,255,255)],
            'R': ["Red",(100,180,0),(135,255,255)],
            'B': ["Blue",(0, 100,50),(30, 255, 125)],
            'L': ["Light Blue",(0, 100,125),(30, 255, 255)],
            }
    elif surface == "black":
        colors_dict = {
            'G': ["Green",(60,100,100),(80,255,255)],
            'R': ["Red",(115,100,100),(135,255,255)],
            'B': ["Blue",(0,50,100),(20,255,255)],
            'L': ["Light Blue",(21,50,230),(40, 255, 255)],
        }
    elif surface == "wood":
        colors_dict = {
            'G': ["Green",(50,50,30),(80,255,255)],
            'R': ["Red",(100,150,0),(135,255,180)],
            'B': ["Blue",(0, 100,50),(20, 255, 150)],
            'L': ["Light Blue",(0, 100,125),(30, 255, 255)],
        }

    name, colorLower, colorUpper = colors_dict.get(color, ["!! Color Not Defined -> Re-run the code",None,None])
    print(f"Tracker {n} assigned: {name}")
    return colorLower,colorUpper

def getContours(hsv,colorLower,colorUpper):
    mask = cv2.inRange(hsv, colorLower, colorUpper) #Generate initial mask
    mask = cv2.erode(mask, None, iterations=2) #Erode to remove small blob errors that are not object
    mask = cv2.dilate(mask, None, iterations=2) #Dilate to keep original object of interest size
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    return cnts

def getCenters(cnts, drawCircle = True, frame = None):
    c = max(cnts, key=cv2.contourArea)
    ((x, y), radius) = cv2.minEnclosingCircle(c)
    M = cv2.moments(c)
    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    # Markers vary from 12 to 20 pixels in radius that vary with inclination and distance
    if radius in range(12,20) and drawCircle: #If radius is at least 11 pixels, proceed
        #Draw the circle and centroid, updating the tracked points
        cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
        cv2.circle(frame, center, 5, (0, 0, 255), -1)
        return (center,x,y)
    return (center,x,y) if radius in range(12,20) else (None,None,None)

def drawTrail(pts,frame):
    for i in range(1, len(pts)): #For all tracked points
        if pts[i - 1] is None or pts[i] is None: #Ignore empty points
            continue
        thickness = int(np.sqrt(64 / float(i + 1)) * 2.5) #Generate the continuous line thickness
        cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness) #Draw connecting line

def track_error(color):
    print(f"Cannot detect {color} marker")
    return (None, None, None)

print("!!!!! Write the file name before executing to save data")

# Get trackers color from user
if robot == "yobot":
    Tracker1_color = "R"
    Tracker2_color = "L"
    Tracker3_color = "B"
if robot == "robot":
    Tracker1_color = "G"
    Tracker2_color = "O"
    Tracker3_color = "B"

T1_Lower,T1_Upper = getColorLimits(Tracker1_color,1)
T2_Lower,T2_Upper = getColorLimits(Tracker2_color,2)
T3_Lower,T3_Upper = getColorLimits(Tracker3_color,3)

# Create a CSV file to store the data
timestamp = int(time.time()) #Returns an integer of time in seconds
nameCode = robot + "_" + spines + "_" + angle + "_" + surface + "_" + gait + "_" + test_num #Write a name code here if you want to
imageName = '/home/kyle/Documents/brobot_testing/' + f"{nameCode}_image.png"
filename = '/home/kyle/Documents/brobot_testing/' + f"{nameCode}_tracking.csv"

# Data
T1_plot_x, T1_plot_y = [],[]
T2_plot_x, T2_plot_y = [],[]
T3_plot_x, T3_plot_y = [],[]
centroid_x_data,centroid_y_data = [],[]

# Stats
cumulative_distance = 0.0
cumulative_rotation = 0.0
centroid0 = np.array([]) # Previous frame centroid
start_centroid = np.array([0, 0, 0]) # Previous frame centroid
distance_data = []
rotation_data = []
distance_cutoff = 2.0 # Cutoff to ignore movements less than 2 pixels 
rotation_cutoff = 2.0 # Cutoff to ignore rotations less than 2 degrees

cv2.startWindowThread() #Start window for the camera
cv2.namedWindow('Combined Window', cv2.WND_PROP_FULLSCREEN)
#cv2.setWindowProperty('Combined Window',cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
image = np.zeros((2*480, 2*640, 3), dtype=np.uint8)
startFrame = 0

# Camera
picam2 = Picamera2() 
#config = picam2.create_preview_configuration(
#    controls={'FrameRate': 2, 'ExposureTime': 1000, 'AnalogueGain': 1.0, 'ColourGains': (1, 1)})
#picam2.configure(config)
picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous, "AfSpeed": controls.AfSpeedEnum.Fast})
config = picam2.create_preview_configuration(
    controls={'FrameRate': 10})
picam2.configure(config)
#picam2.set_controls({"FrameRate": 1})
#picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})
picam2.start()    
pts = deque(maxlen=5)
pts2 = deque(maxlen=5)
pts3 = deque(maxlen=5)
feedRunning = True

frames = []
    
while feedRunning:
    # Rewrite the following 3 lines of code 
    frame_bgr = picam2.capture_array()
    frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB) #Fixes Larry's Mistake
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV) #Converts the RGB values into HSV values

    #Look for contours in mask and set center to object center
    cnts = getContours(hsv, T1_Lower, T1_Upper)
    cnts2 = getContours(hsv, T2_Lower, T2_Upper)
    cnts3 = getContours(hsv, T3_Lower, T3_Upper)

    center, T1_x, T1_y = getCenters(cnts, True, frame) if len(cnts) > 0 else track_error(Tracker1_color)
    #pts.appendleft(center) #Update the points
    #drawTrail(pts,frame)
    
    center2,T2_x,T2_y = getCenters(cnts2, True, frame) if len(cnts2) > 0 else track_error(Tracker2_color)
    #pts2.appendleft(center2) #Update the points
    #drawTrail(pts2,frame)
    
    center3,T3_x,T3_y = getCenters(cnts3, True, frame) if len(cnts3) > 0 else track_error(Tracker3_color)
    #pts3.appendleft(center3) #Update the points
    #drawTrail(pts3,frame)

    if None in [center, center2, center3]:
        continue
    centroid_x = np.mean([T1_x,T2_x,T3_x]) 
    centroid_y = np.mean([T1_y,T2_y,T3_y])

    if startFrame < 6:
        start_x = centroid_x
        start_y = centroid_y
        start_centroid = np.array([centroid_x, centroid_y,0])
        start_T1 = np.array([T1_x, T1_y,0])
        startFrame += 1
        image[480:480+480,0:640,:] = frame[:,:,:3]
        centroid0 = start_centroid
        continue

    centroid1 = np.array([centroid_x, centroid_y, 0]) # Current frame centroid
    distance = np.linalg.norm(centroid0 - centroid1)
    
    if distance>distance_cutoff:
        distance_data.append(distance)
        centroid0 = centroid1 # For next frame
        cumulative_distance += distance # Euclidean distance
    else:
        distance_data.append(0)
        
    displacement = np.linalg.norm(start_centroid - centroid1)
    
    # Append Data
    T1_plot_x.append(T1_x)
    T1_plot_y.append(T1_y)
    T2_plot_x.append(T2_x)
    T2_plot_y.append(T2_y)
    T3_plot_x.append(T3_x)
    T3_plot_y.append(T3_y)
    centroid_x_data.append(centroid_x)
    centroid_y_data.append(centroid_y)

    # get the Matplotlib figure as an image
    plot_image = PI.update_plot(T1_plot_x, T1_plot_y, T2_plot_x, T2_plot_y, T3_plot_x, T3_plot_y, centroid_x_data, centroid_y_data)

    # Copy the Matplotlib plot image into the OpenCV image
    image[0:480,0:640,:] = frame[:,:,:3]
    image[0:0 + plot_image.shape[0], 640:640 + plot_image.shape[1], :] = plot_image[:, :, :3]
    
    # Add the stats image to the combined image
    # Rotation is calculated based on the vector from centroid to the first marker
    rotation = SI.getRotation(start_centroid[0], start_centroid[1], start_T1[0], start_T1[1], centroid_x, centroid_y, T1_x, T1_y)
    rotation = rotation if rotation > 2 else 0
    cumulative_rotation += rotation
    rotation_data.append(rotation)

    displacement_x = start_x - centroid_x
    displacement_y = start_y - centroid_y

    stats_image = SI.getStatsImage(distance, cumulative_distance,rotation, cumulative_rotation, displacement,displacement_x,displacement_y) 
    image[480:480 + stats_image.shape[0], 640:640 + stats_image.shape[1], :] = stats_image[:, :, :3]

    frames.append(image)

    # Display the combined image in the OpenCV window
    cv2.imshow('Combined Window', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        feedRunning = False
        print("---End by user--")
        break

cv2.imwrite(imageName, image) #Saves the final frame
#picam2.release() #Stop camera
cv2.destroyAllWindows()
print("Logging data")
with open('/home/kyle/Documents/brobot_testing/all_tracking_data.csv', 'a') as csvfile:
    writer = csv.writer(csvfile)
    displacement_x = start_x - centroid_x
    displacement_y = start_y - centroid_y
    writer.writerow([nameCode, f"{cumulative_distance:.2f}", f"{rotation:.2f}", f"{displacement:.2f}", f"{displacement_x:.2f}", f"{displacement_y:.2f}"])
    csvfile.close()
with open(filename, 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(zip(T1_plot_x,T1_plot_y,T2_plot_x,T2_plot_y,T3_plot_x,T3_plot_y,centroid_x_data,centroid_y_data,distance_data))
    writer.writerow([f"Total Distance: {cumulative_distance:.2f}"])
    writer.writerow([f"Total Rotation: {rotation:.2f}"])
    writer.writerow([f"Total Displacement: {displacement:.2f}"])
    writer.writerow([f"Total Displacement: {centroid_x:.2f}"])
    writer.writerow([f"Total Displacement: {centroid_y:.2f}"])
    csvfile.close()


movieName = '/home/kyle/Documents/brobot_testing/' + f"{nameCode}_Movie.mp4"
movie = cv2.VideoWriter(movieName,cv2.VideoWriter_fourcc(*'mp4v'), 10,(960,1280))
for frame in frames:
    movie.write(frame)
movie.release()
