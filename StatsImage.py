import cv2
import numpy as np

font = cv2.FONT_HERSHEY_SIMPLEX
font_color = (0, 0, 0)

label_height = 30
label_spacer = 15
label1_org_xy = (20, label_height)
label2_org_xy = (20, 2*label_height + label_spacer)
label3_org_xy = (20, 3*label_height + 2*label_spacer)
label4_org_xy = (20, 4*label_height + 3*label_spacer)
label5_org_xy = (20, 5*label_height + 4*label_spacer)
label6_org_xy = (20, 6*label_height + 5*label_spacer)
label7_org_xy = (20, 7*label_height + 6*label_spacer)

# Define text parameters
fontFace = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
thickness = 1

def getStatsImage(distance=0, cumulative_distance=0, rotation=0, cumulative_rotation = 0, displacement=0,displacement_x=0,displacement_y=0):
    stats_image = np.zeros((480, 640, 3), dtype=np.uint8)
    stats_image.fill(255) # Background Color

    puttext = lambda text,org: cv2.putText(stats_image, text, org, font, fontScale, font_color, thickness)

    label1_text = f"Distance: {distance} "
    label2_text = f"Cumulative Distance: {cumulative_distance:.2f}"
    label3_text = f"Rotation: {rotation:.2f}"
    label4_text = f"Cumulative Rotation: {cumulative_rotation:.2f}"
    label5_text = f"Displacement: {displacement:.2f}"
    label6_text = f"Displacement in x: {displacement_x:.2f}"
    label7_text = f"Displacement in y: {displacement_y:.2f}"
    puttext(label1_text, label1_org_xy)
    puttext(label2_text, label2_org_xy)
    puttext(label3_text, label3_org_xy)
    puttext(label4_text, label4_org_xy)
    puttext(label5_text, label5_org_xy)
    puttext(label6_text, label6_org_xy)
    puttext(label7_text, label7_org_xy)

    return stats_image

def getRotation(A0=1, A1=2, B0=4, B1=6, C0=1, C1=3, D0=3, D1=1):
    # Calculate vectors AB and CD
    AB = np.array([B0 - A0, B1 - A1])
    CD = np.array([D0 - C0, D1 - C1])

    # Calculate the dot product of AB and CD
    dot_product = np.dot(AB, CD)

    # Calculate the magnitudes of AB and CD
    magnitude_AB = np.linalg.norm(AB)
    magnitude_CD = np.linalg.norm(CD)

    # Calculate the angle between the two lines in radians
    angle_radians = np.arccos(dot_product / (magnitude_AB * magnitude_CD))

    # Convert radians to degrees
    angle_degrees = np.degrees(angle_radians)

    return angle_degrees