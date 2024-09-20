# get the stats of the data

import numpy as np

def getRotation(A0=1, A1=2, B0=4, B1=6, C0=1, C1=3, D0=3, D1=1):
    # A: Start centroid, B: Start T1, C: Current centroid, D: Current T1
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