# April Tag detection using pyapriltags and OpenCV

import cv2
import numpy as np
from pyapriltags import Detector
import time
import Stats
import roboDetector
import logging

logging.basicConfig(level=logging.INFO) # CRITICAL, ERROR, WARNING, INFO, DEBUG
logger = logging.getLogger()

if logger.isEnabledFor(logging.DEBUG):
    logging.info("Debug mode is enabled")

def getTagSides(tag):
    # get the corners of the tag side that is closer to the centroid (hub)
    if tag.tag_id == 7:
        return (2,3)
    elif tag.tag_id in list([3,5,8]):
        return (1,2)
    elif tag.tag_id == 4:
        return (0,1)
    else:
        return 'Unknown'

def findTags(filename, params, material, start_frame, reconfig = False):
    # Define detector options with refined parameters
    # For accurate detection, see the processed paramaters in the startFrame.csv file
    if params.lower() == 'blue':
        qd, qs, re, ds = roboDetector.blue()
    else:
        qd, qs, re, ds = roboDetector.red()

    at_detector = Detector(
        families='tag36h11',    # Family of tags to detect
        nthreads=4,             # Number of threads to use
        quad_decimate=qd,      # Decimation factor (1.0 for no downsampling, >1 for speed, <1 for accuracy)
        quad_sigma=qs,         # Gaussian blur (0.0 for no blurring, >0 for denoising)
        refine_edges=re,      # Refine the detected tag edges
        decode_sharpening=ds, # Sharpening factor for decoding
        debug=False             # Debug mode
    )

    # Load the video
    filename = 'Processed 2/'+filename+'.mp4'
    cap = cv2.VideoCapture(filename)
    csv_filename = filename.replace('mp4', 'csv')
    csv_file = open(csv_filename, 'w')

    # For flat frame detection (see Gait Analayis file)
    firstFrame = start_frame # frame number of the first frame
    gaitCycle = 33 # number of frames per gait cycle

    # Undetected frames
    undetected_frames = []

    # Stats - variables
    distance = 0
    distance_cutoff = 5
    cumulative_distance = 0
    displacement = 0
    rotation = 0
    rotation_cutoff = 5
    cumulative_rotation = 0
    rotional_displacement = 0
    centroid_array = []

    # For missing tags
    centroid_t1 = 0 #centroid to tag distance
    centroid_t2 = 0
    centroid_t3 = 0

    # # Saving a video
    # movie_frames = []
    # fps = int(cap.get(cv2.CAP_PROP_FPS))
    # frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # frame_height = 900 #int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # frame_size = (frame_width, frame_height)
    # movieName = f"{filename}_cMovie.mp4"    
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # out = cv2.VideoWriter(movieName, fourcc, fps, frame_size)

    while cap.isOpened():
        ret, frame = cap.read()
        logger.debug(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if not ret:
            break
        elif (cap.get(cv2.CAP_PROP_POS_FRAMES)-firstFrame)%gaitCycle !=0:
            continue

        # get the frame size, width and height
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        # delete all the pixels that are not black or white
        if reconfig:
            diff_frame = roboDetector.reconfigureFrame(frame.copy(),material)
        else:
            diff_frame = frame.copy()

        # Convert the frame to grayscale
        gray = cv2.cvtColor(diff_frame, cv2.COLOR_BGR2GRAY)

        # Detect the tags
        tags = at_detector.detect(gray)
        
        # Print the detected tag IDs
        logger.debug('For frame: {}, Detected tags: {}'.format(cap.get(cv2.CAP_PROP_POS_FRAMES), [tag.tag_id for tag in tags]))

        # Logger print

        # Write the frame number, detected tag IDs, center to a csv file
        csv_file.write(str(cap.get(cv2.CAP_PROP_POS_FRAMES)) + ',')

        for tag in tags:
            if not logger.isEnabledFor(logging.DEBUG):
                break
            logger.debug('For frame: {}, Tag ID: {},Tag Center: {}'.format(cap.get(cv2.CAP_PROP_POS_FRAMES),tag.tag_id,tag.center))
            #csv_file.write(str(tag.tag_id) + ',')
            #csv_file.write(str(round(tag.center[0], 2)) + ',' + str(round(tag.center[1], 2)) + ',')

            # Draw the tag outline
            for idx in range(len(tag.corners)):
                cv2.line(frame, tuple(tag.corners[idx-1].astype(int)),
                        tuple(tag.corners[idx].astype(int)), (0, 255, 0), 2)
                
            # Draw the red line on top of the tag
            cv2.line(frame, tuple(tag.corners[0].astype(int)), tuple(tag.corners[1].astype(int)), (0, 0, 255), 4)

            # Draw the tag ID
            cv2.putText(frame, str(tag.tag_id), org=(tag.center.astype(int) - 10),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(0, 0, 255), thickness=2)

        # Draw the centroid if all tags are detected
        if len(tags) == 3:        
            centroid = np.mean([tag.center for tag in tags], axis=0).astype(int)
            #draw x mark at the centroid increasing the size of the mark
            if logger.isEnabledFor(logging.DEBUG):
                cv2.drawMarker(frame, tuple(centroid), (255, 0, 255), markerType=cv2.MARKER_STAR, markerSize=20, thickness=6)

            if 'centroid_prev' not in locals(): #for first frame
                centroid_prev = centroid
                cumulative_distance = 0
                first_centroid = centroid
                initial_tags = tags
                first_T1 = tags[0].center
                first_T2 = tags[1].center
                first_T3 = tags[2].center
                frame_prev = frame.copy()
                centroid_t1 = np.linalg.norm(centroid - tags[0].center)
                centroid_t2 = np.linalg.norm(centroid - tags[1].center)
                centroid_t3 = np.linalg.norm(centroid - tags[2].center)
                # length of the side of the tag
                sideLength1 = tags[0].corners[1][0] - tags[0].corners[0][0]
                sideLength2 = tags[0].corners[1][0] - tags[0].corners[0][0]
                sideLength3 = tags[0].corners[1][0] - tags[0].corners[0][0]        
                csv_file.write(str(round(first_centroid[0], 2)) + ',' + str(round(first_centroid[1], 2)) + ',')
                csv_file.write(str(round(first_T1[0], 2)) + ',' + str(round(first_T1[1], 2)) + ',')
                csv_file.write(str(round(first_T2[0], 2)) + ',' + str(round(first_T2[1], 2)) + ',')
                csv_file.write(str(round(first_T3[0], 2)) + ',' + str(round(first_T3[1], 2)) + '\n')
                csv_file.write(str(cap.get(cv2.CAP_PROP_POS_FRAMES)) + ',')
            centroid_array.append(centroid)
            csv_file.write('Centroid' + ',')
            csv_file.write(str(round(centroid[0], 2)) + ',' + str(round(centroid[1], 2)) + ',')
        elif len(tags) > 0 and len(centroid_array) > 0:
            tag = tags[0]
            tag_idx = getTagSides(tag)
            if tag_idx == 'Unknown':
                tag = tags[1]
                tag_idx = getTagSides(tag)
            
            if tag_idx == 'Unknown':
                continue

            if tag.tag_id == initial_tags[0].tag_id:
                dist = centroid_t1 + sideLength1/2#100
            elif tag.tag_id == initial_tags[1].tag_id:
                dist = centroid_t2 + sideLength2/2#100
            else:
                dist = centroid_t3 + sideLength3/2#100

            line = np.array([tag.corners[tag_idx[0],:], tag.corners[tag_idx[1],:]])

            # Other Lines
            if logger.isEnabledFor(logging.DEBUG):
                colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)] #Blue, Green, Red, Cyan
                cv2.line(frame, tuple(tag.corners[tag_idx[0],:].astype(int)), tuple(tag.corners[tag_idx[1],:].astype(int)), color=colors[0], thickness=4)
                cv2.line(frame, tuple(tag.center.astype(int)), tuple(line.mean(axis=0).astype(int)), color=colors[1], thickness=4)
            
                # Draw perpendicular line that passes through the center of the tag and the center of the line and extends by a distance of centroid_t1 pixels
                cv2.line(frame, tuple(tag.center.astype(int)), tuple((line.mean(axis=0) - dist*(tag.center - line.mean(axis=0))/np.linalg.norm(tag.center - line.mean(axis=0))).astype(int)), color=colors[2], thickness=4)

            # Add the white marker at the end of the line
            centroid = (line.mean(axis=0) - dist*(tag.center - line.mean(axis=0))/np.linalg.norm(tag.center - line.mean(axis=0))).astype(int)
            #cv2.drawMarker(frame, centroid, (255, 255, 255), markerType=cv2.MARKER_STAR, markerSize=5, thickness=2)
            centroid_array.append(centroid)
            csv_file.write('CentroidEst' + ',') # Estimated centroid
            csv_file.write(str(round(centroid[0], 2)) + ',' + str(round(centroid[1], 2)) + ',')
        else:
            undetected_frames.append(cap.get(cv2.CAP_PROP_POS_FRAMES))

        # get stats and write to csv file
        if len(tags) > 0:
            if 'centroid_prev' in locals():
                #Plot the centroids on the frame
                for i in range(len(centroid_array)):
                    cv2.drawMarker(frame, tuple(centroid_array[i]), (31+3*i, 11+3*i, 242+1*i), markerType=cv2.MARKER_DIAMOND, markerSize=30, thickness=30)
                cv2.drawMarker(frame, tuple(centroid_array[-1]), (255, 0, 0), markerType=cv2.MARKER_SQUARE, markerSize=40, thickness=40)
                cv2.drawMarker(frame, tuple(centroid_array[0]), (0, 255, 0), markerType=cv2.MARKER_STAR, markerSize=40, thickness=40)
                #cv2.imwrite(filename.replace('mp4', 'png').format(cap.get(cv2.CAP_PROP_POS_FRAMES)), frame)

                #crop the image from the centroid with a distance of 100 pixels in height
                crop_img = frame.copy()
                final_centroid = tuple(centroid_array[-1].astype(int))
                crop_img = frame[final_centroid[1]-450:final_centroid[1].astype(int)+450, :]
                #cv2.imwrite(filename.replace('.mp4', '_crop.png').format(cap.get(cv2.CAP_PROP_POS_FRAMES)), crop_img)

                distance = np.linalg.norm(centroid_prev - centroid)
                # ! Change below to get the rotation from the detected tagID

                if tags[0].tag_id == initial_tags[0].tag_id:
                    rotation = Stats.getRotation(first_centroid[0], first_centroid[1], first_T1[0], first_T1[1], centroid[0], centroid[1], tags[0].center[0], tags[0].center[1])
                elif tags[0].tag_id == initial_tags[1].tag_id:
                    rotation = Stats.getRotation(first_centroid[0], first_centroid[1], first_T2[0], first_T2[1], centroid[0], centroid[1], tags[0].center[0], tags[0].center[1])
                else:
                    rotation = Stats.getRotation(first_centroid[0], first_centroid[1], first_T3[0], first_T3[1], centroid[0], centroid[1], tags[0].center[0], tags[0].center[1])
            
            if distance > distance_cutoff:
                centroid_prev = centroid
                cumulative_distance += distance
                csv_file.write('Distance' + ','+ str(round(distance, 2)) + ',' + 'Cumulative Distance' + ',' + str(round(cumulative_distance, 2))+ ',')
            else:
                csv_file.write('Distance' + ','+ str(round(0, 2)) + ',' + 'Cumulative Distance' + ',' + str(round(cumulative_distance, 2))+ ',')

            if rotation > rotation_cutoff:
                cumulative_rotation += rotation
                csv_file.write('Rotation' + ',' + str(round(rotation, 2)) + ',' + 'Cumulative Rotation' + ',' + str(round(cumulative_rotation, 2))+ ',')
            else:
                csv_file.write('Rotation' + ',' + str(round(0, 2)) + ',' + 'Cumulative Rotation' + ',' + str(round(cumulative_rotation, 2))+ ',')

        if 'first_centroid' in locals():
            displacement = np.linalg.norm(first_centroid - centroid)
            csv_file.write('Displacement' + ',' + str(round(displacement, 2)) + ',')
            rotional_displacement = Stats.getRotation(first_centroid[0], first_centroid[1], first_T1[0], first_T1[1], centroid[0], centroid[1], tags[0].center[0], tags[0].center[1])
            csv_file.write('Rotational Displacement' + ',' + str(round(rotional_displacement, 2)) + ',')
        else:
            csv_file.write('Displacement' + ',' + str(round(0, 2)) + ',')
            csv_file.write('Rotational Displacement' + ',' + str(round(0, 2)) + ',')

        csv_file.write('\n')
        logger.debug('Distance: {:.2f}, Cumulative Distance: {:.2f}, Displacement: {:.2f}, Rotation: {:.2f}, Cumulative Rotation: {:.2f}'.format(distance,cumulative_distance,displacement,rotation,cumulative_rotation))
 
        # Display the frame
        cv2.imshow(filename, crop_img)
        # movie_frames.append(crop_img.copy())
        # out.write(crop_img.copy())
        # time.sleep(0.1)

        # Wait for user input
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    # out.release()
    cv2.destroyAllWindows()
    csv_file.close()
    statistics = str('Cumulative Distance, {:.2f}'.format(cumulative_distance)+','+
                   'Cumulative Rotation, {:.2f}'.format(cumulative_rotation)+','+ 'Displacement, {:.2f}'.format(displacement)+','+
                   'Rotational Displacement, {:.2f}'.format(rotional_displacement))
    return undetected_frames, statistics

if __name__ == '__main__':
    # Define the parameters for the apriltag detector
    params = 'red'
    material = '23'
    reconfig = True
    start_frame = 19
    filename = 'Sequence 45'
    _,_,movieFrames = findTags(filename, params, material, start_frame, reconfig)
    '''
    movieName = f"{filename}_cMovie.mp4"    
    movie = cv2.VideoWriter(movieName,cv2.VideoWriter_fourcc('m','p','4','v'),1,movieFrames[0].shape[1],movieFrames[0].shape[0])
    for i in range(len(movieFrames)):
        cv2.imshow('Movie', movieFrames[i])
        cv2.waitKey(100)
    '''
    print("Finished processing file:", filename)