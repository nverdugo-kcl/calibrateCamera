import numpy as np
import os
import matplotlib.pyplot as plt
from config import *
from cv2 import aruco

showPics_while_calib = True
load_from_file       = True
# Define the aruco dictionary and charuco board
dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
board      = cv2.aruco.CharucoBoard((SQUARES_VERTICALLY, SQUARES_HORIZONTALLY), SQUARE_LENGTH, MARKER_LENGTH, dictionary)
params     = cv2.aruco.DetectorParameters()
# Load PNG images from folder
image_files = [os.path.join(PATH_TO_YOUR_IMAGES, f) for f in os.listdir(PATH_TO_YOUR_IMAGES) if f.endswith(".tiff")]
image_files.sort()  # Ensure files are in order
if not load_from_file:
   all_charuco_corners = []
   all_charuco_ids = []
   for image_file in image_files:
       print(image_file)
       image = cv2.imread(image_file)
       image_copy = image.copy()
       marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(image, dictionary, parameters=params)
       print(len(marker_corners))
       
       # If at least one marker is detected
       if len(marker_ids) > 0:
           cv2.aruco.drawDetectedMarkers(image_copy, marker_corners, marker_ids)
           charuco_retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(marker_corners, marker_ids, image, board)
           if showPics_while_calib:
              cv2.aruco.drawDetectedCornersCharuco(image_copy, charuco_corners, charuco_ids, (255, 0, 0))
              cv2.imshow('Image', image_copy)
              cv2.waitKey(0)
              if charuco_retval:
                  all_charuco_corners.append(charuco_corners)
                  all_charuco_ids.append(charuco_ids)
   # Calibrate camera
   cameraMatrixInit = np.array([[ 1000.,    0., image.shape[0]/2.],
                                    [    0., 1000., image.shape[1]/2.],
                                    [    0.,    0.,           1.]])
   
   distCoeffsInit = np.zeros((5,1))
   flags = (cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_FIX_ASPECT_RATIO)
       
   (ret, camera_matrix, dist_coeffs,
        rotation_vectors, translation_vectors,
        stdDeviationsIntrinsics, stdDeviationsExtrinsics,
        perViewErrors) = cv2.aruco.calibrateCameraCharucoExtended(all_charuco_corners, all_charuco_ids, board, image.shape[:2], cameraMatrix=cameraMatrixInit,
                         distCoeffs=distCoeffsInit,
                         flags=flags,
                         criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 10000, 1e-9),
                         )
   # Save calibration data
   np.save('camera_matrix.npy', camera_matrix)
   print(camera_matrix)
   np.save('dist_coeffs.npy', dist_coeffs)
   print(dist_coeffs)
else:
    camera_matrix = np.load('camera_matrix.npy')
    print(camera_matrix)
    dist_coeffs   = np.load('dist_coeffs.npy')
    print(dist_coeffs)
# Iterate through displaying all the images
for id, image_file in enumerate(image_files):
    #image = cv2.imread(image_file)
    #undistorted_image = cv2.undistort(image, camera_matrix, dist_coeffs)
    #imS = cv2.resize(undistorted_image, (640, 480)) 
    #cv2.imshow('Undistorted Image', imS)
#     plt.figure()
    frame = cv2.imread(image_file)
    img_undist = cv2.undistort(frame, camera_matrix, dist_coeffs)
#     plt.subplot(1,2,1)
#     plt.imshow(frame)
#     plt.title("Raw image")
#     plt.axis("off")
#     plt.subplot(1,2,2)
#     plt.imshow(img_undist)
#     plt.title("Corrected image")
#     plt.axis("off")
#     plt.savefig("./Calib_out/"+str(id)+".png")
#     plt.close()
    #cv2.waitKey(0)
    # Save the undistorted image
    cv2.imwrite('./Calib_out/'+str(id)+'.png', img_undist)  
cv2.destroyAllWindows()