
# import subprocess
# import os
# import pyttsx3
# import cv2 as cv
# import mediapipe as mp
# from playsound import playsound
# import numpy as np
# import pyttsx3
# import time
# import math
# from numpy.lib import utils
# from matplotlib import pyplot as plt
# # distance from camera to object(face) measured
# # centimeter
# Known_distance = 45

# # width of face in the real world or Object Plane
# # centimeter
# # Known_width = round(faces[0][2] * 0.0264583333,2)
# # print(Known_width)
# Known_width = 14.3

# # Colors
# GREEN = (0, 255, 0)
# RED = (0, 0, 255)
# WHITE = (255, 255, 255)
# BLACK = (0, 0, 0)

# mpPose = mp.solutions.pose
# mpFaceMesh = mp.solutions.face_mesh
# facemesh = mpFaceMesh.FaceMesh(max_num_faces = 1)
# mpDraw = mp.solutions.drawing_utils
# drawing = mpDraw.DrawingSpec(thickness = 1 , circle_radius = 1)
# pose = mpPose.Pose()
# capture = cv.VideoCapture(0)
# lst=[]
# n=0
# scale = 3
# ptime = 0
# count = 0
# brake = 0
# x=150
# y=195
# height = 0

# def speak(audio):

#     engine = pyttsx3.init()
#     voices = engine.getProperty('voices')
#     engine.setProperty('rate',150)

#     engine.setProperty('voice', voices[0].id)
#     engine.say(audio)

#     # Blocks while processing all the currently
#     # queued commands
#     engine.runAndWait()

# # defining the fonts
# fonts = cv.FONT_HERSHEY_COMPLEX

# # face detector object
# face_detector = cv.CascadeClassifier("./opencv/haarcascade_frontalface_default.xml")

# # focal length finder function
# def Focal_Length_Finder(measured_distance, real_width, width_in_rf_image):

#     # finding the focal length
#     focal_length = (width_in_rf_image * measured_distance) / real_width
#     return focal_length

# # distance estimation function
# def Distance_finder(Focal_Length, real_face_width, face_width_in_frame):

#     distance = (real_face_width * Focal_Length)/face_width_in_frame

#     # return the distance
#     return distance


# def face_data(image):

#     face_width = 0 # making face width to zero

#     gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
#     plt.imshow(gray_image)

#     # detecting face in the image
#     faces = face_detector.detectMultiScale(gray_image, 1.3, 5)

#     # looping through the faces detect in the image
#     # getting coordinates x, y , width and height
#     for (x, y, h, w) in faces:

#         # draw the rectangle on the face
#         cv.rectangle(image, (x, y), (x+w, y+h), GREEN, 2)

#         # getting face width in the pixels
#         face_width = w

#     # return the face width in pixel
#     return face_width


# # reading reference_image from directory
# ref_image = cv.imread("./sample_data/input/Abdullah.jpeg")

# # find the face width(pixels) in the reference_image
# ref_image_face_width = face_data(ref_image)

# # get the focal by calling "Focal_Length_Finder"
# # face width in reference(pixels),
# # Known_distance(centimeters),
# # known_width(centimeters)
# Focal_length_found = Focal_Length_Finder(
#     Known_distance, Known_width, ref_image_face_width)

# # show the reference image
# # cv.imshow("ref_image", ref_image)

# # initialize the camera object so that we
# # can get frame from it
# cap = cv.VideoCapture(0)

# # looping through frame, incoming from
# # camera/video
# while True:

#     # reading the frame from camera
#     _, frame = cap.read()

#     # calling face_data function to find
#     # the width of face(pixels) in the frame
#     face_width_in_frame = face_data(frame)

#     # check if the face is zero then not
#     # find the distance
#     if face_width_in_frame != 0:

#         # finding the distance by calling function
#         # Distance distance finder function need
#         # these arguments the Focal_Length,
#         # Known_width(centimeters),
#         # and Known_distance(centimeters)
#         Distance = Distance_finder(
#             Focal_length_found, Known_width, face_width_in_frame)

#         cv.putText(
#             frame, f"Distance: {round(Distance,2)} cms", (30, 35),
#         fonts, 0.6, GREEN, 2)

#         # draw line as background of text
#         cv.line(frame, (30, 30), (230, 30), RED, 32)
#         cv.line(frame, (30, 30), (230, 30), BLACK, 28)
#         Distance = round(Distance)
#         if Distance in range(500 , 550):
#             speak("Stand there and dont move")
#             while True:
#                 isTrue,img = capture.read()
#                 img_rgb = cv.cvtColor(img , cv.COLOR_BGR2RGB)
#                 result = pose.process(img_rgb)
#                 if result.pose_landmarks:
#                     mpDraw.draw_landmarks(img, result.pose_landmarks,mpPose.POSE_CONNECTIONS)
#                     for id,lm in enumerate(result.pose_landmarks.landmark):
#                         lst[n] = lst.append([id,lm.x,lm.y])
#                         n+1
#                         # print(lm.z)
#                         # if len(lst)!=0:
#                         #     print(lst[3])
#                         h , w , c = img.shape
#                         if id == 32 or id==31 :
#                             cx1 , cy1 = int(lm.x*w) , int(lm.y*h)
#                             cv.circle(img,(cx1,cy1),15,(0,0,0),cv.FILLED)
#                             d = ((cx2-cx1)**2 + (cy2-cy1)**2)**0.5
#                             # height = round(utils.findDis((cx1,cy1//scale,cx2,cy2//scale)/10),1)
#                             di = round(d*0.5)
#                             speak(f"You are {di} centimeters tall")
#                             height = di
#                             filename = "./sample_data/input/user_image_" + str(di) + ".jpg"
#                             cv.imwrite(filename, frame)
# #                             speak("Press q.")
# #                             if ord('q'):
# #                                 cv.destroyAllWindows()
# #                             break

#                             dom = ((lm.z-0)**2 + (lm.y-0)**2)**0.5
#                             # height = round(utils.findDis((cx1,cy1//scale,cx2,cy2//scale)/10),1)

#                             cv.putText(img ,"Height : ",(40,70),cv.FONT_HERSHEY_COMPLEX,1,(255,255,0),thickness=2)
#                             cv.putText(img ,str(di),(180,70),cv.FONT_HERSHEY_DUPLEX,1,(255,255,0),thickness=2)
#                             cv.putText(img ,"cms" ,(240,70),cv.FONT_HERSHEY_PLAIN,2,(255,255,0),thickness=2)
#                             cv.putText(img ,"Stand atleast 3 meter away" ,(40,450),cv.FONT_HERSHEY_PLAIN,2,	(0,0,255),thickness=2)
#                             break
#                             # cv.putText(img ,"Go back" ,(240,70),cv.FONT_HERSHEY_PLAIN,2,(255,255,0),thickness=2)
#                         if id == 6:
#                             cx2 , cy2 = int(lm.x*w) , int(lm.y*h)
#                             # cx2 = cx230
#                             cy2 = cy2 + 20
#                             cv.circle(img,(cx2,cy2),15,(0,0,0),cv.FILLED)
#                 img = cv.resize(img , (700,500))
#                 ctime = time.time()
#                 fps = 1/(ctime-ptime)
#                 ptime=ctime
#                 cv.putText(img , "FPS : ",(40,30),cv.FONT_HERSHEY_PLAIN,2,(0,0,0),thickness=2)
#                 cv.putText(img , str(int(fps)),(160,30),cv.FONT_HERSHEY_PLAIN,2,(0,0,0),thickness=2)
#                 cv.imshow("Task",img)
# #                 if cv.waitKey(20) & 0xFF == ord('q'):
# #                     break
#                 break
#             capture.release()

#             cv.destroyAllWindows()

#             break
#         elif Distance < 500 :
#             speak("Step back")
#         elif Distance > 550:
#             speak("Come a little closer")

#         # Drawing Text on the screen
#         cv.putText(
#             frame, f"Distance: {round(Distance,2)} cms", (30, 35),
#         fonts, 0.6, GREEN, 2)

#     # show the frame on the screen
#     cv.imshow("frame", frame)

#     # quit the program if you press 'q' on keyboard
#     if cv.waitKey(1) == ord("q"):
#         break

# # closing the camera
# cap.release()

# # closing the the windows that are opened
# cv.destroyAllWindows()
