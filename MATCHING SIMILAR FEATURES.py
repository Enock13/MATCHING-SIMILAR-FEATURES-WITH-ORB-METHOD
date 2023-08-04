import numpy as np
import cv2

Cap= cv2.VideoCapture(0)
Img = cv2.imread("BOOK_IMAGE.jpg")

while True:
    Rec, Frame = Cap.read()
    
    Gray_Frames=cv2.cvtColor(Frame, cv2.COLOR_BGR2GRAY)
    Gray_Img=cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY)
    
    #NUMBERS OF POINTS TO DETECT
    Num_pt=500
    #START THE FUNCTION .ORB_create(NUMBERS OF POINTS)
    Orb = cv2.ORB_create(Num_pt)
    
    #DETECT THE POINTS WITH FUNCTION .detectAndComputer(Img, None )     CAUTION: IN THE VARIABLES OF THE BEGINNING ALWAYS TO PUT KEYPOINT AND DESCRIPTOR AS VARIABLES
    Keypoints1, Descriptor1= Orb.detectAndCompute(Gray_Frames,None)
    Keypoints2, Descriptor2= Orb.detectAndCompute(Gray_Img,None)
    
    print(Descriptor1)
    print(Descriptor2)
    
    #DRAW THE POINT DETECTED
    Frames_Display= cv2.drawKeypoints(Frame,Keypoints1,outImage=np.array([]), color=(255,0,0),
                                      flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
    Img_Display= cv2.drawKeypoints(Img,Keypoints2,outImage=np.array([]), color=(255,0,0),
                                      flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
    
    
    #MATCH THE POINT 
    #FIRST CREATE A OBJECT COMPARE OF DESCRIPTORS
    Match=cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
    Matches=Match.match(Descriptor1,Descriptor2)
    
    #NOW WE ORDER THE LIST OF MATCHES WITH THE FUNCTION sorted()
    Matches=sorted(Matches, key=lambda x: x.distance, reverse=False)
    
    #FILTER MATCHES
    Good_Matches=int(len(Matches)*0.5)
    Matches=Matches[:Good_Matches]
    
    for Mat in Matches:
        query_idx = Mat.queryIdx
        train_idx = Mat.trainIdx
        distance = Mat.distance
        print(f"queryIdx: {query_idx}, trainIdx: {train_idx}, distance: {distance}")
    
    #DRAW THE MATCHES
    Img_Matches = cv2.drawMatches(Frame,Keypoints1,Img,Keypoints2,Matches,None)
    
    
    #SHOW THE MATCHES IN SCREEN
    cv2.imshow("MATCHES",Img_Matches)
    
    #SHOW THE POINTS IN SCREEN
    cv2.imshow("IMAGE",Img_Display)
    
    t = cv2.waitKey(1)
    if t == 27:
        break
    
    
Cap.release()
cv2.destroyAllWindows()