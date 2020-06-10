import cv2 as cv,cv2
import numpy as np
##############################
widthImg = 480
heightImg = 640
#############################

path = "doc.jpg"
cap = cv.VideoCapture(0)
cap.set(3,widthImg)
cap.set(4,heightImg)
cap.set(10,150)

def preProcessing(img):
    imgGray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    imgBlur = cv.GaussianBlur(imgGray,(5,5),1)
    imgCanny = cv.Canny(imgBlur,200,200)
    kernel = np.ones((5,5),)
    imgDial = cv.dilate((imgCanny),kernel,iterations = 2)
    imgThres = cv.erode(imgDial,kernel,iterations = 1)
    return imgThres

def getContours(img):
    biggest = np.array([])
    maxArea = 0
    contours, hierarchy = cv.findContours(img,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        
        if area>50000:
            #cv.drawContours(imgContour,cnt,-1,(255,0,0),3)
            peri = cv.arcLength(cnt,True)
            
            approx = cv.approxPolyDP(cnt,0.02*peri,True)
            if area>maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area
            print(len(approx))
            objCor = len(approx)
            x,y,w,h = cv.boundingRect(approx)
    cv.drawContours(imgContour,biggest,-1,(255,0,0),20)
    return biggest

def reorder(myPoints):
    myPoints = myPoints.reshape((4,2))
    myPointsNew = np.zeros((4,1,2),np.int32)
    add = myPoints.sum(1);
    #print("add",add)

    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    #print("NewPoints",myPointsNew)
    diff = np.diff(myPoints,axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    print("NewPoints",myPointsNew)
    return myPointsNew
def getWarp(img,biggest):
    print(biggest)
    biggest = reorder(biggest)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0,0],[widthImg,0],[0,heightImg],[widthImg,heightImg]])
    matrix = cv.getPerspectiveTransform(pts1,pts2)
    imgOutput = cv.warpPerspective(img,matrix,(widthImg,heightImg))
    imgCropped = imgOutput[20:imgOutput.shape[0]-20,20:imgOutput.shape[1]-20]
    imgCropped = cv.resize(imgCropped,(widthImg,heightImg))

    return imgCropped
    

img = cv.imread(path)
cv.resize(img,(widthImg,heightImg))
imgContour = img.copy()
imgThres = preProcessing(img)
biggest = getContours(imgThres)
print(biggest)
imgWrapped = getWarp(img,biggest)
cv.imshow("Result",imgWrapped)


    
