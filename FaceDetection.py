from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
import cv2
import matplotlib.pyplot as plt
import math


def skinDetection(src,ycrcb,size):
    # skin mask
    b, g, r = cv2.split(src)
    y, cr, cb = cv2.split(ycrcb)
    skinMask = np.zeros((size[0], size[1]), dtype = np.uint8)
    for i in range(0, size[0]):
        for j in range(0, size[1]):
            if (r[i,j] > 95) and (g[i,j] > 40) and (b[i,j] > 20) and (r[i,j] > g[i,j]) \
                    and (r[i,j] > b[i, j]) and (abs(r[i,j] - g[i,j]) > 15) \
                    and ((max(r[i, j], g[i, j], b[i,j]) - min(r[i,j], g[i,j], b[i,j])) > 15) \
                    and (cr[i,j] >= (0.3448 * cb[i,j]) + 76.2069) \
                    and (cr[i,j] >= (-4.5652 * cb[i,j]) + 234.5652) \
                    and (cr[i,j] <= (-1.15 * cb[i,j]) + 301.75) \
                    and (cr[i,j] <= (-2.2857 * cb[i,j]) + 432.85):
                skinMask[i, j] = 255
    kernel = np.ones((5,5), np.uint8)
    skinMask = cv2.morphologyEx(skinMask, cv2.MORPH_OPEN, kernel)
    skinMask = cv2.morphologyEx(skinMask, cv2.MORPH_CLOSE, kernel)

    # find contours in the edge map
    cnts = cv2.findContours(skinMask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # approximate contours rects
    contours_poly = [None] * len(cnts)
    boundRect = [None] * len(cnts)
    hull = [None]*len(cnts)
    for i, c in enumerate(cnts):
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        boundRect[i] = cv2.boundingRect(contours_poly[i])
        hull[i] = cv2.convexHull(contours_poly[i])
    boundRect = list(boundRect)
    return(cnts,hull,boundRect)

def eyeDetection(ycrcb,skinArea,skinMask,size):
    mask = np.zeros((size[0],size[1]),dtype=np.float32)
    cv2.drawContours(mask, [skinMask], -1, (1), -1)
    mask = mask[(int(skinArea[1])):(int(skinArea[1]+skinArea[3])),\
                 (int(skinArea[0])):(int(skinArea[0]+skinArea[2]))]
    ycrcb = ycrcb[(int(skinArea[1])):(int(skinArea[1]+skinArea[3])),\
                 (int(skinArea[0])):(int(skinArea[0]+skinArea[2]))]
    y, cr, cb = cv2.split(ycrcb)

    # eye map chroma
    eyeMapC = ((cb**2)+(255-cr)**2+(cb/cr))/3
    eyeMapC = eyeMapC*255/eyeMapC.max()
    eyeMapC = eyeMapC*mask
    #cv2.imwrite('images/eyemapc.png', np.uint8(eyeMapC))
    
    # eye map luma
    y = y*255/y.max()
    SE = np.array([[0.7498, 1.1247, 1.4996, 1.8745, 1.4996, 1.1247, 0.7498],
          [1.1247, 1.4996, 1.8745, 2.2494, 1.8745, 1.4996, 1.1247],
          [1.4996, 1.8745, 2.2494, 2.6243, 2.2494, 1.8745, 1.4996],
          [1.8745, 2.2494, 2.6243, 2.9992, 2.6243, 2.2494, 1.8745],
          [1.4996, 1.8745, 2.2494, 2.6243, 2.2494, 1.8745, 1.4996],
          [1.1247, 1.4996, 1.8745, 2.2494, 1.8745, 1.4996, 1.1247],
          [0.7498, 1.1247, 1.4996, 1.8745, 1.4996, 1.1247, 0.7498]])
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    dilate = cv2.dilate(y, SE)
    erode = cv2.erode(y, SE)
    eyeMapL = dilate/(erode+1)
    eyeMapL = eyeMapL*255/eyeMapL.max()
    eyeMapL = eyeMapL*mask
    #cv2.imwrite('images/eyemapl.png', np.uint8(eyeMapL))

    # eye map total
    eyeMap = eyeMapC * eyeMapL
    eyeMap = cv2.morphologyEx(eyeMap, cv2.MORPH_OPEN, kernel)
    eyeMap = cv2.morphologyEx(eyeMap, cv2.MORPH_CLOSE, kernel)
    #eyeMap = cv2.dilate(eyeMap, kernel)
    eyeMap = eyeMap*255/eyeMap.max()
    eyeMap = eyeMap*mask
    
    
    #cv2.imwrite('images/eyemap.png', np.uint8(eyeMap))

    # Find contours
    ret,thresh = cv2.threshold(eyeMap,254,255,0)
    cnts = cv2.findContours(np.uint8(thresh), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    i = 0
    while len(cnts) < 5 and i < 255:
        i = i+1
        ret,thresh = cv2.threshold(eyeMap,254-i,255,0)
        cnts = cv2.findContours(np.uint8(thresh), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

    cX = np.zeros(len(cnts),dtype=np.uint8)
    cY = np.zeros(len(cnts),dtype=np.uint8)
    for i in range(0, len(cnts)):
        for c in [cnts[i]]:
            # calculate moments for each contour
            M = cv2.moments(c)
            if M["m00"] != 0:
                cX[i] = int(M["m10"] / M["m00"])
                cY[i] = int(M["m01"] / M["m00"])
            else:
                contours_poly = [None]*len(cnts)
                boundRect = [None]*len(cnts)
                for i, c in enumerate(cnts):
                    contours_poly[i] = cv2.approxPolyDP(c, 3, True)
                    boundRect[i] = cv2.boundingRect(contours_poly[i])
                for i in range(0, len(cnts)):
                    cX[i] = int((2*boundRect[i][0]+boundRect[i][2])/2)
                    cY[i] = int((2*boundRect[i][1]+boundRect[i][3])/2)
    cX = np.int32(cX)
    cY = np.int32(cY)
    cX = cX+skinArea[0]
    cY = cY+skinArea[1]

    return(cX,cY)

def mouthDetection(ycrcb,skinArea,skinMask,size):
    mask = np.zeros((size[0],size[1]),dtype=np.float32)
    cv2.drawContours(mask, [skinMask], -1, (1), -1)
    mask = mask[(int(skinArea[1])):(int(skinArea[1]+skinArea[3])),\
                 (int(skinArea[0])):(int(skinArea[0]+skinArea[2]))]
    ycrcb = ycrcb[(int(skinArea[1])):(int(skinArea[1]+skinArea[3])),\
                 (int(skinArea[0])):(int(skinArea[0]+skinArea[2]))]
    y, cr, cb = cv2.split(ycrcb)
    
    # mouth map
    n = 0.95*sum(sum(cr**2))/sum(sum(cr/cb))
    mouthMap = cr**2*(cr**2-n*(cr/cb))**2
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    mouthMap = cv2.morphologyEx(mouthMap, cv2.MORPH_OPEN, kernel)
    mouthMap = mouthMap*mask
    mouthMap = mouthMap*255/mouthMap.max()
    
    #cv2.imwrite('images/mouthmap.png', np.uint8(mouthMap))

    # Find contours
    ret,thresh = cv2.threshold(mouthMap,254,255,0)
    cnts = cv2.findContours(np.uint8(thresh), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    i = 0
    while len(cnts) < 3 and i < 255:
        i = i+1
        ret,thresh = cv2.threshold(mouthMap,254-i,255,0)
        cnts = cv2.findContours(np.uint8(thresh), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

    cX = np.zeros(len(cnts),dtype=np.uint8)
    cY = np.zeros(len(cnts),dtype=np.uint8)
    for i in range(0, len(cnts)):
        for c in [cnts[i]]:
            # calculate moments for each contour
            M = cv2.moments(c)
            if M["m00"] != 0:
                cX[i] = int(M["m10"] / M["m00"])
                cY[i] = int(M["m01"] / M["m00"])
            else:
                contours_poly = [None]*len(cnts)
                boundRect = [None]*len(cnts)
                for i, c in enumerate(cnts):
                    contours_poly[i] = cv2.approxPolyDP(c, 3, True)
                    boundRect[i] = cv2.boundingRect(contours_poly[i])
                for i in range(0, len(cnts)):
                    cX[i] = int((2*boundRect[i][0]+boundRect[i][2])/2)
                    cY[i] = int((2*boundRect[i][1]+boundRect[i][3])/2)
    cX = np.int32(cX)
    cY = np.int32(cY)
    cX = cX+skinArea[0]
    cY = cY+skinArea[1]
    
    return(cX,cY)

def faceConfirm(eyeCoor1,eyeCoor2,mouthCoor):
    j = np.float32(mouthCoor)
    if eyeCoor1[0]<eyeCoor2[0]:
        i = np.float32(eyeCoor1)
        k = np.float32(eyeCoor2)
    else:
        i = np.float32(eyeCoor2)
        k = np.float32(eyeCoor1)
    if (abs(dist.euclidean(i,j)-dist.euclidean(j,k))) < 0.25*max((dist.euclidean(i,j),dist.euclidean(j,k))) and\
       (abs(dist.euclidean(i,j)-dist.euclidean(i,k))) < 0.25*max((dist.euclidean(i,j),dist.euclidean(j,k))) and\
       i[1] < j[1] and k[1] < j[1] and i[0] < j[0] < k[0]:
        face = 1
    else:
        face = 0
    return(face)

if __name__ == "__main__":
    #a = 63
    for a in range(1,101):
        src = cv2.imread('img/'+str(a)+'.jpg')
        
        size = src.shape
        m = 640
        n = int(m/size[1]*size[0])
        src = cv2.resize(src,(m,n))
        size = src.shape


        ycrcb = np.float32(cv2.cvtColor(src, cv2.COLOR_BGR2YCrCb))
        
        skinContour,skinHull,skinArea = skinDetection(src,ycrcb,size)
        srca = src.copy()
        srcb = src.copy()
        srcc = src.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX

        for i in range (0,len(skinArea)):
            srca = src.copy()
            srcb = src.copy()
            srcc = src.copy()
            eyecntsX,eyecntsY = eyeDetection(ycrcb.copy(),skinArea[i],skinContour[i],size)
            mouthcntsX,mouthcntsY = mouthDetection(ycrcb.copy(),skinArea[i],skinContour[i],size)

            #draw cnts
            for j in range (0, len(eyecntsX)):
                cv2.circle(srca,(eyecntsX[j],eyecntsY[j]), 5, (255,0,0), -1)
            for k in range (0, len(mouthcntsX)):
                cv2.circle(srca,(mouthcntsX[k],mouthcntsY[k]), 5, (0,255,0), -1)
            if len(mouthcntsX) >= 1 and len(eyecntsX) >= 2:
                for l in range (0, len(mouthcntsX)):
                    for m in range (0, len(eyecntsX)-1):
                        for n in range (m+1, len(eyecntsX)):
                            if faceConfirm((eyecntsX[m],eyecntsY[m]),(eyecntsX[n],eyecntsY[n]),(mouthcntsX[l],mouthcntsY[l])) == 1:
                                cv2.line(srca,(eyecntsX[m],eyecntsY[m]),(eyecntsX[n],eyecntsY[n]),(0,0,255),2)
                                cv2.line(srca,(eyecntsX[m],eyecntsY[m]),(mouthcntsX[l],mouthcntsY[l]),(0,0,255),2)
                                cv2.line(srca,(eyecntsX[n],eyecntsY[n]),(mouthcntsX[l],mouthcntsY[l]),(0,0,255),2)

            eyehullX,eyehullY = eyeDetection(ycrcb.copy(),skinArea[i],skinHull[i],size)
            mouthhullX,mouthhullY = mouthDetection(ycrcb.copy(),skinArea[i],skinHull[i],size)

            #draw hull
            for j in range (0, len(eyehullX)):
                cv2.circle(srcb,(eyehullX[j],eyehullY[j]), 5, (255,0,0), -1)
            for k in range (0, len(mouthhullX)):
                cv2.circle(srcb,(mouthhullX[k],mouthhullY[k]), 5, (0,255,0), -1)
            if len(mouthhullX) >= 1 and len(eyehullX) >= 2:
                for l in range (0, len(mouthhullX)):
                    for m in range (0, len(eyehullX)-1):
                        for n in range (m+1, len(eyehullX)):
                            if faceConfirm((eyehullX[m],eyehullY[m]),(eyehullX[n],eyehullY[n]),(mouthhullX[l],mouthhullY[l])) == 1:
                                cv2.line(srcb,(eyehullX[m],eyehullY[m]),(eyehullX[n],eyehullY[n]),(0,0,255),2)
                                cv2.line(srcb,(eyehullX[m],eyehullY[m]),(mouthhullX[l],mouthhullY[l]),(0,0,255),2)
                                cv2.line(srcb,(eyehullX[n],eyehullY[n]),(mouthhullX[l],mouthhullY[l]),(0,0,255),2)

            #draw cnts hull
            for j in range (0, len(eyecntsX)):
                cv2.circle(srcc,(eyecntsX[j],eyecntsY[j]), 5, (255,0,0), -1)
            for k in range (0, len(mouthcntsX)):
                cv2.circle(srcc,(mouthcntsX[k],mouthcntsY[k]), 5, (0,255,0), -1)
            for j in range (0, len(eyehullX)):
                cv2.circle(srcc,(eyehullX[j],eyehullY[j]), 5, (255,0,0), -1)
            for k in range (0, len(mouthhullX)):
                cv2.circle(srcc,(mouthhullX[k],mouthhullY[k]), 5, (0,255,0), -1)
            if len(mouthcntsX) >= 1 and len(eyecntsX) >= 2:
                for l in range (0, len(mouthcntsX)):
                    for m in range (0, len(eyecntsX)-1):
                        for n in range (m+1, len(eyecntsX)):
                            if faceConfirm((eyecntsX[m],eyecntsY[m]),(eyecntsX[n],eyecntsY[n]),(mouthcntsX[l],mouthcntsY[l])) == 1:
                                cv2.line(srcc,(eyecntsX[m],eyecntsY[m]),(eyecntsX[n],eyecntsY[n]),(0,0,255),2)
                                cv2.line(srcc,(eyecntsX[m],eyecntsY[m]),(mouthcntsX[l],mouthcntsY[l]),(0,0,255),2)
                                cv2.line(srcc,(eyecntsX[n],eyecntsY[n]),(mouthcntsX[l],mouthcntsY[l]),(0,0,255),2)
            if len(mouthhullX) >= 1 and len(eyehullX) >= 2:
                for l in range (0, len(mouthhullX)):
                    for m in range (0, len(eyehullX)-1):
                        for n in range (m+1, len(eyehullX)):
                            if faceConfirm((eyehullX[m],eyehullY[m]),(eyehullX[n],eyehullY[n]),(mouthhullX[l],mouthhullY[l])) == 1:
                                cv2.line(srcc,(eyehullX[m],eyehullY[m]),(eyehullX[n],eyehullY[n]),(0,0,255),2)
                                cv2.line(srcc,(eyehullX[m],eyehullY[m]),(mouthhullX[l],mouthhullY[l]),(0,0,255),2)
                                cv2.line(srcc,(eyehullX[n],eyehullY[n]),(mouthhullX[l],mouthhullY[l]),(0,0,255),2)
            if len(mouthcntsX) >= 1 and len(eyecntsX) >= 1 and len(eyehullX) >= 1:
                for l in range (0, len(mouthcntsX)):
                    for m in range (0, len(eyecntsX)):
                        for n in range (0, len(eyehullX)):
                            if faceConfirm((eyecntsX[m],eyecntsY[m]),(eyehullX[n],eyehullY[n]),(mouthcntsX[l],mouthcntsY[l])) == 1:
                                cv2.line(srcc,(eyecntsX[m],eyecntsY[m]),(eyehullX[n],eyehullY[n]),(0,0,255),2)
                                cv2.line(srcc,(eyecntsX[m],eyecntsY[m]),(mouthcntsX[l],mouthcntsY[l]),(0,0,255),2)
                                cv2.line(srcc,(eyehullX[n],eyehullY[n]),(mouthcntsX[l],mouthcntsY[l]),(0,0,255),2)
            if len(mouthhullX) >= 1 and len(eyecntsX) >= 1 and len(eyehullX) >= 1:
                for l in range (0, len(mouthhullX)):
                    for m in range (0, len(eyehullX)):
                        for n in range (0, len(eyecntsX)):
                            if faceConfirm((eyehullX[m],eyehullY[m]),(eyecntsX[n],eyecntsY[n]),(mouthhullX[l],mouthhullY[l])) == 1:
                                cv2.line(srcc,(eyehullX[m],eyehullY[m]),(eyecntsX[n],eyecntsY[n]),(0,0,255),2)
                                cv2.line(srcc,(eyehullX[m],eyehullY[m]),(mouthhullX[l],mouthhullY[l]),(0,0,255),2)
                                cv2.line(srcc,(eyecntsX[n],eyecntsY[n]),(mouthhullX[l],mouthhullY[l]),(0,0,255),2)
            if len(mouthhullX) >= 1 and len(eyecntsX) >= 2:
                for l in range (0, len(mouthhullX)):
                    for m in range (0, len(eyecntsX)-1):
                        for n in range (m+1, len(eyecntsX)):
                            if faceConfirm((eyecntsX[m],eyecntsY[m]),(eyecntsX[n],eyecntsY[n]),(mouthhullX[l],mouthhullY[l])) == 1:
                                cv2.line(srcc,(eyecntsX[m],eyecntsY[m]),(eyecntsX[n],eyecntsY[n]),(0,0,255),2)
                                cv2.line(srcc,(eyecntsX[m],eyecntsY[m]),(mouthhullX[l],mouthhullY[l]),(0,0,255),2)
                                cv2.line(srcc,(eyecntsX[n],eyecntsY[n]),(mouthhullX[l],mouthhullY[l]),(0,0,255),2)
            if len(mouthcntsX) >= 1 and len(eyehullX) >= 2:
                for l in range (0, len(mouthcntsX)):
                    for m in range (0, len(eyehullX)-1):
                        for n in range (m+1, len(eyehullX)):
                            if faceConfirm((eyehullX[m],eyehullY[m]),(eyehullX[n],eyehullY[n]),(mouthcntsX[l],mouthcntsY[l])) == 1:
                                cv2.line(srcc,(eyehullX[m],eyehullY[m]),(eyehullX[n],eyehullY[n]),(0,0,255),2)
                                cv2.line(srcc,(eyehullX[m],eyehullY[m]),(mouthcntsX[l],mouthcntsY[l]),(0,0,255),2)
                                cv2.line(srcc,(eyehullX[n],eyehullY[n]),(mouthcntsX[l],mouthcntsY[l]),(0,0,255),2)
            srca = srca[(int(skinArea[i][1])):(int(skinArea[i][1]+skinArea[i][3])),\
                        (int(skinArea[i][0])):(int(skinArea[i][0]+skinArea[i][2]))]
            cv2.imwrite('kosong/'+str(a)+'/'+str(a)+' '+str(i)+' cnts face.jpg',np.uint8(srca))
            srcb = srcb[(int(skinArea[i][1])):(int(skinArea[i][1]+skinArea[i][3])),\
                        (int(skinArea[i][0])):(int(skinArea[i][0]+skinArea[i][2]))]
            cv2.imwrite('kosong/'+str(a)+'/'+str(a)+' '+str(i)+' hull face.jpg',np.uint8(srcb))
            srcc = srcc[(int(skinArea[i][1])):(int(skinArea[i][1]+skinArea[i][3])),\
                        (int(skinArea[i][0])):(int(skinArea[i][0]+skinArea[i][2]))]
            cv2.imwrite('kosong/'+str(a)+'/'+str(a)+' '+str(i)+' cnts hull face.jpg',np.uint8(srcc))
            
                        

##        (text_width, text_height) = cv2.getTextSize(str(a)+' Contour', font, 0.8, thickness=1)[0]
##        box_coords = ((10, size[0]-10), (10+text_width-2, size[0]-10-text_height-2))
##        cv2.rectangle(srca, box_coords[0], box_coords[1], (0, 0, 0), cv2.FILLED)
##        cv2.putText(srca,str(a)+' Contour',(10,size[0]-10), font, 0.8,(255,255,255),1,cv2.LINE_AA)
##        
##        (text_width, text_height) = cv2.getTextSize(str(a)+' Convex Hull', font, 0.8, thickness=1)[0]
##        box_coords = ((10, size[0]-10), (10+text_width-2, size[0]-10-text_height-2))
##        cv2.rectangle(srcb, box_coords[0], box_coords[1], (0, 0, 0), cv2.FILLED)
##        cv2.putText(srcb,str(a)+' Convex Hull',(10,size[0]-10), font, 0.8,(255,255,255),1,cv2.LINE_AA)
##
##        (text_width, text_height) = cv2.getTextSize(str(a)+' Contour + Convex Hull', font, 0.8, thickness=1)[0]
##        box_coords = ((10, size[0]-10), (10+text_width-2, size[0]-10-text_height-2))
##        cv2.rectangle(srcc, box_coords[0], box_coords[1], (0, 0, 0), cv2.FILLED)
##        cv2.putText(srcc,str(a)+' Contour + Convex Hull',(10,size[0]-10), font, 0.8,(255,255,255),1,cv2.LINE_AA)
##
##        (text_width, text_height) = cv2.getTextSize(str(a), font, 0.8, thickness=1)[0]
##        box_coords = ((10, size[0]-10), (10+text_width-2, size[0]-10-text_height-2))
##        cv2.rectangle(src, box_coords[0], box_coords[1], (0, 0, 0), cv2.FILLED)
##        cv2.putText(src,str(a),(10,size[0]-10), font, 0.8,(255,255,255),1,cv2.LINE_AA)
        
##        cv2.imshow('img',np.concatenate((srca,srcb,srcc),axis=1))
##        cv2.imwrite('img - result2/'+str(a)+'.jpg',np.concatenate((srca,srcb,srcc),axis=1))
##        cv2.imwrite('src/'+str(a)+'.jpg',src)
    
