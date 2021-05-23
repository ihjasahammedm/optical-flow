# Author : Ihjas Ahammed M
# License : MIT
import os
import cv2
import numpy as np
import time
import flow_vis
from numpy.core.fromnumeric import around


def visualize_flow(image, flow, threshold=1, step=10, scale=1):
    img = image.copy()
    shape = img.shape
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    mag_n = cv2.normalize(mag,None,100,255,cv2.NORM_MINMAX)
    mag_n = mag_n.astype(np.uint8)
    magNotInf = np.logical_not(np.isinf(mag))
    magThresholded = mag > threshold
    magFiltered = np.logical_and(magNotInf, magThresholded)
    magFilteredIndice = np.where(magFiltered)

    thickness = 5
    tipLength = 0.5
    count = 0
    if len(magFilteredIndice[0] != 0):
        for u, v in zip(magFilteredIndice[0], magFilteredIndice[1]) :
            if count % 50 == 0:
                start = (v, u)
                # delta = mag[u,v] * np.stack([np.cos(ang[u,v]), np.sin(ang[u,v])])   #[delta_u, delta_v]
                # delta *= 50
                # end = (int(v + delta[1]), int(u + delta[0]))
                end = (int(v + flow[u,v,0] * scale), int(u + flow[u,v,1] * scale))
                color = (0, int(mag_n[u,v]), 0)
                out = cv2.arrowedLine(img, start, end, color, thickness, tipLength=tipLength) 
            count += 1
    else:
        out = img
    return out


vidPath = './videos/sample2.mp4'
cap = cv2.VideoCapture(vidPath)
ret = True
displayFps = 50
firstFrame = True
frameCount = 1
frameTotal = cap.get(cv2.CAP_PROP_FRAME_COUNT)
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
tStart = time.time()

outFileName = os.path.splitext(os.path.basename(vidPath))[0] + '.mp4'
outDir = './video_out'
outFPS = 5
outPath = os.path.join(outDir, outFileName)
os.makedirs(outDir, exist_ok = True)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
width = 1080
height = 720
output = cv2.VideoWriter(outPath, fourcc, outFPS, (width, height))

while ret:
    ret, rawImg = cap.read()
    if ret:
        print(f'Processing frame number {frameCount}/{frameTotal}')
        nextFrame = cv2.cvtColor(rawImg, cv2.COLOR_BGR2RGB)
        nextFrameGray = cv2.cvtColor(nextFrame, cv2.COLOR_BGR2GRAY)
        if firstFrame:
            prvsFrame = nextFrameGray
            # trackFlow = np.zeros_like(nextFrame)
            firstFrame = False
        flow = cv2.calcOpticalFlowFarneback(prvsFrame, nextFrameGray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        arrowVis = visualize_flow(nextFrame, flow, threshold=3, step=10, scale=10)
        hsvVis = flow_vis.flow_to_color(flow, convert_to_bgr=False)
        out = arrowVis
        # out = np.hstack((arrowVis, hsvVis))
        # overly = cv2.addWeighted(nextFrame, 0.5, trackFlow, 0.5, 0)
        prvsFrame = nextFrameGray
        windowSize = (width, height)
        resizedImg = cv2.resize(out, windowSize)
        output.write(resizedImg)
        cv2.imshow('Video', resizedImg)
        cv2.waitKey(int(1000/displayFps))
        tElapsed = time.time() - tStart
        print(f'Avg FPS:{frameCount/tElapsed}')
        frameCount += 1
output.release()
cap.release()
cv2.destroyAllWindows()
print('Finished')





