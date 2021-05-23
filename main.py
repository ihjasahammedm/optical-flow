# Author : Ihjas Ahammed M
# License : MIT
import os
import cv2
import numpy as np
import time
def visualize_flow(image, flow, step=100):
    img = image.copy()
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    xgrid, ygrid = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    flowStart = np.stack([xgrid, ygrid])
#     print(flow_start[:,:10,:10])
#     print(mag[:10, :10])
#     print(ang[:10, :10])
    delta = mag * np.stack([np.cos(ang), np.sin(ang)])
    flowEnd = flowStart + delta
    flowEnd = flowEnd.astype(np.int32)
#     print(delta.shape)
    mag_n = mag/(np.max(mag)) 
    # Line thickness of 9 px 
    thickness = 9
    for x in range(0, image.shape[1],step):
        for y in range(0, image.shape[0],step):
            start = (x, y)
            print(flowEnd[:, y, x])
            end = (flowEnd[0,y,x], flowEnd[1,y,x])
            color = (0, int(mag_n[y,x] * 255), 0)
            out = cv2.arrowedLine(img, start, end, color, thickness) 
    return out


vidPath = 'videos/sample1.mp4'
cap = cv2.VideoCapture(vidPath)
ret = True
displayFps = 50
firstFrame = True
frameCount = 1
frameTotal = cap.get(cv2.CAP_PROP_FRAME_COUNT)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
tStart = time.time()

outFileName = os.path.splitext(os.path.basename(vidPath))[0] + '.mp4'
outDir = './video_out'
outFPS = 5
outPath = os.path.join(outDir, outFileName)
os.makedirs(outDir, exist_ok = True)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
print(width, height)
output = cv2.VideoWriter(outPath, fourcc, outFPS, (width, height))

while ret:
    ret, rawImg = cap.read()
    if ret:
        print(f'Processing frame number {frameCount}/{frameTotal}')
        nextFrame = cv2.cvtColor(rawImg, cv2.COLOR_BGR2RGB)
        nextFrameGray = cv2.cvtColor(nextFrame, cv2.COLOR_BGR2GRAY)
        if firstFrame:
            prvsFrame = nextFrameGray
            firstFrame = False
        flow = cv2.calcOpticalFlowFarneback(prvsFrame, nextFrameGray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        out = visualize_flow(nextFrame, flow, step = 50)
        output.write(out)
        prvsFrame = nextFrameGray
        windowSize = (720, 512)
        resizedImg = cv2.resize(out, windowSize)
        cv2.imshow('Video', resizedImg)
        cv2.waitKey(int(1000/displayFps))
        tElapsed = time.time() - tStart
        print(f'Avg FPS:{frameCount/tElapsed}')
        frameCount += 1
output.release()
cap.release()
cv2.destroyAllWindows()
print('Finished')





