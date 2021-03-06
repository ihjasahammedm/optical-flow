# Author : Ihjas Ahammed M
# License : MIT
import os
import argparse
import cv2
import numpy as np
import time
import flow_vis

def visualize_flow(image, flow, threshold=1, scale=1, density=0.3):
    """
    Method used to visulize the flow vectors
    Args:
    image (numpy array) : image frame
    flow  (numpy array) : dense flow output from optical flow
    threshold (int)     : flow with a value greater than threshold only will be visualized
    scale (int)         : actual magnitude of flow can be scaled with this factor for better visualization
    density (float)     : make the flows sparse with this argument. Accepted values : 0 to 1
    """
    img = image.copy()
    shape = img.shape
    # convert to polar co-ords
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    mag_n = cv2.normalize(mag, None, 100, 255, cv2.NORM_MINMAX)
    mag_n = mag_n.astype(np.uint8)
    # filter the flow values based on threshold, also remove inf and nan
    # values if any
    magNotInfNan = np.logical_not(np.logical_or(np.isinf(mag), np.isnan(mag)))
    magThresholded = mag > threshold
    magFiltered = np.logical_and(magNotInfNan, magThresholded)
    magFilteredIndice = np.where(magFiltered)
    n_filtered = len(magFilteredIndice[0])
    # set thickness and tip length of flow arrows
    thickness = 5
    tipLength = 0.3
    # number of samples to be selected randomly from the filtered flow samples
    # random sampling without replacement is done with the flow magnitude as
    # weight
    n_samples = int(density * n_filtered)

    if n_samples > 0:
        np.random.seed(0)
        probability = mag[magFilteredIndice]
        probability = probability / np.sum(probability)
        sampled_indices = np.random.choice(
            n_filtered, n_samples, p=probability)

        for sample in sampled_indices:
            # draw the flow vector
            u = magFilteredIndice[0][sample]
            v = magFilteredIndice[1][sample]
            start = (v, u)
            end = (int(v + flow[u, v, 0] * scale),
                   int(u + flow[u, v, 1] * scale))
            color = (0, int(mag_n[u, v]), 0)
            out = cv2.arrowedLine(
                img,
                start,
                end,
                color,
                thickness,
                tipLength=tipLength)
    else:
        out = img
    return out

def main(args):
    vidPath = args.input_vid_path
    cap = cv2.VideoCapture(vidPath)
    ret = True
    displayFps = 50
    frameTotal = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    tStart = time.time()

    # output config
    outFileName = os.path.splitext(os.path.basename(vidPath))[0] + '.mp4'
    outDir = args.output_dir
    outFPS = args.output_fps
    outPath = os.path.join(outDir, outFileName)
    os.makedirs(outDir, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # output width and height
    width = 1080
    height = 720
    output = cv2.VideoWriter(outPath, fourcc, outFPS, (width, height))
    # window size for displaying output
    windowSize = (width, height)
    # read first frame
    ret, firstFrame = cap.read()
    prvsFrame = cv2.cvtColor(firstFrame, cv2.COLOR_BGR2GRAY)
    frameCount = 1
    # motion will be estimated from second frame onwards
    while ret:
        ret, rawImg = cap.read()
        if ret:
            frameCount += 1
            print(f'Processing frame number {frameCount}/{frameTotal}')
            nextFrame = cv2.cvtColor(rawImg, cv2.COLOR_BGR2RGB)
            nextFrameGray = cv2.cvtColor(nextFrame, cv2.COLOR_RGB2GRAY)
            # calculate flow
            flow = cv2.calcOpticalFlowFarneback(
                prvsFrame, nextFrameGray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            # visualize flow as vectors
            arrowVis = visualize_flow(
                nextFrame,
                flow,
                threshold=5,
                scale=10,
                density=0.01)
            # visualize flow in HSV color space
            hsvVis = flow_vis.flow_to_color(flow, convert_to_bgr=False)
            out = np.hstack((arrowVis, hsvVis))
            prvsFrame = nextFrameGray
            resizedImg = cv2.resize(out, windowSize)
            output.write(resizedImg)
            cv2.imshow('Video', resizedImg)
            cv2.waitKey(int(1000 / displayFps))
            tElapsed = time.time() - tStart
            print(f'Avg FPS processed:{frameCount/tElapsed}')
    output.release()
    cap.release()
    cv2.destroyAllWindows()
    print('Finished')

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--input_vid_path', type= str, help= 'Path to the video to be processed')
    arg_parser.add_argument('--output_dir', required=False, type= str, help= 'Directory in which output will be saved', default='./video_out')
    arg_parser.add_argument('--output_fps', required=False, type= int, help= 'Output video FPS', default= 5)
    args = arg_parser.parse_args()
    main(args)
