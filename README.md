# optical-flow
This repo demonstrates the motion estimation with optical flow using opencv python.  
**Optical flow** is the method of estimating per pixel motion between two consecutive frames in a video. It is based on the intensity changes between the two frames. Motion due to both camera movement and dynamics in the scene can be estimated.  
[Link to a short video explaining about optical-flow concept](https://www.youtube.com/watch?v=3P911eOFzrU)  
There are mainly two types of optical flow methods:  
1. Sparse Optical Flow : It computes the motion vector for the specific set of objects (for example – detected corners on image). Ex: Pyramid Lucas-Kanade, Sparse RLOF
2. Dense Optical Flow : Motion is estimated for each pixel in the image. Ex : Dense Pyramid Lucas-Kanade, Farneback, PCAFlow, DeepFlow

## Optical flow visualization 
Dense optical flow using Farneback is used in this repo, It can be easily modified to use other dense methods as well. Final output video contains two types of visualization placed side by side.  
1. Flow field visulization as vector arrows 
2. Flow visualization in HSV color space ([using FlowVis library](https://pypi.org/project/flow-vis/))

### How to use this repo
```
python main.py --input_vid_path 'INPUT_VIDEO_PATH' --output_dir 'OUTPUT_DIRECTORY' --output_fps 'OUTPUT_FPS'
```
* input_vid_path (str) : path to input video for which motion to be visualized
* output_dir (str) [optional] : directory in which output video to be saved, default = './videos_out'
* output_fps (int) [optional] : FPS for output video, default = 5

### Sample output 
<figure class="video_container">
  <video controls="true" allowfullscreen="true" poster="path/to/poster_image.png">
    <source src="video_out/sample1.mp4" type="video/mp4">
  </video>
</figure>

### References
1. https://learnopencv.com/optical-flow-in-opencv/
2. https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html
3. Sample video credits: [Ekaterena Bolovstov, Pexels](https://www.pexels.com/video/flat-lay-of-wooden-and-concrete-shapes-7295977/)

