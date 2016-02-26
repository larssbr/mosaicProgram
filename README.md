# Planar Seafloor Mosaic Reconstruction program

## Requirements:
* Tested on windows 10 Using Python 2.7.11
* OpenCV 2.4.9

* Innstaled python using miniconda
Then : conda install numpy, scipy, matplotlib
Then : pip install imutils

Developed in pycharm

## Usage ###################

Make sure posAnd_dir.csv is located within the same directory as the images. The posAnd_dir.csv file should have the following format:
```
filename,latitude,longitude,altitude,yaw,pitch,roll
dji_0644.jpg,­123.114661,38.426805,90.689292,9.367337,1.260910,0.385252
...
```

Call this in the terminal
```
./mosaic.py -d datasets/example1 -is 0.1 -os 0.9
or call the following in the terminal
python mosaic.py -d datasets/example1 -is 0.1 -os 0.9
```

``-is`` is the scaling on the input images (smaller value = smaller images, faster processing)
``-os`` is the scaling on the output container (smaller value = smaller output mosaic)
``-m`` is for intermediate matching visualizations.

## Examples

The following shows the intermediate feature matching

![matching](docs/sift_features_and_matching.png)

The following shows the result after stitching 24 images

![output](docs/final_output.png)

## Need to do

Port the code from opencv3 to opencv 2.4.9 --> DONE

Make it work without csv pose file -- under construction

## Assumptions of imageset -------------------------------------------------------

Program assumes all images in the directory folder set by the user is in the same SIZE.
--> I.e  pixels in height X Width are the same size
--> Thereby setting the height and width to scale the images for the mosaic creation

1 The alltitude of the pictoures has to be around the same hight over ground to work
The underwater terrain is planar. -->  so the altitude of the images is about the same

• The ROV is passively stable in pitch and roll, therefore an affine transformation is used.
---> 2 The pitch is +- 2 degrees

---> 3 The roll is +- 2 degrees

4 The yaw can change more

5 The turbidity of the water allows for sufficient visibility for reasonable
optical imaging of the working area.
--> the water has to be clear

6 The light present in the scene is sufficient to allow the camera to obtain
satisfactory seafloor imagery.

## Future work as propossed by Celiberti's master thesis ##################################################

Implement a "image blending module" along the mosaicking pipeline, to produce higher quality mosaics.

--> Region Mosaic creation
A way to merge to or more created mosaics into one big mosaic

---> Experiment
try other image registration methods such as SURF (Speeded Up Robust Features), or use feature-less based
technique which are faster than the feature-based ones.

---> Memory control
Cut the mosaic whenever its resolutions goes above a certain height and width limits( a treshold).
Another possible solution consists in re-scaling all the images and homography transformation once the
mosaic image reaches a certain resolution limit.
In this way the mosaic image occupies always the same size in memory

## Future work  presented by Lars Brusletto ##################################################

--> click on the newly made mosaic image. From this click it schould be possible to derrive the [lattitude, longditude]
Then: use [lat,long] to make a new path for the ROV to return to this position

########### Data fusion of estimates of ROVs  latitude,longitude,altitude,yaw,pitch,roll parameters in real time ############

---> Int\egrate labview with python

---> Math part
Adding an homography filter, and feedback pose estimations, which along with navigation data, help
the vision system to find better correspondences, between interest points. A
filter for the homography has been already proposed by Mahony et al (40).
It’s basically a nonlinear filter which uses information from gyroscope, to
estimate coefficients of the homography matrix. In this way, the estimated
pose measurements help the pose filter to get better results.

