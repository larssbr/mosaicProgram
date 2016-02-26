
## SIFT(scale invariant feature transform) is about:
    Detecting keypoints in scale space
    Assigning an orientation to each keypoint
    Assigning a descriptor to each keypoint that is characteristic for the image locally around the keypoint
    
## RANSAC is about
    a way to deal with a lot of outliers in data when fitting a model.
    
## Coordinate system
    The coordinate system being can be different between  Python/Numpy/Matplotlib/OpenCV

## Transpose
     Be aware that sometimes Python/Numpy may require the transpose of projective transformations (in relation to what we are used to).
    
    
## Preprocess
    Yaw is important since it decides the direction the viechle has when the image was taken
    --> with this information we need less keypoints and features
    
## Filter matches
    FILTER MATCHES
    
## Compute homography with best matches( matches - filtered out matches)

## Computation saving
    prevent adding of the same image in the mosaic, or one that has negligible transformation with the respect to the previous image

## Mosaic Image Blending
    Seams are produced when adjacent pixel intensities differ, in stitched images, due to change lighting conditions.
    page 52 in master thesis
    - transition smoothing also known as alpha blending methods
    - optimal seam finding.

## FREE RESOURCES
    