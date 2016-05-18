# Seafloor Mosaic image stitching software

import argparse
import os
import logging
import imutils
import cv2
import numpy as np
from matplotlib import pyplot as plt

from importDataset import Images  # Import the functions made in importDataset

logging.basicConfig(format='%(asctime)s %(levelname)s [%(module)s] %(message)s', level=logging.INFO)

class MosaicClass:
    # __init__ gets the attributes calculated/processed in Images()
    def __init__(self, images_obj, is_infofile=None):
        self.logger = logging.getLogger()
        self.images = images_obj.imageList
        self.image_width = images_obj.image_width
        self.imageHeight = images_obj.imageHeight
        self.filenames = images_obj.filenames
        if is_infofile == True:
            self.poses = images_obj.infolist

    # rotateimage_aNdCenter uses the csv files position to rotate and place the images so computation times is faster
    def rotateimage_andCenter(self, img, degrees_ccw=0):
        # degrees_ccw is degrees counter clock wise
        scale_factor = 1.0
        (old_y, old_x, old_c) = img.shape    # note: numpy uses (y,x) convention but most OpenCV functions use (x,y)
        #rotate about center of image.
        rotation_matrix2D = cv2.getRotationMatrix2D(center=(old_x/2, old_y/2), angle=degrees_ccw, scale=scale_factor)
        # choose a new image size.
        [new_x, new_y] = old_x*scale_factor, old_y*scale_factor

        # include this if you want to prevent corners being cut off
        r = np.deg2rad(degrees_ccw)
        [new_x, new_y] = (abs(np.sin(r)*new_y) + abs(np.cos(r)*new_x),abs(np.sin(r)*new_x) + abs(np.cos(r)*new_y))
        # find the translation that moves the result to the center of that region.
        (tx, ty) = ((new_x-old_x)/2, (new_y-old_y)/2)
        rotation_matrix2D[0, 2] += tx # third column of matrix holds translation, which takes effect after rotation.
        rotation_matrix2D[1, 2] += ty
        rotated_img = cv2.warpAffine(img,rotation_matrix2D, dsize=(int(new_x), int(new_y)))
        return rotated_img


    def scaleAndCrop(self, img, out_width):
        # scale
        resized = imutils.resize(img, width=out_width)

        # crop where?
        grey = cv2.cvtColor(resized,cv2.COLOR_BGR2GRAY)
        [ret, thresh] = cv2.threshold(grey,10,255,cv2.THRESH_BINARY)
        #TODO: solve isssue : x,y,w,h = cv2.boundingRect(cnt) , TypeError: points is not a numpy array, neither a scalar

        # check to see if we are using OpenCV 2.X
        if imutils.is_cv2():
            [contours, hierarchy] = cv2.findContours(thresh, 1, 2)
            cnt = contours[0]

        # check to see if we are using OpenCV 3
        elif imutils.is_cv3():
            (_, cnts, _) = cv2.findContours(thresh, 1, 2)
            cnt = cnts[0]

        #x,y,w,h = cv2.boundingRect(cnt)
        [x, y, w, h] = cv2.boundingRect(cnt)
        crop = resized[y:y + h, x:x + w]
        return crop

    # (in_width, out_width, window_size, window_shift) = self.initScaling(self.image_width, in_scale, out_scale)
    # (self.image_width = 100,out_scale=1.0, in_scale=1.0)
    def initScaling(self, image_width, in_scale, out_scale):
        # compute scaling values for input and output images
        in_width = int(image_width*in_scale)
        window_size = (in_width*3, in_width*3) # this should be a large canvas, used to create container size

        out_width = int(window_size[0]*out_scale)
        window_shift = [in_width/2, in_width/2]
        # Print out data to terminal
        self.logger.info("Scaling input image widths from {} to {}".format(image_width, in_width))
        self.logger.info("Using canvas container width (input x2): {}".format(window_size[0]))
        self.logger.info("Scaling output image width from {} to {}".format(window_size[0], out_width))
        return (in_width, out_width, window_size, window_shift)

    # rotateImages uses csv file with pose to rezise and to know the yaw
    def rotateImages(self, poses):
        # only if there is not a .csv file
        # pre-process the images: resize and align by vehicle yaw (helps the matcher)
        for i, img in enumerate(self.images):
            # rotateimage_andCenter uses pose from csv file
            pose_i = poses[os.path.basename(self.filenames[i])] # get dict elements from filename
            yaw_i = pose_i['yaw'] # already in degrees as needed by opencv rotation function
            self.logger.info("Rotating image {} by {} degrees".format(i, yaw_i))
            self.images[i] = self.rotateimage_andCenter(self.images[i], yaw_i)


    def resizeImages(self,in_width):
        for i, img in enumerate(self.images):
            self.images[i] = imutils.resize(self.images[i], width=in_width)

    # --> This method does clahe on lab space to keep the color
    #transform to lab color space and conduct clahe filtering on l channel then merge and transform back
    def claheAdjustImages(self):
        #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        clahe = cv2.createCLAHE(clipLimit=6.0,tileGridSize=(8, 8))
        #self.logger.info("The clahe type {}".format(type(clahe.type)))
        for i, bgr_img, in enumerate(self.images):
            self.logger.info("color adjusting image {}".format(i))
            # first convert RGB color image to Lab
            lab_image = cv2.cvtColor(bgr_img, cv2.cv.CV_RGB2Lab)
            #cv2.imshow("lab_image before clahe", lab_image);

            # Extract the L channel
            lab_planes = cv2.split(lab_image)

            # then apply CLAHE algorithm to L channel
            #dst = []  #creating empty array to store L value
            #dst = clahe.apply(lab_planes[0])  # L channel = lab_planes[0]
            lab_planes[0] = clahe.apply(lab_planes[0])

            # Merge the the color planes back into an Lab image
            lab_image = cv2.merge(lab_planes,lab_planes[0])

            #cv2.imshow("lab_image after clahe", lab_image);

            # convert back to RGB space and store the color corrected image
            self.images[i] = cv2.cvtColor(lab_image, cv2.cv.CV_Lab2RGB)

            # overwrite old image
            #self.images[i] = cv2.imwrite(cl1)
            #cv2.imshow("image original", bgr_img);
            #cv2.imshow("image CLAHE", self.images[i]);
            cv2.waitKey()


    # process is the procedural happening step by step, similar to RTM program in LABVIEW
    def process(self, ratio=0.75, reproj_thresh=4.0, showMatches=False, out_scale=1.0, in_scale=1.0, is_infofile = False):

        # --> Color adjust the underwater images
        self.claheAdjustImages()

        # --> scale and rotate the input images accordingly
        (in_width, out_width, window_size, window_shift) = self.initScaling(self.image_width, in_scale, out_scale)

        # --> Resize images
        self.resizeImages(in_width)

        # TODO: if is_infofile then activate the rotateImages() call
        # rotateImages() uses csv file and pose --> super important, without mosaicing is to SLOW
        #if is_infofile==True:
        #    self.rotateImages(self.poses)
        ## Rotate images based on yaw data
            #

        # //-----------(2)  COMPUTE KEYPOINTS and DESCRIPTORS -----------//
        #--> extract the keypoints for each image frame
        kps = []
        features = []
        for i,img in enumerate(self.images):
            self.logger.info("Calculating SIFT features and keypoints for image {} of {}...".format(i+1,len(self.images)))
            (keypts, feats) = self.extractFeatures(img)
            kps.append(keypts)
            features.append(feats)

             # Show Kpts
            #img_kpts = cv2.drawKeypoints(img, keypts)
            #cv2.imshow("imgKpts", img_kpts)
            #cv2.waitKey(0)


        #--> Create the empty mosaic container --> 2D global reference frame
        # create some empty images for use in combining results
        base = np.zeros((window_size[1],window_size[0],3), np.uint8) # 3 color spaces RGB
        container = np.array(base)
        # add base image to the new container
        # TODO:, handle arbitrary base image
        base[window_shift[1]:self.images[0].shape[0]+window_shift[1], window_shift[0]:self.images[0].shape[1]+window_shift[0]] = self.images[0]

        # maybe problem when not using csv is the container consept
        # maybe use self.images[0]  instead of container
        container = self.addImage(base, container, transparent=False)

        # //-----------(3) COMPUTE KEYPOINTS MATCHING -----------//
        #--> calculate keypoints of newest container,
        #--> calculate matching
        #-->  apply transformation
        #-->  stitch into container
        container_kpts = []
        container_feats = []
        kps_matches_list = []
        homography_matrix_list = []
        for i, img in enumerate(self.images[:-1]):
            # find keypoints of new container
            self.logger.info("Calculating SIFT features and keypoints for container, iteration {} of {}...".format(i+1, len(self.images)-1))
            (container_kpts, container_feats) = self.extractFeatures(container)


            #--> compute matches between container points and next image
            self.logger.info("Computing matches for image {} of {}...".format(i+1,len(self.images)-1))
            # self.matchKeypoints() --> return (matches, homography_matrix, status)
            kps_matches = self.matchKeypoints(kps[i+1], container_kpts, features[i+1], container_feats, ratio, reproj_thresh)
            if kps_matches == None:
                self.logger.warning("this image had no kps_matches!")
                continue
            kps_matches_list.append(kps_matches)

            # To align the incoming images, an appropriate planar transformation,the homography matrix, has to be found.
            (matches, homography_matrix, status) = kps_matches_list[-1]
            homography_matrix_list.append(homography_matrix)

            #--> applying homography matrix in the planar transformation
            res = cv2.warpPerspective(self.images[i+1], homography_matrix_list[-1], window_size)

            # add image to container
            container = self.addImage(res, container, transparent=False)    # transparent=False)  # transparent....

            # todo: better edge blending, pyramids, etc.
            # Blending technicue here

            # todo: make isInfile method here
            # if is_infofile:

            #--> visualize the stitching, if termional command has -m
            if showMatches:
                img_matches = self.draw_matches(self.images[i+1], container, kps[i+1], container_kpts, matches, status)
                cv2.imshow("Matches", img_matches)
                cv2.waitKey(1)



        # scale the final output, and crop the container to remove excess blank space
        # (container needs to be big during processing since the transformations may deviate from base image location in any direction)
        # TODO: Fix this error from the following line
        scaled_container = self.scaleAndCrop(container, out_width)

        # --> draw final scaled output
        self.showMosaic(scaled_container)

    # --> draw final scaled output
    def showMosaic(self,scaled_container):
        cv2.imshow("Scaled Output",scaled_container)
        self.logger.info("Press space to close mosaic window...")
        cv2.waitKey(0)
        self.logger.info("The mosaic window is now closed")

    # add a new image to the mosaic
    def addImage(self, image, container, first=False, transparent=False):
        if transparent:
            alpha = 0.5
            beta = ( 1.0 - alpha )
            con = cv2.addWeighted(container, alpha, image, beta, 0.0)   # def addWeighted(src1, alpha, src2, beta, gamma, dst=None, dtype=None)
            cv2.imshow("Container", con)
            cv2.waitKey(0)
            return con

        # if the container is empty, just return the full image
        if first:
            return image


        # ELSE threshold both images, find non-overlapping sections, add to container
        # First make image and container grayscale
        grey_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        grey_container = cv2.cvtColor(container,cv2.COLOR_BGR2GRAY)

        # Apply treshold operator on gray_ image and container
        maxval = 10
        [ret,thresh_image] = cv2.threshold(grey_image,maxval,255,cv2.THRESH_BINARY)   # (src: Any, thresh: Any, maxval: Any, type: Any, dst: Any)
        [ret,thresh_container] = cv2.threshold(grey_container,maxval,255,cv2.THRESH_BINARY)

        # Todo:  Error message from intersect line
        # OpenCV Error: Sizes of input arguments do not match (The operation is neither 'array op array' (where arrays have the same si
        # ze and type), nor 'array op scalar', nor 'scalar op array') in cv::binary_op, file ..\..\..\..\opencv\modules\core\src\arithm
        # .cpp, line 1021
        # container = self.addImage(res, container, transparent=False)
        # File "mosaic.py", line 268, in addImage
        # intersect = cv2.bitwise_and(thresh_image, thresh_container)


        #  find intersection between container and new image
        intersect = cv2.bitwise_and(thresh_image, thresh_container) # find intersection between container and new image
        mask = cv2.subtract(thresh_image, intersect)   # subtract the intersection, leaving just the new part to union

        # Dilation
        # kernel = np.ones((2, 2), 'uint8') # for dilation below
        kernel = np.array([[1, 1],
                          [1, 1]])
        kernel.astype(dtype=np.uint8)

        mask = cv2.dilate(mask, kernel,iterations=1) # make the mask slightly larger so we don't get blank lines on the edges
        masked_image = cv2.bitwise_and(image, image, mask=mask) # apply mask

        con = cv2.add(container, masked_image)   # add the new pixels
        return con

    def extractFeatures(self, image):
        # convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # detect and extract features from the image
        # --> need SIFT because the mosaic can be made from camera that takes images with different Scale and Rotation

        # if we are using opencv 2.4.9
        if imutils.is_cv2():
            # the default parameters for cv2.SIFT()
            nfeatures = 0
            nOctaveLayers = 3
            contrastTreshold = 0.04
            edgeTreshold = 10
            sigma = 1.6
            sift = cv2.SIFT(nfeatures, nOctaveLayers, contrastTreshold, edgeTreshold, sigma)
            (kps, features) = sift.detectAndCompute(gray, None)   # (image, None)
        # check to see if we are using OpenCV 3
        elif imutils.is_cv3():
            descriptor = cv2.xfeatures2d.SIFT_create()
            (kps, features) = descriptor.detectAndCompute(gray, None)

        self.logger.info("Found {} keypoints in frame".format(len(kps)))

        # convert the keypoints from KeyPoint objects to np
        kps = np.float32([kp.pt for kp in kps])
        # return a tuple of keypoints and features
        return (kps, features)


    # matchKeypoints returns Homography
    # self.matchKeypoints(kps[i+1], container_kpts, features[i+1], container_feats, ratio, reproj_thresh)
    def matchKeypoints(self, kps_image, kps_container, features_img, features_container, ratio, reproj_thresh):
        # compute the raw matches and build list of actual matches that pass check

        #matcher = cv2.DescriptorMatcher_create("BruteForce")
        #raw_matches = matcher.knnMatch(features_img, features_container, 2)

        # from:  http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_matcher/py_matcher.html
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE,
                            trees = 5)
        search_params = dict(checks = 50)   # or pass empty dictionary
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        raw_matches = flann.knnMatch(features_img, features_container, k=2)

        ######################################################
        # Need to draw only good matches, so create a mask
        #matchesMask = [[0,0] for i in xrange(len(matches))]

        # ratio test as per Lowe's paper
        #for i,(m,n) in enumerate(matches):
        #    if m.distance < 0.7*n.distance:
        #        matchesMask[i]=[1,0]

        #draw_params = dict(matchColor = (0,255,0),
        #           singlePointColor = (255,0,0),
        #           matchesMask = matchesMask,
        #           flags = 0)

        #img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)

        #plt.imshow(img3,),plt.show()
        #########################################################


        # TODO: REMOVE OUTLIERS BASED ON ROV DIRECTION BETWEEN 2 images
        # maybe this is in homography

        self.logger.info("Found {} raw matches".format(len(raw_matches)))

        matches = []
        # loop over the raw matches and remove outliers
        for match in raw_matches:   # for i,(m,n) in enumerate(raw_matches):
            # ensure the distance is within a certain ratio of each
            # other (i.e. Lowe's ratio test)
            if len(match) == 2 and match[0].distance < match[1].distance * ratio:  # if m.distance < 0.7*n.distance:
                matches.append((match[0].trainIdx, match[0].queryIdx))

        self.logger.info("Found {} matches after Lowe's test".format(len(matches)))

        # //----- HOMOGRAPHY ----------------//
        # to compute a homography you need 4 or more matches
        if len(matches) > 4:
            # construct the two sets of points
            pts_image = np.float32([kps_image[i] for (_, i) in matches])
            pts_container = np.float32([kps_container[i] for (i, _) in matches])

            # compute the homography between the two sets of points
            (homography_matrix, status) = cv2.findHomography(pts_image, pts_container, cv2.RANSAC, reproj_thresh)

            # return the matches along with the homograpy matrix
            # and status of each matched point
            return (matches, homography_matrix, status)

        # else, then homograpy could not be computed
        self.logger.warning("Homography could not be computed!")
        return None

    # self.draw_matches(self.images[i+1], container, kps[i+1], container_kpts, matches, status)
    def draw_matches(self, image_1, image_2, kps_image, kps_container, matches, status):
        # initialize the output visualization image
        (height_1, width_1) = image_1.shape[:2]
        (height_2, width_2) = image_2.shape[:2]
        img_matches = np.zeros((max(height_1, height_2), width_1 + width_2, 3), dtype="uint8")
        img_matches[0:height_1, 0:width_1] = image_1
        img_matches[0:height_2, width_1:] = image_2

        # loop over the matches
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            # only process the match if the keypoint was successfully
            # matched
            if s == 1:
                # draw the match
                point_1 = (int(kps_image[queryIdx][0]), int(kps_image[queryIdx][1]))
                point_2 = (int(kps_container[trainIdx][0]) + width_1, int(kps_container[trainIdx][1]))
                cv2.line(img_matches, point_1, point_2, (0, 255, 0), 1)  # Calculates a line between point_1 and point_2

        # return the image with matches
        return img_matches



if __name__=="__main__":
    ### Call from terminal ###
    # python mosaic.py -d datasets/example2 -is 0.1 -os 0.9
    # to show matches
    # python mosaicProgram.py -d datasets/example1 -is 0.1 -os 0.9 -m

    # With isInfoFIle argument
    # python mosaic.py -d datasets/example1 -i True -is 0.1 -os 0.9 -m
    # python mosaic.py -d datasets/example1 -i True -is 2.0 -os 1.0 -m
    # python mosaic.py -d datasets/example1 -i True -is 0.01 -os 0.1 -m
    # python mosaic.py -d datasets/example1 -i True -is 10.0 -os 0.1 -m
    # python mosaic.py -d datasets/example1 -i True -is 1.0 -os 1.0 -m

    # python mosaic.py -d datasets\mosaic_camera\90deg_smallFolder  -i True -is 1.0 -os 1.0 -m



    ## Arguments that get passed in from the terminal
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dir", required=True, help="directory of images (jpg, png)")

    ap.add_argument("-i", "--is_infofile", default=False, type=bool, help="if there is a infofile in directory")

    # -is and -os are corelated

    # higher -is makes it go slower?
    ap.add_argument("-is", "--in_scale", default=1.0, type=float, help="ratio by which to scale the input images (faster processing)")

    # higher -os makes? -os = 1.0 gives same size, -os =0.1 gives smaller images
    ap.add_argument("-os", "--out_scale", default=1.0, type=float, help="ratio for scaling the output image")

    ap.add_argument("-m", "--showmatches", action='store_true', help="show intermediate matches and container")

    ap.add_argument("-u", "--underwaterfix", action='store_true', help="runs the Underwater_image_fix class")

    args = vars(ap.parse_args())

    ## Running the program

    # //-----------(1)  RETRIEVE INPUT DATA -----------//
    imgs = Images() # runs the images() class to 'preprocess' the images
    # Call imgs.loadFromDirectory to run all the methods in Images()
    imgs.loadFromDirectory(args['dir'], is_infofile=args['is_infofile'])   # ,is_infofile)

    # //--------------- Coorect images for underwater errors ------//
    # --> TODO: Correction of lens distortion --->C Camera calibration

    ## runs the MosaicClass() class to 'stitch' the images
    mosaic = MosaicClass(imgs, args['is_infofile'])
    # Call mosaic.process to run all the methods in MosaicClass
    # ratio=0.75 --> # other (i.e. Lowe's ratio test)
    #        if len(match) == 2 and match[0].distance < match[1].distance * ratio:
    #            matches.append((match[0].trainIdx, match[0].queryIdx))
    mosaic.process(ratio=0.75, reproj_thresh=4.0, showMatches=args['showmatches'], out_scale=args['out_scale'], in_scale=args['in_scale'])