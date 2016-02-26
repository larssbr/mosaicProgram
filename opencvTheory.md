
# Drawing keypoints
cv2.drawKeypoints() to draw keypoints

cv2.drawMatches() helps us to draw the matches
img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10], flags=2)

cv2.drawMatchesKnn() which draws all the k best matches. If k=2,
 it will draw two match-lines for each keypoint. 
 So we have to pass a mask if we want to selectively draw it.