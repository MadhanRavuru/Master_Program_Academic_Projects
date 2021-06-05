# Homework 1.1

"master" contains only the exercise files from course website (http://campar.in.tum.de/Chair/TeachingWs19TDCV) and from moodle.

From course website, data.zip is available and from moodle, code.zip is available.

1.1 a

PnP algorithm 3D/2D correspondences

I generated 'labeled_points.mat' by clicking the visible corners in proper order. We can load this file and use it for 'image points' for 8 images.

NOTE: The world coordinate system origin is at vertex 6 (lower left corner of image). Refer vertex numbering in code/vertices.png
This is little ambiguous. I referrred to the homework pdf(to check the world coordinate system in the picture of object pose).

I found the dimensions of the box using vertices and normals of polygon format file
length=0.165  (x axis)
width=0.063   (y axis)
height=0.093  (z axis)
Using these information, we get world coordinate system of the teabox.

cameraparams are found using intrinsic matrix specified from cameraParameters()

Maxreprojerr is set to 1 (according to documentation, default is 1). If the value is high, compuation is high, but accuracy is low.




