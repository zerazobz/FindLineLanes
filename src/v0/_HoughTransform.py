# Hough Transform

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2


def ShowImageData(image, name):
	print()
	print("----------------------------------------------------------------->")
	print(name)
	print("The dimensions of: %s" % (name))
	print("\t" + str(image.shape))
	plt.imshow(image)
	plt.suptitle(name)
	plt.show()

# Read in and grayscale the image
# image = mpimg.imread('exit-ramp.jpg')
image = mpimg.imread('./../../test_images/solidYellowLeft.jpg')
# image = mpimg.imread('test.jpg')
gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
ShowImageData(gray, "gray after cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)")

# Define a kernel size and apply Gaussian smoothing
kernel_size = 5
blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)
ShowImageData(blur_gray, "blur_gray after cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)")

# Define our parameters for Canny and apply
low_threshold = 50
high_threshold = 150
edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
ShowImageData(edges, "edges after cv2.Canny(blur_gray, low_threshold, high_threshold)")

# Next we'll create a masked edges image using cv2.fillPoly()
mask = np.zeros_like(edges)   
ignore_mask_color = 255   
ShowImageData(mask, "mask")

# This time we are defining a four sided polygon to mask
imshape = image.shape
# vertices = np.array([[(0,imshape[0]),(450, 290), (imshape[1], 0), (imshape[1],imshape[0])]], dtype=np.int32)
vertices = np.array([[(0,imshape[0]),(450, 290), (490, 290), (imshape[1],imshape[0])]], dtype=np.int32)
print(vertices)
# plt.plot(x, y, 'b--', lw=4)

cv2.fillPoly(mask, vertices, ignore_mask_color)
ShowImageData(mask, "mask after cv2.fillPoly")
masked_edges = cv2.bitwise_and(edges, mask)
ShowImageData(masked_edges, "masked_edges after cv2.bitwise_and(edges, mask)")

# Define the Hough transform parameters
# Make a blank the same size as our image to draw on
rho = 2 # distance resolution in pixels of the Hough grid
theta = 1 * np.pi/180 # angular resolution in radians of the Hough grid
threshold = 15     # minimum number of votes (intersections in Hough grid cell)
min_line_length = 40 #minimum number of pixels making up a line
max_line_gap = 20    # maximum gap in pixels between connectable line segments
line_image = np.copy(image)*0 # creating a blank to draw lines on
ShowImageData(line_image, "line_image after np.copy(image)*0")

# Run Hough on edge detected image
# Output "lines" is an array containing endpoints of detected line segments
lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)
print()
print("lines")
print("dimensions %s" % (str(lines.shape)))
print("size %s" % (str(lines.size)))
allx1, allx2 = [], []
# Iterate over the output "lines" and draw lines on a blank image
iterator = 0
minLeftX, minLeftY, maxLeftX, maxLeftY =  imshape[1], 0, 0, imshape[0]
minRightX, minRightY, maxRightX, maxRightY = imshape[1], imshape[0], 0, 0

print("imshape")
print(imshape)

for line in lines:
    print("print(line)")
    print(line)
    for x1,y1,x2,y2 in line:
        if (y2 - y1) / (x2 - x1) > 0:
            # print("Right")
            # print("x1: %d, y1: %d, x2: %d, y2: %d" % (x1, y1, x2, y2))
            if x1 < minRightX:
                minRightX = x1
            if y1 < minRightY:
                minRightY = y1
            if x2 > maxRightX:
                maxRightX = x2
            if y2 > maxRightY:
                maxRightY = y2
            # cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0), 2)
        elif (y2 - y1) / (x2 - x1) < 0:
            # print("Left")
            # print("x1: %d, y1: %d, x2: %d, y2: %d" % (x1, y1, x2, y2))
            if x1 < minLeftX:
                minLeftX = x1
            if y1 > minLeftY:
                minLeftY = y1
            if x2 > maxLeftX:
                maxLeftX = x2
            if y2 < maxLeftY:
                maxLeftY = y2
            # cv2.line(line_image,(x1,y1),(x2,y2),(76,201,91), 2)
        iterator += 1

print("Values of minLeftX, minLeftY, maxLeftX, maxLeftY are %d, %d, %d, %d" % (minLeftX, minLeftY, maxLeftX, maxLeftY))
print("Values of minRightX, minRightY, maxRightX, maxRightY are %d, %d, %d, %d" % (minRightX, minRightY, maxRightX, maxRightY))

cv2.line(line_image,(minLeftX,minLeftY),(maxLeftX,maxLeftY),(255,0,0), 12)
cv2.line(line_image,(minRightX,minRightY),(maxRightX,maxRightY),(76,201,91), 12)

ShowImageData(line_image, "line_image after Hough Transform and painting red")

ShowImageData(edges, "Currently edges")
# Create a "color" binary image to combine with line image
color_edges = np.dstack((edges, edges, edges)) 
print("What is color_edges")
print(type(color_edges))
print(str(color_edges.shape))

# Draw the lines on the edge image
# lines_edges = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0) 
lines_edges = cv2.addWeighted(image, 0.8, line_image, 1, 0) 
plt.imshow(lines_edges)
plt.show()
