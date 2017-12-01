#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
# %matplotlib inline
import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, imshape, color=[255, 0, 0], thickness=20):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
#     for line in lines:
#         for x1,y1,x2,y2 in line:
#             cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    iterator = 0
    minLeftX, minLeftY, maxLeftX, maxLeftY =  imshape[1], 0, 0, imshape[0]
    minRightX, minRightY, maxRightX, maxRightY = imshape[1], imshape[0], 0, 0

    for line in lines:
        for x1,y1,x2,y2 in line:
            if (y2 - y1) / (x2 - x1) > 0:
                if x1 < minRightX:
                    minRightX = x1
                if y1 < minRightY:
                    minRightY = y1
                if x2 > maxRightX:
                    maxRightX = x2
                if y2 > maxRightY:
                    maxRightY = y2
            elif (y2 - y1) / (x2 - x1) < 0:
                if x1 < minLeftX:
                    minLeftX = x1
                if y1 > minLeftY:
                    minLeftY = y1
                if x2 > maxLeftX:
                    maxLeftX = x2
                if y2 < maxLeftY:
                    maxLeftY = y2
            iterator += 1

    print("Values of minLeftX, minLeftY, maxLeftX, maxLeftY are %d, %d, %d, %d" % (minLeftX, minLeftY, maxLeftX, maxLeftY))
    print("Values of minRightX, minRightY, maxRightX, maxRightY are %d, %d, %d, %d" % (minRightX, minRightY, maxRightX, maxRightY))

    cv2.line(img,(minLeftX,minLeftY),(maxLeftX,maxLeftY),color, thickness)
    cv2.line(img,(minRightX,minRightY),(maxRightX,maxRightY),color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines, line_img.shape)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)



print()
print("Empezando el procesamiento...")
# TODO: Build your pipeline that will draw lane lines on the test_images
# then save them to the test_images directory.

kernelsize = 5
low_threshold = 50
high_threshold = 150
rho = 2 # distance resolution in pixels of the Hough grid
theta = 1 * np.pi/180 # angular resolution in radians of the Hough grid
threshold = 15     # minimum number of votes (intersections in Hough grid cell)
min_line_length = 40 #minimum number of pixels making up a line
max_line_gap = 20    # maximum gap in pixels between connectable line segments


# for imagePath in os.listdir("test_images/"):
for imagePath in ['solidYellowLeft.jpg']:
    print()
    print("We are processing the image: %s"%(imagePath))
    image = mpimg.imread('test_images/' + imagePath)
    plt.suptitle("image")
    plt.imshow(image)
    plt.show()

    imshape = image.shape
    
    grayImage = grayscale(image)
    blurImage = gaussian_blur(grayImage, kernelsize)
    cannyImage = canny(blurImage, low_threshold, high_threshold)
    
    vertices = np.array([[(0,imshape[0]),(450, 290), (490, 290), (imshape[1],imshape[0])]], dtype=np.int32)
    maskedImage = region_of_interest(cannyImage, vertices)
    
    plt.suptitle("maskedImage")
    plt.imshow(maskedImage)
    plt.show()

    imageWithLines = hough_lines(maskedImage, rho, theta, threshold, min_line_length, max_line_gap)
    # color_edges = np.dstack((cannyImage, cannyImage, cannyImage)) 
    finalImage = weighted_img(imageWithLines, image, α=0.8, β=1., λ=0.)

    print("The image with name: %s"%(imagePath))
    
    plt.suptitle("finalImage")
    plt.imshow(finalImage)
    plt.show()

#     plt.imshow(finalImage)
    mpimg.imsave("test_images_output/" + imagePath.rsplit( ".", 1 )[ 0 ] + ".png", finalImage)


