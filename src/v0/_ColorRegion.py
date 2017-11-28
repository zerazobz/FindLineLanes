# Color Region
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

def printTwoFirstAndOneLas(array, name):
	print("")
	print("----------------------------------------------------------------->")
	print("las longitudes y valores del array '%s' son: " % (name))
	print(array.shape)
	print("\t" + str(array.shape[0])) # y axis
	print("\t" + str(array.shape[1])) # x axis

	if array.shape[0] > 0 and array.shape[1] > 0:
		print("The element [0][0] is:")
		print("\t" + str(array[0][0]))

	if array.shape[0] > 0 and array.shape[1] > 1:
		print("The element [0][1] are:")
		print("\t" + str(array[0][1]))

	if array.shape[0] > 2 and array.shape[1] > 1:
		print("The element [0][2] are:")
		print("\t" + str(array[0][2]))

	if array.shape[0] > 1 and array.shape[1] > 0:
		print("The element [1][0] are:")
		print("\t" + str(array[1][0]))

	if array.shape[0] > 1 and array.shape[1] > 1:
		print("The element [1][1] are:")
		print("\t" + str(array[1][1]))

	maxxlength = array.shape[0] - 1
	maxylength = array.shape[1] - 1

	if maxxlength > 0 and maxylength > 0:
		print("The last element at [%s][%s] are:" % (str(maxxlength), str(maxylength)))
		print("\t" + str(array[maxxlength][maxylength]))

def lookForWhitePixels(image):
	print()
	print("----------------------------------------------------------------->")
	print("Looking for white pixels")
	# print(image)

	iterator = 0

	for x in image:
		for y in x:
			# if (y == [255, 255, 255]).all():
			if not np.array_equal(y, [0, 0, 0]):
				iterator += 1
	print("The total of white pixles are %s" % (iterator))

def getTruePointsandIndexes(booleanArray, originalIndexArray):
	print()
	print("----------------------------------------------------------------->")
	print("The dimensions of the boolean array are: %s" %(str(booleanArray.shape)))
	print("The dimensions of the original index array are: %s" %(str(originalIndexArray.shape)))
	print("Index list")
	print(booleanArray)
	np.where(booleanArray == True)
	# np.asarray(np.where(booleanArray == True)).T.tolist()	
	# for x in booleanArray:
	# 	for y in x:
	# 		if y == True:
	# 			print(y)

# Read in the image
image = mpimg.imread('test_images/solidWhiteRight.jpg')

# Grab the x and y size and make a copy of the image
ysize = image.shape[0]
xsize = image.shape[1]
print('The width is: ' + str(xsize))
print('The height is: ' + str(ysize))
print('-----------------------')

color_select = np.copy(image)
line_image = np.copy(image)

# Define color selection criteria
# MODIFY THESE VARIABLES TO MAKE YOUR COLOR SELECTION
red_threshold = 200
green_threshold = 200
blue_threshold = 200

rgb_threshold = [red_threshold, green_threshold, blue_threshold]
#	>>>	This is getting a grayscale

# Define the vertices of a triangular mask.
# Keep in mind the origin (x=0, y=0) is in the upper left
# MODIFY THESE VALUES TO ISOLATE THE REGION 
# WHERE THE LANE LINES ARE IN THE IMAGE
left_bottom = [130, 539]
right_bottom = [880, 539]
apex = [470, 295]

# Perform a linear fit (y=Ax+B) to each of the three sides of the triangle
# np.polyfit returns the coefficients [A, B] of the fit
print('Left Bottom[0]: ' + str(left_bottom[0]))
print('Apex[0]: ' + str(apex[0]))
print('Left Bottom[1]: ' + str(left_bottom[1]))
print('Apex[1]: ' + str(apex[1]))

fit_left = np.polyfit((left_bottom[0], apex[0]), (left_bottom[1], apex[1]), 1)
# fit_left= np.polyfit((left_bottom[0], left_bottom[1]), (apex[0], apex[1]), 1)
fit_right = np.polyfit((right_bottom[0], apex[0]), (right_bottom[1], apex[1]), 1)
fit_bottom = np.polyfit((left_bottom[0], right_bottom[0]), (left_bottom[1], right_bottom[1]), 1)
print('After linear fit:')
print("The coeficient for the left line are: ")
print("\t" + str(fit_left))
print("The coeficient for the right line are: ")
print("\t" + str(fit_right))
print("The coeficient for the bottom line are: ")
print("\t" + str(fit_bottom))

lax = [left_bottom[0], apex[0]]
lay = [left_bottom[1], apex[1]]

plt.plot(lax, lay, 'y--', lw=4)

rax = [right_bottom[0], apex[0]]
ray = [right_bottom[1], apex[1]]

plt.plot(rax, ray, 'g--', lw=4)

bax = [left_bottom[0], right_bottom[0]]
bay = [left_bottom[1], right_bottom[1]]

plt.plot(bax, bay, 'p--', lw=4)

print()
print("Image before")
print("Image dimensions")
print(image.shape)
print("image[0][0]")
print("\t" + str(image[0][0]))
print("image[0][0][0]")
print("\t" + str(image[0][0][0]))
print("image[0][0][1]")
print("\t" + str(image[0][0][1]))
print("image[0][0][2]")
print("\t" + str(image[0][0][2]))

# Mask pixels below the threshold
color_thresholds = (image[:,:,0] < rgb_threshold[0]) | \
                    (image[:,:,1] < rgb_threshold[1]) | \
                    (image[:,:,2] < rgb_threshold[2])
print()
print("Color Threshold")
print("Color Dimensions")
print(color_thresholds.shape)
printTwoFirstAndOneLas(color_thresholds, "Color Thresholds")

totalFalse, totalTrue = 0, 0

print()
for x in color_thresholds:
	for y in x:
		if y == False:
			totalFalse += 1
		else:
			totalTrue += 1

print("False incidents: %s" % (str(totalFalse)))
print("True incidents: %s" % (str(totalTrue)))

# Find the region inside the lines
XX, YY = np.meshgrid(np.arange(0, xsize), np.arange(0, ysize))
print()
print('meshgrid')
print("XX.shape")
print(XX.shape)
printTwoFirstAndOneLas(XX, "meshgrid -- XX")

print("YY.shape")
print(YY.shape)
printTwoFirstAndOneLas(YY, "meshgrid -- YY")

print()
print('Polynomio: ')
# print("XX*fit_left[0] + fit_left[1])")
# print(XX*fit_left[0] + fit_left[1])
polynomioleft = XX*fit_left[0] + fit_left[1]
print("polynomio.shape")
print(polynomioleft.shape)

printTwoFirstAndOneLas(polynomioleft, "Polinomio Left [XX*fit_left[0] + fit_left[1]]")

region_thresholds = (YY > (XX*fit_left[0] + fit_left[1])) & \
                    (YY > (XX*fit_right[0] + fit_right[1])) & \
                    (YY < (XX*fit_bottom[0] + fit_bottom[1]))
printTwoFirstAndOneLas(region_thresholds, "Region threshold")
print("getTruePointsandIndexes")
getTruePointsandIndexes(region_thresholds, XX*fit_left[0] + fit_left[1])

# Mask color and region selection
# color_select[color_thresholds | ~region_thresholds] = [0, 0, 0]
color_select[region_thresholds] = [0, 0, 0]

# Color pixels red where both color and region selections met
# line_image[~color_thresholds & region_thresholds] = [255, 0, 0]
line_image[~color_thresholds & region_thresholds] = [255, 0, 0]
printTwoFirstAndOneLas(line_image, "line image ndarray")

# Display the image and show region and color selections
plt.imshow(image)
x = [left_bottom[0], right_bottom[0], apex[0], left_bottom[0]]
y = [left_bottom[1], right_bottom[1], apex[1], left_bottom[1]]

# plt.plot(x, y, 'b--', lw=4)
plt.imshow(color_select)
plt.suptitle("color_select")
plt.show()

# imageToSearch = np.copy(color_select)
# lookForWhitePixels(imageToSearch)
# plt.imshow(imageToSearch)
# plt.suptitle("imageToSearch")
# plt.show()

# plt.plot(x, y, 'b--', lw=4)
plt.imshow(line_image)
plt.suptitle("line_image")
plt.show()