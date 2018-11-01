# This example is based off of the basic examples in the documentation. Changed
# so that it actually writes new images and has ways to control and keep
# track of how many. 

from imgaug import augmenters as iaa
import cv2

# Define the sequence of filters to apply to the image set
seq = iaa.Sequential([
	iaa.Fliplr(0.5),  # horizontal flips
	iaa.Crop(percent=(0, 0.1)),  # random crops

	# Small gaussian blur with random sigma between 0 and 0.5.
	# But we only blur about 50% of all images.
	iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.5))),

	# Strengthen or weaken the contrast in each image.
	iaa.ContrastNormalization((0.75, 1.5)),

	# Add gaussian noise.
	# For 50% of all images, we sample the noise once per pixel.
	# For the other 50% of all images, we sample the noise per pixel AND
	# channel. This can change the color (not only brightness) of the
	# pixels.
	iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),

	# Make some images brighter and some darker.
	# In 20% of all cases, we sample the multiplier once per channel,
	# which can end up changing the color of the images.
	iaa.Multiply((0.8, 1.2), per_channel=0.2),

	# Apply affine transformations to each image.
	# Scale/zoom them, translate/move them, rotate them and shear them.
	iaa.Affine(
		scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
		translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
		rotate=(-25, 25),
		shear=(-8, 8)
	)
], random_order=True)  # apply augmenters in random order

# create an empty list to append images to
img_list = []

# use opencv imgread to load the image file and store it as img
img = cv2.imread('test_image.jpg')

# to augment more than one image just make a loop that appends the images in
# a folder.

# append it to the img_list list
img_list.append(img)

# this variable controls how many augmented images you want per image you are
# Augmenting.
aug_count = 50

# This variable is to keep track of how many new variations we have created
# to print at the end of the program.
new_img_count = 0

for i in range(0, aug_count):
	if i % 25 == 0:
		print(str(round((new_img_count / aug_count) * 100)) + "% " + "complete")
	# apply the sequence of filters to the list of images and store them in
	images_aug = seq.augment_images(img_list)
	for index, image in enumerate(images_aug):
		# name the file using the current count of augmentation passes over
		# the list of images and the index of the image in the list of images
		# we are currently augmenting. Place it in /output/
		file_name = "./output/new" + str(i) + "_" + str(index) + ".jpg"

		# write the new image file using opencv
		cv2.imwrite(file_name, image)

		# increment the new_img_count
		new_img_count += 1

# print the number of new images created
print("You created " + str(new_img_count) + " new variations")
