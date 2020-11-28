import sys
import cv2
from pathlib import Path
import os
import glob
from tqdm import tqdm

finalSZ = 1024

def imgOp(img, filenameStem, filenameExt):

	height, width, channels = img.shape

	# if image is already 1024 x 1024
	if height == finalSZ and width == finalSZ:
		cv2.imwrite(filenameStem+filenameExt, img)


	# if image is larger than 1024 x 1024 and dimensions
	# are multiples of 1024 (2048, 4096,...)
	# break into equal pieces
	elif height % finalSZ == 0 and width % finalSZ == 0:

		multiplier = int(height / finalSZ)

		counter = 0

		for i in range(0, multiplier):
			for j in range(0, multiplier):

				sub_image = img[0+(finalSZ*i):finalSZ*(i+1), 0+(finalSZ*j):finalSZ*(j+1)]

				
				
				counter = counter+1
				subfilename = filenameStem + "-" + str(counter) + filenameExt

				
				cv2.imwrite(subfilename, sub_image)




	else:

		# if image dimensions are less than 1024 x 1024, 
		# then resize to 1024 x 1024
		if height < finalSZ and width < finalSZ:

			resized_image = cv2.resize(image, (finalSZ,finalSZ))
			rszfilename = filenameStem + "-" + "resized" + filenameExt

			cv2.imwrite(rszfilename, sub_image)


		else:

			# all other cases, skip image
			print("Skipping image " + filenameStem + "as it is bigger than 512x512 and not a multple of 512.")

   


	


if __name__ == "__main__": 

	if len(sys.argv) != 4:
		sys.exit("Error: Invalid number of arguments given")

	if str(sys.argv[2]) != "f" and str(sys.argv[2]) != "d":
		sys.exit("Error: Invalid argument given: " + str(sys.argv[2]))





	if str(sys.argv[2]) == "f":

		img = cv2.imread(sys.argv[1])
		filenameStem = Path(sys.argv[1]).stem
		filenameExt = Path(sys.argv[1]).suffix

		height, width, channels = img.shape

		if height != width:
			print("Skipping image " + filenameStem + " as height does not equal width")

		else:
			os.chdir(sys.argv[3])
			imgOp(img, filenameStem, filenameExt)





	else:
		
		os.chdir(sys.argv[1])
		types = ('*.png', '*.tif', '*.tiff', '*.jpg',  '*.jpeg')
		image_files = []
		for files in types:
			image_files.extend(glob.glob(files))

		abspath = os.path.abspath('.') + '/'
		image_files = [abspath + fp  for fp in image_files]

		#os.chdir("../")
		os.chdir(sys.argv[3])

		#print("Files found for resizing:")
		#print(image_files)

		for image_file in tqdm(image_files):

			print(image_file)
			img = cv2.imread(image_file)

			filenameStem = Path(image_file).stem
			filenameExt = Path(image_file).suffix

			height, width, channels = img.shape

			if height != width:
				print("Skipping image " + filenameStem + " as height does not equal width")

			else:
				
				imgOp(img, filenameStem, filenameExt)



