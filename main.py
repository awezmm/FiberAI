print("Importing Modules")
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import bisect
import torch, torchvision, detectron2
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode

from skimage.morphology import skeletonize
import matplotlib.pyplot as plt 
from operator import itemgetter
import os
import csv
import scipy.io
from skimage import exposure
import argparse
import glob


script_path = ""


def imshow(img):
	cv2.imshow("w", img) 
	cv2.waitKey(0)  
	cv2.destroyAllWindows() 






def imadjust(img):

	p1, p99 = np.percentile(img, (1, 99))
	img_rescale = exposure.rescale_intensity(img, in_range=(p1, p99))

	return img_rescale.copy()


    
def thinning(BW):
	return (skeletonize(BW//255) * 255).astype(np.uint8).copy()


def inds(mask):
	thinnedMaskInds = np.where(mask == 255)
	thinnedMaskInds = np.asarray(mask).T
	thinnedMaskInds = sorted(mask, key=itemgetter(1))

	return thinnedMaskInds


def pureColor(thinnedMask, croppedColorMask, thinnedMaskInds):
	
	pure_colored = croppedColorMask.copy()
	thinnedMaskInds = np.where(thinnedMask == 255)
	thinnedMaskInds = np.asarray(thinnedMaskInds).T

	for ind in thinnedMaskInds:

		if croppedColorMask[ind[0]][ind[1]][1] > croppedColorMask[ind[0]][ind[1]][2]:
			pure_colored[ind[0]][ind[1]][0] = 0
			pure_colored[ind[0]][ind[1]][1] = 255
			pure_colored[ind[0]][ind[1]][2] = 0

		elif croppedColorMask[ind[0]][ind[1]][2] > croppedColorMask[ind[0]][ind[1]][1]:
			pure_colored[ind[0]][ind[1]][0] = 0
			pure_colored[ind[0]][ind[1]][1] = 0
			pure_colored[ind[0]][ind[1]][2] = 255


		else:
			pure_colored[ind[0]][ind[1]][0] = 0
			pure_colored[ind[0]][ind[1]][1] = 0
			pure_colored[ind[0]][ind[1]][2] = 0



	pure_overlay = cv2.bitwise_and(pure_colored,pure_colored, mask = thinnedMask)
	#imshow(pure_overlay)
	return pure_overlay


	

def boundaryInfo(img, thinnedMask, thinnedMaskInds):
	
	thinnedMaskInds = np.where(thinnedMask == 255)
	thinnedMaskInds = np.asarray(thinnedMaskInds).T
	thinnedMaskInds = sorted(thinnedMaskInds, key=itemgetter(1))

	
	objectsPattern= ""
	objectsLengths = []
	objectsIndices = []

	currentObjectColor = ''
	currentObjectLength = 1

	objectStartingInd = 0
	for i,ind in enumerate(thinnedMaskInds):

		if img[ind[0]][ind[1]][1] > img[ind[0]][ind[1]][2]:
			foundColor = 'G'
		elif img[ind[0]][ind[1]][2] > img[ind[0]][ind[1]][1]:
			foundColor = 'R'
		else:
			foundColor = 'B'


		if i == 0:
			currentObjectColor = foundColor

		if currentObjectColor != foundColor:
			objectsPattern = objectsPattern + currentObjectColor
			objectsLengths.append(currentObjectLength)
			objectsIndices.append([objectStartingInd, i-1])

			currentObjectColor = foundColor
			currentObjectLength = 1;
			objectStartingInd = i;

		else:
			if i != 0:
				currentObjectLength = currentObjectLength + 1


			if i == len(thinnedMaskInds) - 1:
				objectsPattern = objectsPattern + currentObjectColor
				objectsLengths.append(currentObjectLength)
				objectsIndices.append([objectStartingInd, i-1])



	return objectsPattern, objectsLengths, objectsIndices





def closeBlackGaps(img, thinnedMask, objectsPattern, objectsLengths, objectsIndices, thinnedMaskInds):
	output = img.copy()


	thinnedMaskInds = np.where(thinnedMask == 255)
	thinnedMaskInds = np.asarray(thinnedMaskInds).T
	thinnedMaskInds = sorted(thinnedMaskInds, key=itemgetter(1))

	#print(len(objectsPattern))
	for i,obj in enumerate(objectsPattern):

		if i != 0 and i != len(objectsPattern) - 1:
			#print("hit")
			if obj == 'B' and objectsPattern[i-1] == objectsPattern[i+1]:

				for j in thinnedMaskInds[objectsIndices[i][0]-1:objectsIndices[i][1]+1]:
					if objectsPattern[i-1] == 'R':
						output[j[0]][j[1]][1] = 0;
						output[j[0]][j[1]][2] = 255;
					else:
						output[j[0]][j[1]][1] = 255;
						output[j[0]][j[1]][2] = 0;

	
	return output


def closeBlackOverlayGaps(img, thinnedMask, objectsPattern, objectsLengths, objectsIndices, thinnedMaskInds):
	output = img.copy()

	thinnedMaskInds = np.where(thinnedMask == 255)
	thinnedMaskInds = np.asarray(thinnedMaskInds).T
	thinnedMaskInds = sorted(thinnedMaskInds, key=itemgetter(1))

	for i,obj in enumerate(objectsPattern):

		if i != 0 and i != len(objectsPattern) - 1:
			if obj == 'B':

				if objectsLengths[i-1] >= objectsLengths[i+1]:
					longerColor = objectsPattern[i-1]

				else:
					longerColor = objectsPattern[i+1]

				for j in thinnedMaskInds[objectsIndices[i][0]-1:objectsIndices[i][1]+1]:
					if longerColor == 'R':
						output[j[0]][j[1]][1] = 0;
						output[j[0]][j[1]][2] = 255;
					else:
						output[j[0]][j[1]][1] = 255;
						output[j[0]][j[1]][2] = 0;

	
	return output





def closeColoredGaps(img, thinnedMask, objectsPattern, objectsLengths, objectsIndices, thinnedMaskInds):

	output = img.copy()

	thinnedMaskInds = np.where(thinnedMask == 255)
	thinnedMaskInds = np.asarray(thinnedMaskInds).T
	thinnedMaskInds = sorted(thinnedMaskInds, key=itemgetter(1))

	for i,obj in enumerate(objectsPattern):

		if i != 0 and i != len(objectsPattern) - 1:

			if objectsPattern[i-1] != obj and obj != objectsPattern[i+1] and objectsPattern[i-1] == objectsPattern[i+1] and objectsLengths[i-1] > objectsLengths[i] and objectsLengths[i+1] > objectsLengths[i]:

				for j in thinnedMaskInds[objectsIndices[i][0]-1:objectsIndices[i][1]+1]:
					if objectsPattern[i-1] == 'R':
						output[j[0]][j[1]][1] = 0;
						output[j[0]][j[1]][2] = 255;
					else:
						output[j[0]][j[1]][1] = 255;
						output[j[0]][j[1]][2] = 0;

	
	return output








				





def preprocess(img, z_STANDARD = 0.95):

	hsvImage = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	hChannel = hsvImage[:,:,0]
	sChannel = hsvImage[:,:,1]
	vChannel = hsvImage[:,:,2]

	vChannelMean = np.mean(vChannel)
	vChannelStd = np.std(vChannel)
	
	threshValue = vChannelMean + vChannelStd * z_STANDARD

	ret, Vthresh_mask =  cv2.threshold(vChannel, threshValue, 255, cv2.THRESH_BINARY)
	
	redChannel = img[:,:,2]
	greenChannel = img[:,:,1]
	blueChannel = img[:,:,0]

	redChannelEH = imadjust(redChannel)
	greenChannelEH = imadjust(greenChannel)

	rgbEH = cv2.merge((blueChannel,greenChannelEH,redChannelEH))
	cv2.imwrite("eh.png", rgbEH)


	result = cv2.bitwise_and(rgbEH,rgbEH, mask = Vthresh_mask)
	
	return result.copy()


def buildPredictor(model_path = "model_final.pth", threshold = 0.5, num_class = 4):

	model_full_path = script_path + model_path
	cfg = get_cfg()

	cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
	cfg.MODEL.WEIGHTS = model_full_path
	cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
	cfg.MODEL.DEVICE = "cpu"
	cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_class
	predictor = DefaultPredictor(cfg)
	return predictor



def getMeasurements(img, outputs, imgNum):

	instances = outputs["instances"]
	boundingBoxes = instances.pred_boxes.tensor.numpy()
	scores = instances.scores.numpy()
	predMasks = instances.pred_masks.numpy()

	num_instances = len(scores)

	
	
	csvData = [["Image", "Fiber", "Red", "Green"]]
	

	if not os.path.isdir("CroppedFibers"):
		os.mkdir("CroppedFibers")

	if not os.path.isdir("FiberSegmentations"):
		os.mkdir("FiberSegmentations")

	#plt.figure(figsize=(14,14))
	#plt.figure()
	for i in range(0, num_instances):

		if i == 9 or i == 6:
			continue


		curr_boundingBoxes = boundingBoxes[i]
		curr_predMask = predMasks[i]
		x1 = int(curr_boundingBoxes[0] - 1)
		y1 = int(curr_boundingBoxes[1] - 1)
		x2 = int(curr_boundingBoxes[2] + 1)
		y2 = int(curr_boundingBoxes[3] + 1)




		cropped_image = img[y1:y2, x1:x2]

		cropped_predMask = curr_predMask[y1:y2, x1:x2].astype(np.uint8) * 255

		
		

		thinned_predMask = thinning(cropped_predMask)
		

		thinnedMaskInds = inds(thinned_predMask)

		pure_colored = pureColor(thinned_predMask, cropped_image, thinnedMaskInds)


		plt.figure(figsize=(14,14))

		

		plt.subplot(161)
		plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
		plt.axis('off')

		plt.subplot(162)
		plt.imshow(cv2.cvtColor(cropped_predMask, cv2.COLOR_BGR2RGB))
		plt.axis('off')


		#plt.subplot2grid((10,5), (i,1))
		plt.subplot(163)
		plt.imshow(cv2.cvtColor(pure_colored, cv2.COLOR_BGR2RGB))
		plt.axis('off')

		
		objectsPattern, objectsLengths, objectsIndices = boundaryInfo(pure_colored, thinned_predMask, thinnedMaskInds)
		output = closeBlackGaps(pure_colored, thinned_predMask, objectsPattern, objectsLengths, objectsIndices, thinnedMaskInds)

		plt.subplot(164)
		#plt.subplot2grid((10,5), (i,2))
		plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
		plt.axis('off')
		objectsPattern, objectsLengths, objectsIndices = boundaryInfo(output, thinned_predMask, thinnedMaskInds)
		output = closeBlackOverlayGaps(output, thinned_predMask, objectsPattern, objectsLengths, objectsIndices, thinnedMaskInds)

		#plt.subplot2grid((10,5), (i,3))
		plt.subplot(165)
		plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
		plt.axis('off')
		objectsPattern, objectsLengths, objectsIndices = boundaryInfo(output, thinned_predMask, thinnedMaskInds)
		output = closeColoredGaps(output, thinned_predMask, objectsPattern, objectsLengths, objectsIndices, thinnedMaskInds)

		#plt.subplot2grid((10,5), (i,4))
		plt.subplot(166)
		plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
		plt.axis('off')
		plt.savefig('compin'+str(i)+'.png', bbox_inches='tight')
		plt.close()
		#plt.show()
		#break





		
		redLength = len(np.where(output[:,:,2] == 255)[0])
		greenLength = len(np.where(output[:,:,1] == 255)[0])


		csvData.append([imgNum, i+1, redLength, greenLength])

		


		cv2.imwrite("CroppedFibers/Fiber"+str(i)+".png", cropped_image)
		cv2.imwrite("FiberSegmentations/Fiber"+str(i)+".png", output)
		



	with open('outputs.csv', 'w', newline='') as file:
		writer = csv.writer(file)
		writer.writerows(csvData)


	
	bLabels = list(range(1,num_instances+1))
	bLabels = [str(i) for i in bLabels]
	v = Visualizer(img, scale=0.8, instance_mode=ColorMode.IMAGE_BW)
	v = v.overlay_instances(boxes=instances.pred_boxes, labels=bLabels, masks=instances.pred_masks, alpha=0.1)
	plt.figure(figsize = (14, 10))
	plt.imshow(v.get_image()[:, :, ::-1])
	plt.savefig('visualization.png', bbox_inches='tight')
	

	return csvData, boundingBoxes

	


	

		

		
		
		





def main_analyze(img_paths, output_folder):

	


	print("Building Predictor")
	predictor = buildPredictor()


	print("Analyzing Images")

	output_directory = output_folder
	os.chdir(output_directory)



	allCSVData = [["Image", "Fiber", "Red", "Green"]]


	allBoundingboxes = []
	allFiberLengths = []

	for i,img_path in enumerate(img_paths):

		
		orig_img = cv2.imread(img_path)
		height, width, channels = orig_img.shape

		if channels != 3:
			print("Error!, image " + img_path + " does not have 3 chanels. It will be skipped.")
			continue

		resized_img = orig_img
		if height != 1024 or width != 1024:
			print("Image " + img_path + " will be resized to dimensions of 1024 x 1024")
			resized_img = cv2.resize(orig_img, (1024, 1024))


		preprocessed_img = resized_img.copy()
		cv2.imwrite("finalpreprocess.png", preprocessed_img)
		cv2.imwrite("original.png", resized_img)
		
		

		outputs = predictor(preprocessed_img)
		instances = outputs['instances']

		filtered_instances = instances[instances.pred_classes == 2]
		outputs = {"instances": filtered_instances}
	

	

		if not os.path.isdir("Image"+str(i+1)):
			os.mkdir("Image"+str(i+1))
		os.chdir("Image"+str(i+1))
		currCSVData, currboundingBoxes = getMeasurements(preprocessed_img, outputs, i+1)

		currCSVData.pop(0)

		allCSVData.extend(currCSVData)
		
		currCSVData = np.array(currCSVData)
		currFiberLengths = currCSVData[:,[2,3]]
		allFiberLengths.append(currFiberLengths)

		allBoundingboxes.append(currboundingBoxes)

		os.chdir("..")

	with open('outputs.csv', 'w', newline='') as file:
		writer = csv.writer(file)
		writer.writerows(allCSVData)

	#print(allFiberLengths)
	scipy.io.savemat('outputs.mat', {"imgPaths": img_paths, "fiberLengths": allFiberLengths,  "boundingBoxes": allBoundingboxes})
	


if __name__ == "__main__":

	script_path = os.path.dirname(os.path.realpath(__file__)) + "/"

	parser = argparse.ArgumentParser()
	parser.add_argument("-image", "-image", default=".")
	parser.add_argument("-folder", "-folder", default=".")
	parser.add_argument("-out", "-out", default=".")
	
	
	args = parser.parse_args()

	if args.image == "." and args.folder == ".":
		sys.exit("Error: Either provide a single input image or a input folder of images sized 1024 x 1024")


	if args.out == ".":
		sys.exit("Error: Create an output folder to save result in, and provide its full path")

	if args.image != ".":
		os.chdir(args.out)
		main_analyze(args.image, args.out)
	else:

		os.chdir(args.folder)
		types = ('*.png', '*.tif', '*.tiff', '*.jpg',  '*.jpeg')
		image_files = []
		for files in types:
			image_files.extend(glob.glob(files))

		abspath = os.path.abspath('.') + '/'
		image_files = [abspath + fp  for fp in image_files]
		os.chdir(args.out)
		main_analyze(image_files, args.out)










	
