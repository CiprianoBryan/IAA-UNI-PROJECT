import cv2
import os

import numpy
import random
from torch.utils.data import Dataset

imagesYesPath = "Dataset/Yes/"
imagesNotPath = "Dataset/Not/"
imagesPath = [imagesNotPath, imagesYesPath]

SHAPE = 20

def segmentize (filePath, shape_image):
    image = cv2.imread(filePath)
    print('\t\tFile Shape:', image.shape)
    height, width, dept = image.shape # (14879, 14498, 3)

    n = height//shape_image
    m = width//shape_image
    images_div = []
    percent = 0
    for i in range(n):
        for j in range(m):
            id = j + i*m
            image_div = image[i*shape_image:i*shape_image+shape_image, j*shape_image:j*shape_image+shape_image]
            if (id + 1)*100//(n*m) == percent:
                percent += 10
                print("\t\tImages div: {}/{}, {}%".format(id + 1, n*m, (id + 1)*100//(n*m)))
            images_div.append(image_div)
            # DELETE
            if percent == 20:
                return images_div
            # DELETE
    return images_div

def joinBandsRegion(imagesRegion):
    # Swap(Band, Divs): (4 x (n*m) x 20x20x3) -> ((n*m) x 4 x20x20x3)
    imagesRegion = numpy.array(imagesRegion)
    imagesRegion = numpy.transpose(imagesRegion, (1, 0, 2, 3, 4))
    # Swap(Band, Pixels): ((n*m) x 4 x (20x20) x 3) -> ((n*m) x (20x20) x 4 x 3)
    imagesRegion = numpy.transpose(imagesRegion, (0, 2, 3, 1, 4))
    return imagesRegion

def addTag(imagesBand, label):
    taggedImagesBand = []
    for i in range(len(imagesBand)):
        taggedImagesBand.append((imagesBand[i], label))
    return taggedImagesBand
    
def getImages():
    totalImages = []
    for i in range(2):
        for folder in os.listdir(imagesPath[i]):
            print('Into folder:', imagesPath[i] + folder)
            imagesRegion = []
            for filename in os.listdir(imagesPath[i] + folder):
                print('\tSegmentize file:', filename)
                segmentizeImage = segmentize(imagesPath[i] + folder + '/' + filename, SHAPE)
                imagesRegion.append(segmentizeImage)
            print('\tJoinBands')
            imagesBand = joinBandsRegion(imagesRegion)
            taggedImagesBand = addTag(imagesBand, i)
            totalImages.extend(taggedImagesBand)
        # DELETE
        break
        # DELETE
    print('Total de data input:', len(totalImages))
    print('Dimension data input:', totalImages[0][0].shape)

    return totalImages

def download_dataset():
	dataset = getImages()
	random.shuffle(dataset)
	return dataset

def split_data_x_y(data):
	data_input = []
	data_output = []
	for reg in data:
		data_input.append(reg[0])
		data_output.append(reg[1])
	return numpy.array(data_input), numpy.array(data_output)

def split_data_train_test(data, split_size):
	len_train = int(len(data)*split_size)
	return data[:len_train], data[len_train:]

class Normalizer():
	def transform(data):
		for id, reg in enumerate(data):
			normalize_image = (reg[0].astype(numpy.float32))/255.0
			data[id] = (normalize_image, reg[1])
		return data

class TimeSeriesDataset(Dataset):
	def __init__(self, x, y):
		self.x = numpy.transpose(x.reshape(x.shape[0], x.shape[1], x.shape[2], -1), (0, 3, 1, 2))
		self.y = y
		
	def __len__(self):
		return len(self.x)

	def __getitem__(self, idx):
		return (self.x[idx], self.y[idx])