#!/usr/bin/env python
'''
Bag of words (Features)

1. Extract features from samples of each class in dataset into ONE array.
2. Cluster into N number of "words".  Sugggest Kmeans clustering for this. This will construct the "dictionary" of all the "words" or similar features. We'll call this the "codebook". For N I suggest using square root of total number of "features": codebook = int(sqrt(nfeatures))
3. for each image in dataset: Extract features. Using these extracted features, create a histogram indicating the number of times each "word" appears in that image. This histogram will contain the same number of bins as "words" in the codebook. For instance if one of the words(bins in the histogram) is "34.997" and one of the feature data points is "35", the histogram bin labled "34.997" would be incremented by 1.   
'''

import cPickle as pickle
from skimage.feature import daisy
import argparse
import csv
from glob import glob
import os
import numpy as np
from scipy.cluster import vq
import cv2 
import random


#constants
PRE_ALLOCATION_BUFFER = 1000
K_THRESH = 1 # early stopping threshold for kmeans originally at 1e-5, increased for speedup
EXTENSIONS = [".jpg", ".bmp", ".png", ".pgm", ".tif", ".tiff"]
DATASETPATH = 'images'
CODEBOOK_FILE = 'codebook.file'
MODEL_FILE_LR = 'LogisticRegression.model'
MODEL_FILE_SVM = 'SVM.model'
MODEL_FILE_KNN = 'KNN.model'
HISTOGRAMS_FILE = 'histogram.dat'
OCTAVES = 128
NUM_FEATURES = 200

def parse_arguments():
    parser = argparse.ArgumentParser(description='tests multiple feature sets and learning machines to find optimal pairing')
    parser.add_argument('-d', help='path to the dataset', required=False, default=DATASETPATH)
    args = parser.parse_args()
    return args


def get_categories(datasetpath):
    print datasetpath
    cat_paths = [files
                 for files in glob(datasetpath + "/*")
                  if os.path.isdir(files)]
    cat_paths.sort()
    cats = [os.path.basename(cat_path) for cat_path in cat_paths]
    return cats


def get_imgfiles(path):
    all_files = []
    all_files.extend([os.path.join(path, os.path.basename(fname))
                    for fname in glob(path + "/*")
                    if os.path.splitext(fname)[-1].lower() in EXTENSIONS])
    return all_files


def save_data(features, classID):
	data_filename = HISTOGRAMS_FILE
	print 'writing image features to file: ', data_filename
	#write class data to file
	f_handle = open(data_filename, 'a')
	f_handle.write(str(classID))
	f_handle.write(', ')
	f_handle.close()

	f_handle = open(data_filename, 'a')
	for i in range(len(features)):
		f_handle.write(str(features[i]))
		f_handle.write(" ")
	f_handle.write('\n')
	f_handle.close()

def load_data(filename):
	data = []
	classID = []
	features = []
	features_temp_array = []
	print 'reading features and classID: ', filename
	f_handle = open(filename, 'r')
	reader = csv.reader(f_handle)
	#read data from file into arrays
	for row in reader:
		data.append(row)

	for row in range(0, len(data)):
		#print features[row][1]
		classID.append(int(data[row][0]))
		features_temp_array.append(data[row][1].split(" "))

	#removes ending element which is a space
	for x in range(len(features_temp_array)):
		features_temp_array[x].pop()
		features_temp_array[x].pop(0)

	#convert all strings in array to numbers
	temp_array = []
	for x in range(len(features_temp_array)):
		temp_array = [ float(s) for s in features_temp_array[x] ]
		features.append(temp_array)

	#make numpy arrays
	features = np.asarray(features)
	#print classID, features 
	return classID, features

def computeHistograms(codebook, descriptors):
	code, dist = vq.vq(descriptors, codebook)
	bins=range(codebook.shape[0] + 1)
	#print "bins:", bins
	histogram_of_words, bin_edges = np.histogram(code, bins, normed=True)
	#print histogram_of_words
	return histogram_of_words

def writeHistogramsToFile(nwords, labels, fnames, all_word_histgrams, features_fname):
    data_rows = np.zeros(nwords + 1) # +1 for the category label
    for fname in fnames:
        histogram = all_word_histgrams[fname]
        if (histogram.shape[0] != nwords): # scipy deletes empty clusters
            nwords = histogram.shape[0]
            data_rows = np.zeros(nwords + 1)
            print 'nclusters have been reduced to ' + str(nwords)
        data_row = np.hstack((labels[fname], histogram))
        data_rows = np.vstack((data_rows, data_row))
    data_rows = data_rows[1:]
    fmt = '%i '
    for i in range(nwords):
        fmt = fmt + '%f '
    np.savetxt(features_fname, data_rows, fmt)

def dict2numpy(dict):
    nkeys = len(dict)
    array = np.zeros((nkeys * PRE_ALLOCATION_BUFFER, OCTAVES))
    pivot = 0
    for key in dict.keys():
        value = dict[key]
        nelements = value.shape[0]
        while pivot + nelements > array.shape[0]:
            padding = np.zeros_like(array)
            array = np.vstack((array, padding))
        array[pivot:pivot + nelements] = value
        pivot += nelements
    array = np.resize(array, (pivot, OCTAVES))
    return array



def extractSURF(img, features_to_return, error_margin):
	gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	threshold = 500
	kp = 0
	iterations = 0
	completed = False
	top_error_margin = features_to_return + int(features_to_return * error_margin)
	bottom_error_margin = features_to_return - int(features_to_return * error_margin)
	while completed == False:
		if threshold <= 2 : completed = True
		features = cv2.SURF(threshold)
		kp, descriptors = features.detectAndCompute(gray, None)
		iterations += 1
		print "number of kp, descriptors:", len(kp), len(descriptors)	
		print "Threshold:", threshold
		print "Target # features range:", bottom_error_margin, " - ", top_error_margin 
		print "Iterations:", iterations
		if (len(kp) < bottom_error_margin):
			threshold = int(threshold * 0.91) - random.randint(0,1)
		elif (len(kp) > top_error_margin ):	
			threshold = int(threshold * 1.11) + random.randint(0,1)
		else:
			completed = True
		#if delta < 1: delta = 1
		if threshold < 5 : threshold = threshold  - 1
				
	return gray, kp, descriptors



def processSURF(input_files):
	global OCTAVES; OCTAVES = 128
	print "extracting SUFR features"
	cv2.namedWindow('SURF Features', cv2.CV_WINDOW_AUTOSIZE)
	all_features_dict = {}
	for i, fname in enumerate(input_files):
		print "calculating features for", fname
		img = cv2.imread(fname)
		gray, kp, descriptors = extractSURF(img, 200, 0.01)
		img2 = cv2.drawKeypoints(gray,kp)
		cv2.imshow('SURF Features', img2)	
		cv2.waitKey(30)	
		all_features_dict[fname] = descriptors
		#print all_features_dict
		#h = raw_input('')
	#cv2.destroyWindow('Features') 
	return all_features_dict



def testSURF(datasetpath, cats):
#--------------------------------------------
    all_files = []
    all_files_labels = {}
    all_features = {}
    cat_label = {}
    for cat, label in zip(cats, range(ncats)):
        cat_path = os.path.join(datasetpath, cat)
        cat_files = get_imgfiles(cat_path)
        cat_features = processSURF(cat_files)
        all_files = all_files + cat_files
        all_features.update(cat_features)
        cat_label[cat] = label
        for i in cat_files:
            all_files_labels[i] = label
#--------------------------------------------
    print "---------------------"
    print "## computing the visual words via k-means"
    all_features_array = dict2numpy(all_features)
    nfeatures = all_features_array.shape[0]
    nclusters = int(np.sqrt(nfeatures))
    print "## Number of words in codebook:", nclusters
    #all_features_array = whiten(all_features_array)
    codebook, distortion = vq.kmeans(all_features_array,
                                             nclusters,
                                             thresh=K_THRESH)

    with open(CODEBOOK_FILE, 'wb') as f:
        pickle.dump(codebook, f, protocol=pickle.HIGHEST_PROTOCOL)
#--------------------------------------------

	print "---------------------"
	print "## compute the visual words histograms for each image"
	all_word_histgrams = {}
	for imagefname in all_features:
		word_histgram = computeHistograms(codebook, all_features[imagefname])
		all_word_histgrams[imagefname] = word_histgram

	print "---------------------"
	print "## write the histograms to file"
	#for i in all_word_histgrams:
	#	print all_word_histgrams[i]

	writeHistogramsToFile(nclusters,
		                  all_files_labels,
		                  all_files,
		                  all_word_histgrams,
		                  HISTOGRAMS_FILE)
#--------------------------------------------

	print 'loading data file'
	data_file = np.loadtxt(HISTOGRAMS_FILE)
	classID = data_file[:,0].astype(int)
	Features = np.delete(data_file, 0,1)
	print classID[0], Features[0]

#--------------------------------------------
	print "---------------------"
	print "## train learning machines"

	from sklearn.linear_model import LogisticRegression
	clf2 = LogisticRegression().fit(Features, classID)
	pickle.dump( clf2 , open( MODEL_FILE_LR, "wb" ) )

	from sklearn import svm
	model = svm.SVC()
	model.fit(Features, classID)
	pickle.dump( model, open( MODEL_FILE_SVM, "wb" ) )

	from sklearn.neighbors import KNeighborsClassifier
	neigh = KNeighborsClassifier(n_neighbors=2)
	neigh.fit(Features, classID)
	pickle.dump( neigh, open(MODEL_FILE_KNN, "wb" ) )



def runtests_SURF():

	#--------------------------------------------
	#	TEST
	#--------------------------------------------
	#model_file = args.m
	#codebook_file = args.c
	all_files = []
	all_files_labels = {}
	all_features = {}
	print "going to test these files:"
	cats = get_categories("test_images/")
	datasetpath = "test_images/"
	print cats
	for cat in cats:
			cat_path = os.path.join(datasetpath, cat)
			cat_files = get_imgfiles(cat_path)
			for fnames in cat_files:
				print "calculating features for", fnames
				img = cv2.imread(fnames)
				gray, kp, features = extractSURF(img, 200, 0.01)

				for i in fnames:
					all_files_labels[i] = 0 # label is unknown

				print "---------------------"
				print "## loading codebook from " + CODEBOOK_FILE
				with open(CODEBOOK_FILE, 'rb') as f:
					codebook = pickle.load(f)

				print "---------------------"
				print "## computing visual word histograms"
				#print features
				word_histgram = computeHistograms(codebook, features)

				print "---------------------"
				print "## pass histogram to prediction models"
				nclusters = codebook.shape[0]
				#writeHistogramsToFile(nclusters,
				#					  all_files_labels,
				#					  fnames,
				#					  all_word_histgrams,
				#					  HISTOGRAMS_FILE)

				#print 'loading data file'
				#data_file = loadtxt(HISTOGRAMS_FILE)
				#print data_file, type(data_file)
				classID = None
				#classID = data_file[0].astype(int)
				#Features = np.delete(data_file, 0,1)
				#features = data_file[1:]
				#print features.shape[0]
				#features = features.reshape(1, (features.shape[0]))
				#print classID, word_histgram
				#sys.exit()

				print "---------------------"
				print "predictions for: ", fnames

				model_svm = pickle.load( open( MODEL_FILE_SVM, "rb" ) )
				classID_svm = model_svm.predict(word_histgram)
				print "SVM predicted classID:", classID_svm
				print "SVM decision: ", model_svm.decision_function(word_histgram)

				from sklearn.neighbors import KNeighborsClassifier
				KNN_clf = KNeighborsClassifier(n_neighbors=2)
				KNN_clf = pickle.load( open( MODEL_FILE_KNN, "rb" ) )
				KNN_classID = KNN_clf.predict(word_histgram)
				print "KNN predicted classID:", KNN_classID 
				print "KNN predicted prob:", KNN_clf.predict_proba(word_histgram)

				#from sklearn.svm import LinearSVC
				#LinearSVC_clf = LinearSVC()
				#LinearSVC_clf = pickle.load( open(MODEL_FILE, "rb" ) )
				#LinearSVC_class = LinearSVC_clf.predict(features)
				#print "LinearSVC_clf predicted classID:", LinearSVC_class
				#print "LinearSVC predicted prob:", LinearSVC_clf.predict_proba(features)

				from sklearn.linear_model import LogisticRegression
				clf2 = pickle.load(open(MODEL_FILE_LR, "rb" ) )
				print "LR2 predicted classID:",clf2.predict(word_histgram)
				print "LR2 predicted prob:", clf2.predict_proba(word_histgram)

				h = raw_input('')

	return 

if __name__ == '__main__':
	print "---------------------"
	print "## loading the images and extracting features"
	args = parse_arguments()
	datasetpath = args.d
	cats = get_categories(datasetpath)
	ncats = len(cats)
	print "searching for folders at " + datasetpath + '/'
	if ncats < 1:
		raise ValueError('Only ' + str(ncats) + ' categories found. Wrong path?')
	print "found following folders / categories:"
	print cats
	print "---------------------"

	testSURF(datasetpath, cats)
	runtests_SURF()




