import tensorflow as tf
import numpy as np
import os
import sys

# lib path to load data etc.
libpath = "../common/"
sys.path.append(os.path.abspath(libpath))
# Functions and classes for loading and using the Inception model.
import inception

def classify(image_path):
	# Display the image.
	# display(Image(image_path))

	# Use the Inception model to classify the image.
	pred = model.classify(image_path=image_path)

	# Print the scores and names for the top-10 predictions.
	model.print_scores(pred=pred, k=10, only_first_name=True)

if __name__ == '__main__':
	if len(sys.argv) != 2:
		sys.exit(1)
	inception.maybe_download()
	model = inception.Inception()
	fpath = sys.argv[1]
	if os.path.isfile(fpath):
		classify(fpath)
