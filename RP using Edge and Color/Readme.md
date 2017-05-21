#
In this directory, there are supposed to be two directories.
One is '0_Good', and the other is 'Feature Training'.

The directory '0_Good' contains good examples of license plate photos.
The directory 'Feature Training' has a following structure:
	Feature Training
		|
		|-- NEG: 	random regions from each picture
		|-- negative: 	all negative region proposals( with IOU threshold is 0.7)
		|-- POS: 	ground truth in each picture
		|-- positive:	all positive region proposals( with IOU threshold is 0.7)
#
Explanation for each python file:
ColorSegDemo.py: 	shows results of color segmentation on each picture in the '0_Good' directory
EdgeAndColorDemo.py:	shows one step-by-step result of combined edge feature and color segmentation method
RegionProp.py:		shows results of edge and color based region proposal method on each picture
svm.py:			a library from the book 'Machine Learning in Action' with simple alterations
svmTraining.py:		trains a linear svm model using samples in 'Feature Training' directory
#
Usage:
For ColorSegDemo.py, EdgeAndColorDemo.py, RegionProp.py, and svmTraining.py, no command line argument is needed.
They can be called with a simple command:

	python	****.py

where **** is the corresponding name of the file.
