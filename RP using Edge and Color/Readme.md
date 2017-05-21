## Directory
In this directory, there are supposed to be two directories.<br>
One is '0_Good', and the other is 'Feature Training'.<br>
<br>
The directory '0_Good' contains good examples of license plate photos.<br>
The directory 'Feature Training' has a following structure:<br>

	Feature Training 
		|
		|-- NEG:	random regions from each picture
		|-- negative:	all negative region proposals( with IOU threshold is 0.7)
		|-- POS:	ground truth in each picture
		|-- positive:	all positive region proposals( with IOU threshold is 0.7)

## Explanation for each python file
ColorSegDemo.py: 	shows results of color segmentation on each picture in the '0_Good' directory<br>
EdgeAndColorDemo.py:	shows one step-by-step result of combined edge feature and color segmentation method<br>
RegionProp.py:		shows results of edge and color based region proposal method on each picture<br>
svm.py:			a library from the book 'Machine Learning in Action' with simple alterations<br>
svmTraining.py:		trains a linear svm model using samples in 'Feature Training' directory<br>
## Usage
For ColorSegDemo.py, EdgeAndColorDemo.py, RegionProp.py, and svmTraining.py, no command line argument is needed.<br>
They can be called with a simple command:<br>

	python	****.py

where **** is the corresponding name of the file.
