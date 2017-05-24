## Filename
- For pictures, their names shold be numbers of the same length, e.g. 
	`0001.bmp`<br> 
	`0023.jpg`<br>
	`0406.jpg`<br>
	`2565.jpg`<br>

- For the variable in the Region proposal file used for training process, it should be a cell array variable in MatLab, with each cell being an `uint8` array containing all proposals for corresponding picture in a format of `[x1 y1 x2 y2]`

- Each picture has two region proposal file respectively, used in testing process. One is named `****_boxes.mat`, where `****` refers to picture's filename. In this file, total number of region proposals is 2000. The other file is named `****_boxes1.mat`, with 10000 proposals inside. 