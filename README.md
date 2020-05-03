# Convert CSV to YOLO data format 

Testing Version - work in progress 

In YOLO image data store in a text file(img.txt) file for the corresponding image(img.jpg) in the same directory.
each text file include with  object-class x y width height for each object in line. 

Where :<br/>
object-class=integer number of object from 0 to (classes-1)<br/>
x,y,width,height=float values relative to width and height of image, it can be equal from (0.0 to 1.0]<br/>

CSV data fomat :

image,xmin,ymin,xmax,ymax,label
