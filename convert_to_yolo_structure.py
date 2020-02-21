import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import os.path

IMAGE_HEIGHT=512
IMAGE_WIDTH=512
IMAGE_TYPE=".jpg"
CLASSES_NAMES='classes.txt'
INPUT_CSV_FILE="data1.csv"
TEST_SIZE=0.2
IMAGE_FOLDER="data/custom/images/"
TRAIN_DATA_FILE="train.txt"
TEST_DATA_FILE="val.txt"

#calculate x & y width
def calRange(valMax,valMin,valImg):
    return (valMax-valMin)/valImg

#calculate x & y coordinate
def calCordinate(valMax,valMin,valImg):
    return (valMin+(valMax-valMin)/2)/valImg

def combine_multiple_lists(l1,l2,l3,l4,l5): 
    return list(map(lambda a,b,c,d,e:(a,b,c,d,e), l1,l2,l3,l4,l5)) 

def combine_lists(l1,l2): 
    return list(map(lambda x,y:(x,y), l1,l2)) 


csv_read=pd.read_csv(INPUT_CSV_FILE)
images=csv_read['image'].tolist()
labels_list = csv_read['label']
xmin=csv_read['xmin']
xmax=csv_read['xmax']
ymin=csv_read['ymin']
ymax=csv_read['ymax']

labels = labels_list.unique()
labeldict = dict(zip(labels,range(len(labels))))
SortedLabelDict = sorted(labeldict.items() ,  key=lambda x: x[1])

x_width=calRange(xmax,xmin,IMAGE_WIDTH).tolist()
y_height=calRange(ymax,ymin,IMAGE_HEIGHT).tolist()
x_cord=calCordinate(xmax,xmin,IMAGE_WIDTH).tolist()
y_cord=calCordinate(ymax,ymin,IMAGE_HEIGHT).tolist()

data_label=[]
for i in labels_list:
    for val in SortedLabelDict:
        if i==val[0]:
            data_label.append(val[1])


data_values=combine_multiple_lists(data_label,x_cord,y_cord,x_width,y_height)
data_list=combine_lists(images,data_values)

data_array=defaultdict(list)
for k,v in data_list:
    data_array[k].append(v)

for k,v in tqdm(data_array.items()):
    file_path = os.path.join('labels/', str(k).replace(IMAGE_TYPE,"txt"))
    fl=open(file_path, "w")
    fl.write((",".join(map(str,v))).replace(",","").replace("[","").replace("(","").replace("]","").replace(")","\n"))
    fl.close()

file = open(CLASSES_NAMES,"w") 
for elem in SortedLabelDict:
	file.write(elem[0]+'\n') 
file.close() 

images=np.array(images)
x_train ,x_test = train_test_split(images,test_size=TEST_SIZE) 

train_file = open(TRAIN_DATA_FILE,"w") 
for data in x_train:
	train_file.write(IMAGE_FOLDER+data+'\n') 
train_file.close() 

test_file = open(TEST_DATA_FILE,"w") 
for data in x_test:
	test_file.write(IMAGE_FOLDER+data+'\n') 
test_file.close() 

