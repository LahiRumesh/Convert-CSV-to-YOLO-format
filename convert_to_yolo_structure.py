import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import os
import argparse


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


parser=argparse.ArgumentParser()
parser.add_argument("--INPUT_CSV_FILE", type=str, default="input.csv", help="Input CSV file name")
parser.add_argument("--IMAGE_HEIGHT", type=int, default=512, help="Image height")
parser.add_argument("--IMAGE_WIDTH", type=int, default=512, help="Image width")
parser.add_argument("--IMAGE_TYPE", type=str, default=".jpg", help="Image type(.jpg | .png)")
parser.add_argument("--CLASSES_NAMES", type=str, default="classes.txt", help="classes file name")
parser.add_argument("--IMAGE_FOLDER", type=str, default="data/custom/images/", help="path to Image Folder")
parser.add_argument("--TRAIN_DATA_FILE", type=str, default="train.txt", help="Training Images set")
parser.add_argument("--TEST_DATA_FILE", type=str, default="valid.txt", help="Testing Images set")
parser.add_argument("--LABEL_FOLDER", type=str, default="labels", help="Labels folder")
parser.add_argument("--TEST_SIZE", type=float, default=0.1, help="Testing size from the data set")

args = parser.parse_args()


csv_read=pd.read_csv(args.INPUT_CSV_FILE)
images=csv_read['image'].tolist()
labels_list = csv_read['label']
xmin=csv_read['xmin']
xmax=csv_read['xmax']
ymin=csv_read['ymin']
ymax=csv_read['ymax']

labels = labels_list.unique()
labeldict = dict(zip(labels,range(len(labels))))
SortedLabelDict = sorted(labeldict.items() ,  key=lambda x: x[1])

x_width=calRange(xmax,xmin,args.IMAGE_WIDTH).tolist()
y_height=calRange(ymax,ymin,args.IMAGE_HEIGHT).tolist()
x_cord=calCordinate(xmax,xmin,args.IMAGE_WIDTH).tolist()
y_cord=calCordinate(ymax,ymin,args.IMAGE_HEIGHT).tolist()

data_label=[]
for i in labels_list:
    for val in SortedLabelDict:
        if i==val[0]:
            data_label.append(val[1])


data_values=combine_multiple_lists(data_label,x_cord,y_cord,x_width,y_height)
data_list=combine_lists(images,data_values)

os.mkdir(args.LABEL_FOLDER)

data_array=defaultdict(list)
for k,v in data_list:
    data_array[k].append(v)

for k,v in tqdm(data_array.items()):
    file_path = os.path.join(args.LABEL_FOLDER, str(k).replace(args.IMAGE_TYPE,".txt"))
    fl=open(file_path, "w")
    fl.write((",".join(map(str,v))).replace(",","").replace("[","").replace("(","").replace("]","").replace(")","\n"))
    fl.close()

classes_file = open(args.CLASSES_NAMES,"w") 
for elem in SortedLabelDict:
	classes_file.write(elem[0]+'\n') 
classes_file.close() 

images=np.array(images)
x_train ,x_test = train_test_split(images,test_size=args.TEST_SIZE) 

train_file = open(args.TRAIN_DATA_FILE,"w") 
for data in x_train:
	train_file.write(args.IMAGE_FOLDER+data+'\n') 
train_file.close() 

test_file = open(args.TEST_DATA_FILE,"w") 
for data in x_test:
	test_file.write(args.IMAGE_FOLDER+data+'\n') 
test_file.close() 

