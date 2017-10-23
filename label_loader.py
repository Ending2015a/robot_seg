import cv2
import numpy as np
import math
import scipy.io as sio
import tensorflow as tf

dataset = tf.contrib.data.Dataset
'''
#Labels to colours are obtained from here:
https://github.com/alexgkendall/SegNet-Tutorial/blob/c922cc4a4fcc7ce279dd998fb2d4a8703f34ebd7/Scripts/test_segmentation_camvid.py

However, the road_marking class is collapsed into the road class in the dataset provided.

Classes:
------------
Sky = [128,128,128]
Building = [128,0,0]
Pole = [192,192,128]
Road_marking = [255,69,0]
Road = [128,64,128]
Pavement = [60,40,222]
Tree = [128,128,0]
SignSymbol = [192,128,128]
Fence = [64,64,128]
Car = [64,0,128]
Pedestrian = [64,64,0]
Bicyclist = [0,128,192]
Unlabelled = [0,0,0]
'''

#==========COMBINE LABEL==============
combine_list = []
tf_combine_list=None

def load_combine_list(path):
    global combine_list
    global tf_combine_list
    combine_list = np.array(sio.loadmat(path)['list'][0])
    tf_combine_list = tf.convert_to_tensor(combine_list)

def tf_combine_annotation(anno):
    if tf_combine_list == None:
        return anno
    anno = tf_combine_list[anno]
    #anno = tf.map_fn(lambda x: tf_combine_list[x], anno)
    return anno
    




#===========LABEL VISUALIZER==============

font_scale = 0.5
font_Face = cv2.FONT_HERSHEY_SIMPLEX

#rgb
label_to_colours =    [[128,128,128],
                     [128,0,0],
                     [192,192,128],
                     [128,64,128],
                     [60,40,222],
                     [128,128,0],
                     [192,128,128],
                     [64,64,128],
                     [64,0,128],
                     [64,64,0],
                     [0,128,192],
                     [0,0,0]]

label_to_texts = ['Sky',
                'Building',
                'Pole',
                'Road',
                'Pavement',
                'Tree',
                'SignSymbol',
                'Fence',
                'Car',
                'Pedestrian',
                'Bicyclist',
                'Unlabelled']

def get_label_max_length(label, texts):
    size = len(label)
    maxsz=0
    for k in texts:
        maxsz = len(k)

    return size, maxsz


def center_text(img, text, color, rect):
    text = text.strip()
    font = font_Face
    textsz = cv2.getTextSize(str(text), font, font_scale, 1)[0]

    textX = (rect.w-textsz[0]) / 2 + rect.x
    textY = (rect.h+textsz[1]) / 2 + rect.y

    cv2.putText(img, text, (textX, textY), font, font_scale, [0, 0, 0], 2)
    cv2.putText(img, text, (textX, textY), font, font_scale, color, 1)
    return img

def draw_label(label, texts):
    size, maxsz = get_label_max_length(label, texts)

    column = min(size, 15)
    row = int(math.ceil(size/15.0))

    wnd_w = (maxsz*10+50)
    wnd_h = 30
    width = row * wnd_w
    height = column * wnd_h

    img = np.zeros((height, width, 3), np.uint8)
    
    class rect:
        def __init__(self, w, h, colmax, rowmax):
            self.x=0
            self.y=0
            self.w=w
            self.h=h
            self.cmax = colmax
            self.rmax = rowmax
            self.c=0
            self.r=0
        def next(self):
            self.y += self.h
            self.c += 1
            if self.c == self.cmax:
               self.x += self.w
               self.r += 1
               self.c = 0
               self.y = 0 
            

    r = rect(wnd_w, wnd_h, column, row)

    for color, text in zip(label, texts):
        #print('({},{},{},{}) color: {}'.format(r.x, r.y, r.w, r.h, color))
        #print('{} {}'.format(color, type(color)))
        #print('   {}'.format(color[::-1]))
        color = np.array(color, dtype=int)
        cv2.rectangle(img, (r.x, r.y), (r.x+r.w, r.y+r.h), color[::-1] ,-1)
        img = center_text(img, text, [255, 255, 255], r)
        r.next()

    return img

def output_label(seg_img, path):
    color_idx = np.unique(seg_img).astype(int)
    color_list = []
    text_list = []
    for i in color_idx:
        color_list.append(label_to_colours[i])
        text_list.append(label_to_texts[i])

    img = draw_label(color_list, text_list)
    cv2.imwrite(path, img)    

def output_all_label(path):
    img = draw_label(label_to_colours, label_to_texts)
    cv2.imwrite(path, img)


def load_label(mat_file):
    import scipy.io as sio
    mat = sio.loadmat(mat_file)
    global label_to_colours
    global label_to_texts
    label_to_colours = mat['colors']
    label_to_texts = mat['names']

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='draw some labels')
    parser.add_argument('--mat', type=str, default='', help='a .mat file which contains \'colors\' and \'names\' columns')
    parser.add_argument('--output', type=str, default='./labels.png', help='output path of label image')
    
    
    args = parser.parse_args()

    assert args.mat != ''
    assert args.output != ''
    
    import os

    input_path = os.path.abspath(args.mat)
    output_path = os.path.abspath(args.output)

    load_label(input_path)
    output_all_label(output_path)
    print('Save label image to: {}'.format(output_path))


