import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image

def q_1(input_path):

    file = open(input_path+"label.txt", 'r')
    lines = file.readlines()

    gt_count = 0
    img_count = 0

    box_count = 0

    sum_area = 0
    sum_h = 0
    sum_w =0

    x_pos = []
    y_pos = []

    box_w = []
    box_h = []

    box_center_x =[]
    box_center_y=[]

    img_name = None
    img_w = None
    img_h = None

    img_ws = []
    img_hs = []

    exceed_w = 0
    exceed_h = 0
    for l in lines:
        if(l[0]=="#"):
            img_count+=1
            img_name = l.split(' ')
            image = Image.open(input_path+'images/'+img_name[1][:-1])
            img_w = image.width
            img_h = image.height

        else:
            lang_split = l.split(" ")
            x,y,w,h = lang_split[:4]
            x= int(x)
            y = int(y)
            w = int(w)
            h = int(h)
            box_count+=1
            x_pos.append((x+w/2)/img_w)
            y_pos.append((y+h/2)/img_h)

            box_center_x.append(x+w/2)
            box_center_y.append(y + h/2)

            img_ws.append(img_w)
            img_hs.append(img_h)



            if(x>=0 and y>=0 and w>=0 and h>=0):
                gt_count+=1

                sum_area += w * h
                sum_h+=h
                sum_w+=w

                if(x+w>img_w):
                    exceed_w+=1
                if(y+h>img_h):
                    exceed_h+=1

    print("Img count: ",img_count)
    print("GT count: ", gt_count)
    print("ave area: ", sum_area/gt_count)
    print("ave w: ", sum_w / gt_count)
    print("ave h: ", sum_h / gt_count)

    print("exceed h: ", exceed_h)
    print("exceed w: ", exceed_w)


    # plt.show()
    # plt.hist(np.array(y_pos), bins=10)
    print("img center: ", np.average(img_ws)/2,np.average(img_hs)/2)
    print("box center: ", np.average(box_center_x),np.average(box_center_y))

    plt.title("box x position ratio in image")
    plt.hist(x_pos, bins=10)
    plt.show()
    plt.title("box y position ratio in image")
    plt.hist(y_pos, bins=10)
    plt.show()

def check_iou(x1,y1,w1,h1,x2,y2,w2,h2):
    x_max = max(x1, x2)
    y_max = max(y1, y2)
    x2_min = min(x1 + w1, x2 + w2)
    y2_min = min(y1 + h1, y2 + h2)
    if x_max < x2_min and y_max < y2_min:
        return True
    else:
        return False

def q_2(input_path):

    file = open(input_path+"label.txt", 'r')
    lines = file.readlines()

    set_f = set()

    gt_count = 0

    overlap_count = 0

    xs = []
    ys = []
    hs = []
    ws = []

    for l in lines:
        # get to next image
        if(l[0]=="#"):
            size = len(xs)
            # print(size)
            for i in range(size):
                for j in range(i+1,size):
                    # x1,y1,w1,h1,x2,y2,w2,h2
                    if(check_iou(xs[i],ys[i],ws[i],hs[i],
                                 xs[j],ys[j],ws[j],hs[j])):
                        # overlap_count+=2
                        set_f.add(j)
                        set_f.add(i)
            overlap_count+=len(set_f)

            # next image
            set_f = set()
            xs = []
            ys = []
            hs = []
            ws = []


        else:
            lang_split = l.split(" ")
            x,y,w,h = lang_split[:4]
            x = int(x)
            y = int(y)
            w = int(w)
            h = int(h)


            if(x>=0 and y>=0 and w>=0 and h>=0):
                xs.append(x)
                ys.append(y)
                hs.append(h)
                ws.append(w)

                gt_count+=1

    print("Overlap: ",overlap_count/gt_count)

print("For training data:")

q_1('../widerface_homework/train/')

q_2('../widerface_homework/train/')

print("For val data: ")

q_1('../widerface_homework/val/')

q_2('../widerface_homework/val/')