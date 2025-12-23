import numpy as np

def check_iou(topx1,topy1,botx1,boty1,topx2,topy2,botx2,boty2,thres):
    x_max = max(topx1, topx2)
    y_max = max(topy1, topy2)
    x2_min = min(botx1, botx2)
    y2_min = min(boty1, boty2)
    if x_max >= x2_min or y_max >= y2_min:
        return False

    intersect = (x2_min- x_max) * (y2_min - y_max)
    area1 = (boty1-topy1) * (botx1-topx1)
    area2 = (boty2 - topy2) * (botx2 - topx2)

    iou = float(intersect)/float(area1+area2-intersect)
    # print(iou)

    if(iou<thres):
        return False

    return True

def nms(dets, thresh):
    # Todo
    indexs = range(len(dets))
    # sort with confidence
    condidences = np.array(dets)[:,4]
    sort_indexs = sorted(indexs,key=lambda i: condidences[i], reverse=True)
    res_indexs = []
    while(len(sort_indexs)!=0):
        keeped_index = sort_indexs[0]
        res_indexs.append(keeped_index)
        # print("1",keeped_index)

        # check each
        for ind in sort_indexs[1:]:
            # print("2", ind)
            topx1, topy1, botx1, boty1 = dets[keeped_index][:4]
            topx2, topy2, botx2, boty2 = dets[ind][:4]
            res = check_iou(topx1,topy1,botx1,boty1,topx2,topy2,botx2,boty2,thresh)
            # interset
            if(res):
                sort_indexs.remove(ind)

        # remove
        sort_indexs = sort_indexs[1:]
    return res_indexs

# test
A = [
    [0,0,1,1,0.9],[0.5,0,1.5,1,0.8],[0,1,1,2,0.7],[0.5,0.5,1.5,1.5,0.6]
]
print(nms(A, 0.01))