import cv2
import numpy as np
from torchvision.ops import nms
import torch

def resolutionizer(image):
    h, w = image.shape[0], image.shape[1]
    maxres = max(h, w)
    scalefactor = 500 / maxres
    h = int(h *scalefactor)
    w = int(w * scalefactor)

    resized_image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)

    return resized_image


def mser_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mser = cv2.MSER_create(delta=5,
                           min_area=100,
                           max_area=20000,
                           max_variation=0.5,
                           min_diversity=0.1)

    regions, bboxes = mser.detectRegions(gray)

    return bboxes



def aspect_filter(bboxes):
    bbox2 = []
    for box in bboxes:
        x,y,w,h = box
        aspect = w / h
        if 0.2 <= aspect <= 1.1:
            bbox2.append(box)
    return bbox2




# https://www.researchgate.net/profile/P-K-Swain/publication/224890763_Text_Extraction_and_Recognition_from_Image_using_Neural_Network/links/0fcfd5077e8fb46a0f000000/Text-Extraction-and-Recognition-from-Image-using-Neural-Network.pdf
def nms_score(region):
    h, w = region.shape[0], region.shape[1]


    #1 Aspect Ratio
    aspect = w/h
    idealaspect = 2/3
    penalty = 2.5

    aspectscore = aspect-idealaspect
    aspectscore = aspectscore * penalty
    aspectscore = 1 - abs(aspectscore)
    aspectscore = max(0, aspectscore)

    #2 Edge https://www.reddit.com/r/opencv/comments/10z8ji6/discussion_more_reliable_edge_detection/
    edges = cv2.Canny(region, 50, 150)
    edgetotal = np.count_nonzero(edges)
    area = h*w
    edgedensity = edgetotal/area
    edgescore = min(1, edgedensity)

    return (aspectscore+edgescore)/2



def nms_filter(image, bboxes):

    regions = []
    fourcoordbox = []

    for box in bboxes:
        x,y,w,h = box
        fourcoordbox.append([x,y,x+w,y+h])
        region = image[y:y + h, x:x + w]
        regions.append(region)

    if len(fourcoordbox) < 1:
        return None

    fourcoordbox = torch.tensor(fourcoordbox, dtype=torch.float32)

    scores = []
    for region in regions:
        score = nms_score(region)
        scores.append(score)

    scores = torch.tensor(scores, dtype=torch.float32)

    # print(fourcoordbox.shape)

    indices = nms(fourcoordbox, scores, 0.3)



    indices = indices.tolist()

    bboxes2 = [bboxes[i] for i in indices]

    return bboxes2



def get_decreasing_pyramid(image):
    h, w = image.shape[0], image.shape[1]
    pyramid= [image.copy()]

    alist = [0.2, 0.35, 0.5, 0.7, 0.8]
    alist.reverse()

    for x in alist:
        h = int(h * x)
        w = int(w * x)
        resized = cv2.resize(image, (w, h))
        pyramid.append(resized)

    blist = [1.2, 1.35, 1.55, 1.8]
    h, w = image.shape[0], image.shape[1]
    for x in blist:
        h = int(h * x)
        w = int(w * x)
        resized = cv2.resize(image, (w, h))
        pyramid.append(resized)


    return pyramid


def mser_pipeline(test_img):
    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    bboxes = mser_detection(test_img)
    if bboxes is not None and len(bboxes) < 0:
        return None
    bboxes = aspect_filter(bboxes)
    if bboxes is not None and len(bboxes) < 0:
        return None
    bboxes = nms_filter(gray_img, bboxes)
    if bboxes is not None and len(bboxes) < 0:
        return None


    return bboxes
