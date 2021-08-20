import os
import sys
import cv2
import csv
import time
import json
import torch
import string
import random
import PIL.Image
import numpy as np
from collections import deque
from operator import itemgetter
from sklearn.utils.linear_assignment_ import linear_assignment     ### ---->        ### 여기까지는 모듈을 import합니다. scikit-learn의 경우에는 0.23.1이하의 버전을 사용해야지
from pprint import pprint                                                           ### linear_assignment를 import할 수 있습니다.
                                                                                    ### linear_assignment 대신 scipy.optimize.linear_sum_assignment를 사용해도 됩니다.
import trt_pose.coco                                                                ### 하지만 scipy를 이용한다면 https://stackoverflow.com/questions/57369848/how-do-i-resolve-use-scipy-optimize-linear-sum-assignment-instead
import trt_pose.models                                                              ### 위 사이트에서 나온 것처럼 tracker_match 함수를 수정해야 합니다.
from torch2trt import TRTModule
import torchvision.transforms as transforms
from trt_pose.parse_objects import ParseObjects                                     ### trt_pose를 import합니다. 

model_w = 224
model_h = 224

ASSET_DIR = 'models/'
OPTIMIZED_MODEL = ASSET_DIR + 'resnet18_baseline_att_224x224_A_epoch_249_trt.pth'   ### tensorrt로 최적화한 모델 위치입니다. ActionAI에서 제공하는 
                                                                                    ### 모델은 사용하면 실행되지 않을 가능성이 높습니다. 
                                                                                    ### 이유는 모델을 최적화한 tensorrt버전과 그 모델을 사용하는 tensorrt버전이 같아야 하는데
                                                                                    ### 만약 버전이 다르다면 engine오류가 뜹니다. 
                                                                                    ### 따라서 ActionAI에서 제공하는 resnet18_baseline_att_224x224_A_epoch_249_trt.pth 파일을 받지 말고
                                                                                    ### 사용자가 trt_pose 저장소에서 resnet18_baseline_att_224x224_A_epoch_249_.pth을 받아서
                                                                                    ### 본인이 tensorrt로 최적화하여 trt모델을 얻어야 합니다.

body_labels = {0:'nose', 1: 'lEye', 2: 'rEye', 3:'lEar', 4:'rEar', 5:'lShoulder', 6:'rShoulder',
               7:'lElbow', 8:'rElbow', 9:'lWrist', 10:'rWrist', 11:'lHip', 12:'rHip', 13:'lKnee', 14:'rKnee',
              15:'lAnkle', 16:'rAnkle', 17:'neck'}
body_idx = dict([[v,k] for k,v in body_labels.items()])

with open(ASSET_DIR + 'human_pose.json', 'r') as f:
    human_pose = json.load(f)

model_trt = TRTModule()
model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))                              ### resnet18_baseline_att_224x224_A_epoch_249_trt.pth 모델을 불러옵니다.
                                                                                    ### jetson nano에서는 cuda에 모델을 불러올 때 상당히 많은 시간이 걸립니다.
                                                                                    ### 하지만 1~2분 이내로 걸리므로 너무 오래걸리면 다른 문제가 있을 가능성이 있습니다.
mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
device = torch.device('cuda')


def id_gen(size=6, chars=string.ascii_uppercase + string.digits):                   ### tracker의 id를 만듭니다. 랜덤으로 6개의 문자를 출력합니다.
    '''
    https://pythontips.com/2013/07/28/generating-a-random-string/
    input: id_gen(3, "6793YUIO")
    output: 'Y3U'
    '''
    return ''.join(random.choice(chars) for x in range(size))

def preprocess(image):                                                              ### 이미지를 전처리합니다.
    global device
    device = torch.device('cuda')
    image = cv2.resize(image, (model_h, model_w))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]

def inference(image):                                                               ### 이미지를 추론합니다.
    data = preprocess(image)                                                        
    cmap, paf = model_trt(data)
    cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
    counts, objects, peaks = parse_objects(cmap, paf) #, cmap_threshold=0.15, link_threshold=0.15)
    body_dict = draw_objects(image, counts, objects, peaks)
    return image, body_dict

def IOU(boxA, boxB):                                                                ### IOU를 구하는 함수입니다.
    # pyimagesearch: determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def get_bbox(kp_list):                                                              ### BoundingBOX를 얻습니다.
    bbox = []                                                                       ### detection된 18개의 점에서 x좌표의 최솟값과 최댓값을 구하고
    for aggs in [min, max]:                                                         ### y좌표의 최솟값과 최댓값을 구하여 BoundingBOX를 구합니다.
        for idx in range(2):
            bound = aggs(kp_list, key=itemgetter(idx))[idx]
            bbox.append(bound)
    return bbox

def tracker_match(trackers, detections, iou_thrd = 0.3):                            ### IOU를 사용하여 이전 frame의 tracker가 현재 frame의 tracker와 맞는 지 조사합니다.
    '''                                                                             
    From current list of trackers and new detections, output matched detections,
    unmatched trackers, unmatched detections.
    https://towardsdatascience.com/computer-vision-for-tracking-8220759eee85
    '''
                                                                                    ### matches는 이전 frame과 동일한 tracker입니다.
    IOU_mat= np.zeros((len(trackers),len(detections)),dtype=np.float32)             ### unmatched_detection은 이전 frame에서는 존재하지만 현재 frame에서는 없거나 IOU를 벗어난 tracker입니다.
    for t,trk in enumerate(trackers):                                               ### unmatche_tracker는 이전 frame에서는 없고, 현재 frame에서 나타난 frame입니다.
        for d,det in enumerate(detections):
            IOU_mat[t,d] = IOU(trk,det)

    # Produces matches
    # Solve the maximizing the sum of IOU assignment problem using the
    # Hungarian algorithm (also known as Munkres algorithm)

    matched_idx = linear_assignment(-IOU_mat)                                       ### linear_assignment함수는 scikit-learn 버전을 0.21~0.23으로 맞춰야 합니다.
                                                                                    ### scipy.optimize에서 linear_sum_assignment를 import하여 대신 사용할 수 있는데
    unmatched_trackers, unmatched_detections = [], []                               ### 만약에 바꿔서 사용한다면 from scipy.optimize import linear_sum_assignment
    for t,trk in enumerate(trackers):                                               ### matched_idx = linear_sum_assignment(-IOU_mat)
if(t not in matched_idx[:,0]):                                                      ### matched_idx = np.asarray(matched_idx)
            unmatched_trackers.append(t)                                            ### matched_idx = np.transpose(matched_idx) 로 바꿔서 사용하시면 됩니다.

    for d, det in enumerate(detections):
        if(d not in matched_idx[:,1]):
            unmatched_detections.append(d)

    matches = []

    # For creating trackers we consider any detection with an
    # overlap less than iou_thrd to signifiy the existence of
    # an untracked object

    for m in matched_idx:
        if(IOU_mat[m[0],m[1]] < iou_thrd):
            unmatched_trackers.append(m[0])
            unmatched_detections.append(m[1])
        else:
            matches.append(m.reshape(1,2))

    if(len(matches)==0):
        matches = np.empty((0,2),dtype=int)
    else:
        matches = np.concatenate(matches,axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

class PersonTracker(object):                                                                                                    ### while문 안에서 사용되는 tracker가 PErsonTracker클래스입니다. 
    def __init__(self):
        self.id = id_gen() #int(time.time() * 1000)
        self.q = deque(maxlen=10)                                                                                               ### tracker는 최대 10 frame의 관절 위치를 저장합니다.
        return

    def set_bbox(self, bbox):                                                                                                   ### 박스의 너비, 높이, 중심을 구합니다.
        self.bbox = bbox
        x1, y1, x2, y2 = bbox
        self.h = 1e-6 + x2 - x1
        self.w = 1e-6 + y2 - y1
        self.centroid = tuple(map(int, ( x1 + self.h / 2, y1 + self.w / 2)))
        return

    def update_pose(self, pose_dict):                                                                                           ### 36개의 zero배열을 만들고 detect된 관절 좌표에 (x,y)을 추가합니다.
        ft_vec = np.zeros(2 * len(body_labels))                                                                                 ### 저장되는 값은 (x-중심좌표)/h, (y-중심좌표)/w 입니다.
        for ky in pose_dict:                                                                                                    ### 저는 (x-중심좌표)/h, (y-중심좌표)/w 와 (x/image.shape[1]), (y/image.shape[0]) 
            idx = body_idx[ky]                                                                                                  ### 을 사용하여 72개의 값을 return하였습니다. 
            ft_vec[2 * idx: 2 * (idx + 1)] = 2 * (np.array(pose_dict[ky]) - np.array(self.centroid)) / np.array((self.h, self.w))### 여기 방법으로만 하면 모델은 제자리에서 움직이는 정보만 받기 때문에  
        self.q.append(ft_vec)                                                                                                   ### 저는 이미지 내의 위치 정보까지 추가하여 사용하였습니다.
        return

    def annotate(self, image):                                                                                                  ### 사람의 BoungBOX와, 현재 상태, 박스 중심을 opencv로 그립니다.
        x1, y1, x2, y2 = self.bbox
        image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 3)
        image = cv2.putText(image, self.activity, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        image = cv2.drawMarker(image, self.centroid, (255, 0, 0), 0, 30, 4)
        return image

class DrawObjects(object):

    def __init__(self, topology, body_labels):
        self.topology = topology
        self.body_labels = body_labels

    def __call__(self, image, object_counts, objects, normalized_peaks):
        topology = self.topology
        height = image.shape[0]
        width = image.shape[1]

        K = topology.shape[0]
        count = int(object_counts[0])
        K = topology.shape[0]
        body_list = []
        for i in range(count):
            body_dict = {}
            color = (112,107,222)
            obj = objects[0][i]
            C = obj.shape[0]
            for j in range(C):
                k = int(obj[j])
                if k >= 0:
                    peak = normalized_peaks[0][j][k]
                    x = round(float(peak[1]) * width)
                    y = round(float(peak[0]) * height)
                    cv2.circle(image, (x, y), 3, color, 2)
                    body_dict[self.body_labels[j]] = (x,y)
            body_list.append(body_dict)
            for k in range(K):
                c_a = topology[k][2]
                c_b = topology[k][3]
                if obj[c_a] >= 0 and obj[c_b] >= 0:
                    peak0 = normalized_peaks[0][c_a][obj[c_a]]
                    peak1 = normalized_peaks[0][c_b][obj[c_b]]
                    x0 = round(float(peak0[1]) * width)
                    y0 = round(float(peak0[0]) * height)
                    x1 = round(float(peak1[1]) * width)
                    y1 = round(float(peak1[0]) * height)
                    cv2.line(image, (x0, y0), (x1, y1), color, 2)
        return body_list

topology = trt_pose.coco.coco_category_to_topology(human_pose)
parse_objects = ParseObjects(topology)
draw_objects = DrawObjects(topology, body_labels)

source = sys.argv[1]
source = int(source) if source.isdigit() else source
cap = cv2.VideoCapture(source)

w = int(cap.get(3))
h = int(cap.get(4))

fourcc_cap = cv2.VideoWriter_fourcc(*'MJPG')                                        ### Opencv의 video 저장 함수입니다.
cap.set(cv2.CAP_PROP_FOURCC, fourcc_cap)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

DEBUG = True                                            
WRITE2CSV = False
WRITE2VIDEO = True
RUNSECONDARY = True

if WRITE2CSV:                                   
    activity = os.path.basename(source)
    dataFile = open('data/{}.csv'.format(activity),'w')
    newFileWriter = csv.writer(dataFile)

if WRITE2VIDEO:
    # Define the codec and create VideoWriter object
    name = 'out.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(name, fourcc, 30.0, (w, h))

if RUNSECONDARY:                                                                                                    
    import tensorflow as tf
    secondary_model = tf.keras.models.load_model('models/lstm_spin_squat.h5')
    window = 3
    pose_vec_dim = 36
    motion_dict = {0: 'spin', 1: 'squat'}

trackers = []
while True:

    ret, frame = cap.read()
    bboxes = []
    if ret:

        image, pose_list = inference(frame)                                                                         ### 이미지를 추론하여 body_dict 리스트를 얻습니다.
        for body in pose_list:                                                                                      ### body_dict리스트에서 각각의 body에서 BoundingBOX를 구합니다.
            bbox = get_bbox(list(body.values()))
            bboxes.append((bbox, body))

        track_boxes = [tracker.bbox for tracker in trackers]                                                        ### 
        matched, unmatched_trackers, unmatched_detections = tracker_match(track_boxes, [b[0] for b in bboxes])      ### 

        for idx, jdx in matched:                                                                                    ### 이전 frame과 매치된 tracker는 기존의 tracker에 append합니다.
            trackers[idx].set_bbox(bboxes[jdx][0])
            trackers[idx].update_pose(bboxes[jdx][1])

        for idx in unmatched_detections:                                                                            ### 기존의 tracker가 현재 frame에서 발견되지 않으면 삭제합니다.
            try:
                trackers.pop(idx)
            except:
                pass

        for idx in unmatched_trackers:                                                                              ### 이전 frame에서 없었으면 새로운 tracker에 추가합니다.
            person = PersonTracker()
            person.set_bbox(bboxes[idx][0])
            person.update_pose(bboxes[idx][1])
            trackers.append(person)

        if RUNSECONDARY:                                                                                            ### tensorflow2.0을 사용하여 행동을 추론하는 모델입니다.
            for tracker in trackers:                                                                                ### LSTM을 이용하였는데 LSTM보다는 1-d cnn을 이용하는 것이 좀 더 빨랐었습니다.
                print(len(tracker.q))                                                                               ### tensorflow를 사용하면 memory를 더 잡아먹는 느낌이었습니다.
                if len(tracker.q) >= 3:                                                                             ### 그래서 저는 구현할 때 pytorch로 변경하여 사용하였습니다.
                    sample = np.array(list(tracker.q)[:3])
                    sample = sample.reshape(1, pose_vec_dim, window)
                    pred_activity = motion_dict[np.argmax(secondary_model.predict(sample)[0])]
                    tracker.activity = pred_activity
                    image = tracker.annotate(image)
                    print(pred_activity)

        if DEBUG:
            pprint([(tracker.id, np.vstack(tracker.q)) for tracker in trackers])

        if WRITE2CSV:                                                                                               
            for tracker in trackers:
                print(len(tracker.q))
                if len(tracker.q) >= 3:
                    newFileWriter.writerow([activity] + list(np.hstack(list(tracker.q)[:3])))

        if WRITE2VIDEO:
            out.write(image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()

try:
    dataFile.close()
except:
    pass

try:
    out.release()
except:
    pass