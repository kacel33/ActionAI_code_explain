# ActionAI 코드 설명 및 분석

## 위 저장소에서 사용한 알고리즘을 정리해 봤습니다.

+ iva.py의 알고리즘은 trt_pose를 이용하여 사람의 pose_dict을 얻습니다.
trt_pose모델은 pose_list를 return하는데 pose_list의 원소는 pose_dict입니다.

+ 그리고 사람 바운딩 박스를 만드는데 그 사람의 관절의 x최솟값, y최솟값, x최댓값, y최댓값을 구하여 사람 바운딩박스를 만듭니다.
그렇게 되면 보통 박스 위쪽은 '눈', 아랫쪽은 발목, 측면은 '어깨'나 '손목'이 됩니다.

+ 그렇게 얻은 각각의 사람마다 pose_dict을 각각의 tracker에 저장합니다.
tracker의 최대 길이는 deque를 이용하여 10으로 정했습니다.
그리고 전의 frame과 현재의 frame을 IOU를 이용해서 비교하여 matched, unmatched_trackers, unmatched_detections를 구분합니다.
+ matched는 전 frame과 현 frame의 IOU값이 0.3이상이 되는 tracker입니다.
matched는 pose를 update해줍니다.

+ unmatched_trackers는 전 frame에서는 존재하지 않다가 현 frame에서 나타난 것입니다. 
unmatched_trackers는 trackers에 append해줍니다.

+ unmatched_detections는 전 frame에서 존재하던 것이 현 frame에서는 없거나 IOU가 낮은 것입니다.
unmatched_detections는 trackers 리스트에서 빼줍니다.
+ tracker.q에는  2 * (np.array(pose_dict[ky]) - np.array(self.centroid)) / np.array((self.h, self.w))에서 보실 수 있듯이, 각 관절 좌표에서 바운딩 박스 중심을 뺀 뒤 너비 또는 높이로 나눠서 * 2 를 해서 저장합니다. 그래서 각 관절 좌표값은 -1에서 1사이가 되는데 정규화가 되는 효과가 있습니다.
+ RUNSECONDARY에 대해서 설명하겠습니다.
이 모델은 1개의 lstm layer와 2개의 dense layer로 구성된 간단한 모델입니다.
window = 3, pose_vec_dim = 36 으로 잡아서
3frame을 lstm모델의 인풋으로 넣어줍니다. 

# 코드를 실행할 때 참고하면 좋을 글

+ 모델 불러오기
> 제가 사용하면서 불편했던 점은 ActionAI 저장소에 링크되어 있는 model assets링크에서 모델을 다운받으면 실행이 안됩니다.
resnet18_baseline_att_224x224_A_epoch_249_trt.pth를 다운받아서 사용하면 안되고,
trt_pose 저장소에서 resnet18_baseline_att_224x224_A_epoch_249.pth 모델을 받아 자신의 환경에서 tensorrt로 모델을 최적화하여 resnet18_baseline_att_224x224_A_epoch_249_trt.pth을 만들어야 합니다.
tensorrt는 다른 버전에서 최적화 한 모델을 사용하지 못하는 듯 합니다.
+ 생각보다 fps가 높지는 않습니다.
> 저는 jetson nano에서 테스트했습니다.
RUNSECONDARY를 돌리지 않을 때 동영상에서 한 사람만 있을 때는 fps가 10 to 13이 나왔지만,
2명일 때는 fps가 7 to 9 3명이상일 때는 5이하로 떨어졌습니다.
참고로 trt_pose에서 resnet18모델을 이용했을 때 jetson nano에서는 fps가 22정도 나온다고 되어있었는데 저는 16이 최대였었습니다.
+ RUNSECONDARY 모델 참고
> RUNSECONDARY를 같이 돌릴 때는 fps가 엄청 떨어졌었습니다.
lstm모델에서 recurrent_dropout=0.2를 사용하여 그런지 cuda지원이 안됐었습니다.
예전에 저는 DACON https://dacon.io/competitions/official/235689/overview/description 대회에서 
PUBLIC 10등 PRAVITE 14등(상위4%)을 했었습니다.
센서 데이터를 이용하여 운동동작을 분류하는 대회였는데 lstm보다 1d-cnn을 사용했을 때 더 빠르고 오차도 더 적었던 것을 생각하여
pytorch를 이용한 1d-cnn모델을 만들어서 적용하였습니다. 모델을 경량화해서 만들면 frame당 0.01초밖에 소모되지 않아 전체 fps에 거의 영향을 주지 않습니다.
+ tracker.q에 저장되어 있는 관절 데이터
> 위에서 설명했듯이 ActionAI에서는 각 관절 좌표에서 바운딩 박스 중심을 뺀 뒤 너비 또는 높이로 나눠서 * 2 를 해서 저장합니다. 하지만 이런식으로 저장하게 되면 제자리걷기와 걷는 것을 구분하기 힘들고, 위치 이동 정보가 없습니다. 그래서 저는 1 frame에 총 72개 데이터를 저장했는데 36개는 ActionAI에서 제공하는 것과 동일하게 만들고 나머지 36은 각 관절 좌표를 전체 이미지의 가로 또는 세로로 나눠서 위치 정보가 생기게 하였습니다.
## 마지막으로 제가 ActionAI를 실행하기 위해서 환경설정 한 방법입니다.
+ https://ddo-code.tistory.com/21
+ https://ddo-code.tistory.com/23
+ https://ddo-code.tistory.com/25
+ https://ddo-code.tistory.com/26
# REFERENCE
+ https://github.com/smellslikeml/ActionAI  모든 코드는 ActionAI에 기반해서 작성하였습니다.