import mmcv
from mmcv import imread
import mmengine
from mmengine.registry import init_default_scope
import numpy as np

"""
models: https://mmpose.readthedocs.io/en/latest/index.html
https://qiita.com/unyacat/items/1245bf595e53e79b5d4a
https://github.com/open-mmlab/mmdeploy/blob/main/docs/en/03-benchmark/supported_models.md

human2dkeypointにしても数が多すぎてどれを選ぶのかわからない 精度と速度のバランスがいいもの
"""

from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

local_runtime = False

try:
    from google.colab.patches import cv2_imshow  # for image visualization in colab
except:
    local_runtime = True


pose_config = 'configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py'
pose_checkpoint = 'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth'
det_config = 'demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py'
det_checkpoint = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
"""dir内にあるもの全てをそのまま読めるわけではないよう。要検証
pose_config = "configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-base_8xb64-210e_coco-256x192.py"
pose_checkpoint = "https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-base_8xb64-210e_coco-256x192-216eae50_20230314.pth"
det_config = 'demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py'
det_checkpoint = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
"""
#device = 'cuda:0'
device = 'cpu'
cfg_options = dict(model=dict(test_cfg=dict(output_heatmaps=True)))

# build detector
detector = init_detector(
    det_config,
    det_checkpoint,
    device=device
)

# build pose estimator
pose_estimator = init_pose_estimator(
    pose_config,
    pose_checkpoint,
    device=device,
    cfg_options=cfg_options
)

# init visualizer
pose_estimator.cfg.visualizer.radius = 3
pose_estimator.cfg.visualizer.line_width = 1
visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
# the dataset_meta is loaded from the checkpoint and
# then pass to the model in init_pose_estimator
visualizer.set_dataset_meta(pose_estimator.dataset_meta)

def visualize_img(img_path, detector, pose_estimator, visualizer,
                  show_interval, out_file):
    """Visualize predicted keypoints (and heatmaps) of one image."""

    # predict bbox
    scope = detector.cfg.get('default_scope', 'mmdet')
    if scope is not None:
        init_default_scope(scope)
    detect_result = inference_detector(detector, img_path)
    pred_instance = detect_result.pred_instances.cpu().numpy()
    bboxes = np.concatenate(
        (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
    bboxes = bboxes[np.logical_and(pred_instance.labels == 0,
                                   pred_instance.scores > 0.3)]
    bboxes = bboxes[nms(bboxes, 0.3)][:, :4]

    # predict keypoints
    pose_results = inference_topdown(pose_estimator, img_path, bboxes)
    data_samples = merge_data_samples(pose_results)

    # show the results
    img = mmcv.imread(img_path, channel_order='rgb')

    visualizer.add_datasample(
        'result',
        img,
        data_sample=data_samples,
        draw_gt=False,
        draw_heatmap=True,
        draw_bbox=True,
        show=False,
        wait_time=show_interval,
        out_file=out_file,
        kpt_thr=0.3)

#img = "/content/channels4_profile.jpg"
img = 'tests/data/coco/000000197388.jpg'

visualize_img(
    img,
    detector,
    pose_estimator,
    visualizer,
    show_interval=0,
    out_file=None)

vis_result = visualizer.get_image()

## 2D 座標の取得

# predict bbox
scope = detector.cfg.get('default_scope', 'mmdet')
if scope is not None:
  init_default_scope(scope)

detect_result = inference_detector(detector, img)
pred_instance = detect_result.pred_instances.cpu().numpy() #検出
bboxes = np.concatenate((pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
bboxes = bboxes[np.logical_and(pred_instance.labels == 0, pred_instance.scores > 0.3)]
bboxes = bboxes[nms(bboxes, 0.3)][:, :4]

# predict keypoints
pose_results = inference_topdown(pose_estimator, img, bboxes) #認識結果
data_samples = merge_data_samples(pose_results) #認識結果を1つのobjectにまとめたもの

import matplotlib.pyplot as plt
threshold = 0.5 #scoreの閾値

"""
pose_resultsはlistであり、1要素は1人物である
・1人が検出できた場合は長さ1であり、5人検出できた場合は長さ5
・身体座標は、pose_result["pred_instances"]["keypoints"]にはいっている
・形式はndarray
・pose_resultは変換しないとPoseDataSampleという形になっているので、分解できない
・pose_result.to_dict()とすると、辞書型に変換できるので、値を扱うことができる
・0: nose,1: r_eye, 2: l_eye, 3: r_ear, 4: l_ear, 5: r_shoulder, 6: l_shoulder 7: r_elbow, 8: l_elbow, 9: r_wrist, 10: l_wrist,11: r_hip, 12: l_hip, 13: r_knee, 14: l_knee, 15: r_ankle, 16: l_ankle

・arrayで座標が入っているので、抜けがあった時に順番ずれない？
-> 絶対にnullができない作り、最も確率が高い座標の値を入れるらしい。座標を取るときはthresholdでnullに置換する処理が必須っぽい

疑問点
・人ごとにIDは振られる？ -> 振られないように見える
"""
for pose_result in pose_results:
  res = pose_result.to_dict()
  kpt = res["pred_instances"]["keypoints"][0] #座標
  kps = res["pred_instances"]["keypoint_scores"][0] #score

  for i, score in enumerate(kps):
    if score < threshold:
      kpt[i] = np.array(None, None)

  #原点を左上から左下に移すための処理
  x_lim = res["img_shape"][1]
  y_lim = res["img_shape"][0]
  kpt_dst = []
  for k in kpt:
    kpt_dst.append([k[0], y_lim-k[1]])

  for i in range(len(kpt)):
    plt.plot(kpt_dst[i][0],kpt_dst[i][1],'.')

  #head
  plt.plot((kpt_dst[0][0],kpt_dst[1][0]), (kpt_dst[0][1],kpt_dst[1][1]), color="green")
  plt.plot((kpt_dst[0][0],kpt_dst[2][0]), (kpt_dst[0][1],kpt_dst[2][1]), color="green")
  plt.plot((kpt_dst[1][0],kpt_dst[2][0]), (kpt_dst[1][1],kpt_dst[2][1]), color="green")
  plt.plot((kpt_dst[1][0],kpt_dst[3][0]), (kpt_dst[1][1],kpt_dst[3][1]), color="green")
  plt.plot((kpt_dst[2][0],kpt_dst[4][0]), (kpt_dst[2][1],kpt_dst[4][1]), color="green")

  #trunk
  plt.plot((kpt_dst[3][0],kpt_dst[5][0]), (kpt_dst[3][1],kpt_dst[5][1]), color="green")
  plt.plot((kpt_dst[4][0],kpt_dst[6][0]), (kpt_dst[4][1],kpt_dst[6][1]), color="green")
  plt.plot((kpt_dst[5][0],kpt_dst[11][0]), (kpt_dst[5][1],kpt_dst[11][1]), color="green")
  plt.plot((kpt_dst[6][0],kpt_dst[12][0]), (kpt_dst[6][1],kpt_dst[12][1]), color="green")
  plt.plot((kpt_dst[11][0],kpt_dst[12][0]), (kpt_dst[11][1],kpt_dst[12][1]), color="green")

  #r arm
  plt.plot((kpt_dst[5][0],kpt_dst[7][0]), (kpt_dst[5][1],kpt_dst[7][1]), color="blue")
  plt.plot((kpt_dst[7][0],kpt_dst[9][0]), (kpt_dst[7][1],kpt_dst[9][1]), color="blue")

  #l arm
  plt.plot((kpt_dst[6][0],kpt_dst[8][0]), (kpt_dst[6][1],kpt_dst[8][1]), color="orange")
  plt.plot((kpt_dst[8][0],kpt_dst[10][0]), (kpt_dst[8][1],kpt_dst[10][1]), color="orange")

  # r_leg
  plt.plot((kpt_dst[11][0],kpt_dst[13][0]), (kpt_dst[11][1],kpt_dst[13][1]), color="blue")
  plt.plot((kpt_dst[13][0],kpt_dst[15][0]), (kpt_dst[13][1],kpt_dst[15][1]), color="blue")

  # l_leg
  plt.plot((kpt_dst[12][0],kpt_dst[14][0]), (kpt_dst[12][1],kpt_dst[14][1]), color="orange")
  plt.plot((kpt_dst[14][0],kpt_dst[16][0]), (kpt_dst[14][1],kpt_dst[16][1]), color="orange")

  plt.gca().set_aspect('equal', adjustable='box')
  plt.show()

## hand の検出

pose_config = 'configs/hand_2d_keypoint/topdown_heatmap/onehand10k/td-hm_res50_8xb32-210e_onehand10k-256x256.py'
pose_checkpoint = 'https://download.openmmlab.com/mmpose/top_down/resnet/res50_onehand10k_256x256-e67998f6_20200813.pth'
det_config = 'demo/mmdetection_cfg/cascade_rcnn_x101_64x4d_fpn_1class.py'
det_checkpoint = 'https://download.openmmlab.com/mmpose/mmdet_pretrained/cascade_rcnn_x101_64x4d_fpn_20e_onehand10k-dac19597_20201030.pth'

#device = 'cuda:0'
device = 'cpu'
cfg_options = dict(model=dict(test_cfg=dict(output_heatmaps=True)))

# build detector
detector = init_detector(
    det_config,
    det_checkpoint,
    device=device
)

# build pose estimator
pose_estimator = init_pose_estimator(
    pose_config,
    pose_checkpoint,
    device=device,
    cfg_options=cfg_options
)
pose_estimator.cfg.visualizer.radius = 3
pose_estimator.cfg.visualizer.line_width = 1

#img = "/content/4a989a7624d11d4e964b861c94128865_w.jpeg"
#img = "/content/37593007-両手を広げて.jpg"
img = "/content/セサル・アスピリクエタチェルシー、2017、フットボール選手、フォト壁紙_3840x2400[10wallpaper.com].jpg"

# predict bbox
scope = detector.cfg.get('default_scope', 'mmdet')
if scope is not None:
  init_default_scope(scope)
detect_result = inference_detector(detector, img)
pred_instance = detect_result.pred_instances.cpu().numpy() #検出
bboxes = np.concatenate((pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
bboxes = bboxes[np.logical_and(pred_instance.labels == 0, pred_instance.scores > 0.3)]
bboxes = bboxes[nms(bboxes, 0.3)][:, :4]

# predict keypoints
pose_results = inference_topdown(pose_estimator, img, bboxes) #認識結果
data_samples = merge_data_samples(pose_results) #認識結果を1つのobjectにまとめたもの

#0:wrist,1:thumb_3, 2:thumb_2, 3:thumb_3, 4:thumb_top, 5:index_3, 6:index_2, 7:index_1, 8:index_top, 9:middle_3, 10:mdille_2, 11:middle_1, 12:middle_top,
#13:ring_3, 14:ring_2, 15:ring_1, 16:ring_top, 17:little_3, 18:little_2, 19:little_1, 20:little_top,

threshold = 0.0
for pose_result in pose_results:
  res = pose_result.to_dict()
  kpt = res["pred_instances"]["keypoints"][0] #座標
  kps = res["pred_instances"]["keypoint_scores"][0] #score

  for i, score in enumerate(kps):
    if score < threshold:
      kpt[i] = np.array(None, None)

  #原点を左上から左下に移すための処理
  x_lim = res["img_shape"][1]
  y_lim = res["img_shape"][0]
  kpt_dst = []
  for k in kpt:
    kpt_dst.append([k[0], y_lim-k[1]])

  for i in range(len(kpt)):
    plt.plot(kpt_dst[i][0],kpt_dst[i][1],'.')

  # thumb
  plt.plot((kpt_dst[0][0],kpt_dst[1][0]), (kpt_dst[0][1],kpt_dst[1][1]), color="green")
  plt.plot((kpt_dst[1][0],kpt_dst[2][0]), (kpt_dst[1][1],kpt_dst[2][1]), color="green")
  plt.plot((kpt_dst[2][0],kpt_dst[3][0]), (kpt_dst[2][1],kpt_dst[3][1]), color="green")
  plt.plot((kpt_dst[3][0],kpt_dst[4][0]), (kpt_dst[3][1],kpt_dst[4][1]), color="green")

  # index
  plt.plot((kpt_dst[0][0],kpt_dst[5][0]), (kpt_dst[0][1],kpt_dst[5][1]), color="blue")
  plt.plot((kpt_dst[5][0],kpt_dst[6][0]), (kpt_dst[5][1],kpt_dst[6][1]), color="blue")
  plt.plot((kpt_dst[6][0],kpt_dst[7][0]), (kpt_dst[6][1],kpt_dst[7][1]), color="blue")
  plt.plot((kpt_dst[7][0],kpt_dst[8][0]), (kpt_dst[7][1],kpt_dst[8][1]), color="blue")

  # middle finger
  plt.plot((kpt_dst[0][0],kpt_dst[9][0]), (kpt_dst[0][1],kpt_dst[9][1]), color="orange")
  plt.plot((kpt_dst[9][0],kpt_dst[10][0]), (kpt_dst[9][1],kpt_dst[10][1]), color="orange")
  plt.plot((kpt_dst[10][0],kpt_dst[11][0]), (kpt_dst[10][1],kpt_dst[11][1]), color="orange")
  plt.plot((kpt_dst[11][0],kpt_dst[12][0]), (kpt_dst[11][1],kpt_dst[12][1]), color="orange")

  # ring finger
  plt.plot((kpt_dst[0][0],kpt_dst[13][0]), (kpt_dst[0][1],kpt_dst[13][1]), color="red")
  plt.plot((kpt_dst[13][0],kpt_dst[14][0]), (kpt_dst[13][1],kpt_dst[14][1]), color="red")
  plt.plot((kpt_dst[14][0],kpt_dst[15][0]), (kpt_dst[14][1],kpt_dst[15][1]), color="red")
  plt.plot((kpt_dst[15][0],kpt_dst[16][0]), (kpt_dst[15][1],kpt_dst[16][1]), color="red")

  # little finger
  plt.plot((kpt_dst[0][0],kpt_dst[17][0]), (kpt_dst[0][1],kpt_dst[17][1]), color="yellow")
  plt.plot((kpt_dst[17][0],kpt_dst[18][0]), (kpt_dst[17][1],kpt_dst[18][1]), color="yellow")
  plt.plot((kpt_dst[18][0],kpt_dst[19][0]), (kpt_dst[18][1],kpt_dst[19][1]), color="yellow")
  plt.plot((kpt_dst[19][0],kpt_dst[20][0]), (kpt_dst[19][1],kpt_dst[20][1]), color="yellow")

  plt.gca().set_aspect('equal', adjustable='box')
plt.show()

## 2dbody + 2dhand
"""
- bodyとhandを正常につなげるのがまぁ難しい（特に複数人写っている場合）
- 姿勢推定において、そんなに重要度が高くない(と思った)
- 画像選びが大変
- 技術的には可能なので、必要が生じた時に実装でいいのでは？
- 一旦塩漬けに
"""

## body

img = "/content/パウロ・ディヴァーラ、ユベントス、2017、フットボールの写真壁紙_3840x2160[10wallpaper.com].jpg"

pose_config = 'configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py'
pose_checkpoint = 'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth'
det_config = 'demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py'
det_checkpoint = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

#device = 'cuda:0'
device = 'cpu'
cfg_options = dict(model=dict(test_cfg=dict(output_heatmaps=True)))

# build detector
detector = init_detector(
    det_config,
    det_checkpoint,
    device=device
)
# build pose estimator
pose_estimator = init_pose_estimator(
    pose_config,
    pose_checkpoint,
    device=device,
    cfg_options=cfg_options
)

# init visualizer
pose_estimator.cfg.visualizer.radius = 3
pose_estimator.cfg.visualizer.line_width = 1
print(pred_instance)

# predict bbox
scope = detector.cfg.get('default_scope', 'mmdet')
if scope is not None:
  init_default_scope(scope)
detect_result = inference_detector(detector, img)
pred_instance = detect_result.pred_instances.cpu().numpy() #検出
bboxes = np.concatenate((pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
bboxes = bboxes[np.logical_and(pred_instance.labels == 0, pred_instance.scores > 0.3)]
bboxes = bboxes[nms(bboxes, 0.3)][:, :4]

# predict keypoints
pose_results = inference_topdown(pose_estimator, img, bboxes) #認識結果

threshold = 0.5
for pose_result in pose_results:
  res = pose_result.to_dict()
  kpt = res["pred_instances"]["keypoints"][0] #座標
  kps = res["pred_instances"]["keypoint_scores"][0] #score

  for i, score in enumerate(kps):
    if score < threshold:
      kpt[i] = np.array(None, None)

  #原点を左上から左下に移すための処理
  x_lim = res["img_shape"][1]
  y_lim = res["img_shape"][0]
  kpt_dst = []
  for k in kpt:
    kpt_dst.append([k[0], y_lim-k[1]])

  for i in range(len(kpt)):
    plt.plot(kpt_dst[i][0],kpt_dst[i][1],'.')

  #head
  plt.plot((kpt_dst[0][0],kpt_dst[1][0]), (kpt_dst[0][1],kpt_dst[1][1]), color="green")
  plt.plot((kpt_dst[0][0],kpt_dst[2][0]), (kpt_dst[0][1],kpt_dst[2][1]), color="green")
  plt.plot((kpt_dst[1][0],kpt_dst[2][0]), (kpt_dst[1][1],kpt_dst[2][1]), color="green")
  plt.plot((kpt_dst[1][0],kpt_dst[3][0]), (kpt_dst[1][1],kpt_dst[3][1]), color="green")
  plt.plot((kpt_dst[2][0],kpt_dst[4][0]), (kpt_dst[2][1],kpt_dst[4][1]), color="green")

  #trunk
  plt.plot((kpt_dst[3][0],kpt_dst[5][0]), (kpt_dst[3][1],kpt_dst[5][1]), color="green")
  plt.plot((kpt_dst[4][0],kpt_dst[6][0]), (kpt_dst[4][1],kpt_dst[6][1]), color="green")
  plt.plot((kpt_dst[5][0],kpt_dst[11][0]), (kpt_dst[5][1],kpt_dst[11][1]), color="green")
  plt.plot((kpt_dst[6][0],kpt_dst[12][0]), (kpt_dst[6][1],kpt_dst[12][1]), color="green")
  plt.plot((kpt_dst[11][0],kpt_dst[12][0]), (kpt_dst[11][1],kpt_dst[12][1]), color="green")

  #r arm
  plt.plot((kpt_dst[5][0],kpt_dst[7][0]), (kpt_dst[5][1],kpt_dst[7][1]), color="blue")
  plt.plot((kpt_dst[7][0],kpt_dst[9][0]), (kpt_dst[7][1],kpt_dst[9][1]), color="blue")

  #l arm
  plt.plot((kpt_dst[6][0],kpt_dst[8][0]), (kpt_dst[6][1],kpt_dst[8][1]), color="orange")
  plt.plot((kpt_dst[8][0],kpt_dst[10][0]), (kpt_dst[8][1],kpt_dst[10][1]), color="orange")

  # r_leg
  plt.plot((kpt_dst[11][0],kpt_dst[13][0]), (kpt_dst[11][1],kpt_dst[13][1]), color="blue")
  plt.plot((kpt_dst[13][0],kpt_dst[15][0]), (kpt_dst[13][1],kpt_dst[15][1]), color="blue")

  # l_leg
  plt.plot((kpt_dst[12][0],kpt_dst[14][0]), (kpt_dst[12][1],kpt_dst[14][1]), color="orange")
  plt.plot((kpt_dst[14][0],kpt_dst[16][0]), (kpt_dst[14][1],kpt_dst[16][1]), color="orange")

  plt.gca().set_aspect('equal', adjustable='box')
  plt.show()

