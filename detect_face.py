import cv2
import time
import argparse

import numpy as np
from PIL import Image

from myutils.anchor_generator import generate_anchors
from myutils.anchor_decode import decode_bbox
from myutils.nms import single_class_non_max_suppression
from load_model.tensorflow_loader import load_tf_model, tf_inference

sess, graph = load_tf_model('model_saved/face_mask_detection.pb')

# anchor configuration
feature_map_sizes = [[33, 33], [17, 17], [9, 9], [5, 5], [3, 3]]
anchor_sizes = [[0.04, 0.056], [0.08, 0.11], [0.16, 0.22], [0.32, 0.45], [0.64, 0.72]]
anchor_ratios = [[1, 0.62, 0.42]] * 5

# generate anchors
anchors = generate_anchors(feature_map_sizes, anchor_sizes, anchor_ratios)
 
# for inference, the batch size is 1, the model ouput shape is [1, N, 4], 
# so we expand dim for anchors to [1, anchor_num, 4]
anchors_exp = np.expand_dims(anchors, axis=0)

id2class = {0: 'Mask', 1: 'NoMask'}

def inference(image, conf_thresh=0.5, iou_thresh=0.4, input_shape=(160,160), draw_result=True):
    '''
    Main function of detection inference
    :param image: 3D numpy array of image
    :param conf_thresh: the min threshold of classification probabity.
    :param iou_thresh: the IOU threshold of NMS
    :param target_shape: the model input size.
    :param draw_result: whether to daw bounding box to the image.
    :param show_result: whether to display the image.
    :return:
    '''
    output_info = []
    height, width, _ = image.shape
    image_resized = cv2.resize(image, input_shape)
    image_np = image_resized/255 # standardization to 0 ~1 
    image_exp = np.expand_dims(image_np, axis=0)
    y_bboxes_output, y_cls_output = tf_inference(sess, graph, image_exp)
    # print('y_bboxes_output : \n', y_bboxes_output[0])
    # print('y_cls_output :\n', y_cls_output[0])

    # removw the batch dimension, for the batch is always 1 for inference
    y_bboxes = decode_bbox(anchors_exp, y_bboxes_output)[0]
    # print(y_bboxes)
    y_cls = y_cls_output[0]
    # print(y_cls)
    # print(decode_bbox(anchors_exp, y_bboxes_output).shape)
    # to speed up, do single class NMS, not multiple classes NMS
    bbox_max_scores = np.max(y_cls, axis=1)
    # print('score', bbox_max_scores)
    bbox_max_score_classes = np.argmax(y_cls, axis=1)
    # print('classes', bbox_max_score_classes)

    # keep_idx is the alive bounding box after nms.
    keep_idxs = single_class_non_max_suppression(y_bboxes,
                                                 bbox_max_scores,
                                                 conf_thresh=conf_thresh,
                                                 iou_thresh=iou_thresh,
                                                 )
    # print(keep_idxs)
    
    for idx in keep_idxs:
        # print(idx)
        conf = float(bbox_max_scores[idx])
        class_id = bbox_max_score_classes[idx]
        bbox = y_bboxes[idx]
        # clip the coordinate, avoid the value exceed the image boundary.
        xmin = max(0, int(bbox[0] * width))
        ymin = max(0, int(bbox[1] * height))
        xmax = min(int(bbox[2] * width), width)
        ymax = min(int(bbox[3] * height), height)

        if draw_result:
            if class_id ==0:
                color = (0,255,0)
            else: color = (255,0,0)
            # image = cv2.circle(image, (int(bbox[2]*width), int(bbox[3]*height)), 2, (255, 255,0), -1)
            image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
            image = cv2.putText(image, "%s: %.2f" % (id2class[class_id], conf), (xmin + 2, ymin - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)
            output_info.append([class_id, conf, xmin, ymin, xmax, ymax])
            # print(xmax-xmin, ymax - ymin)
        
    return output_info

def run_on_video(video_path, output_video_name, conf_thresh):
    cap = cv2.VideoCapture(video_path)
    # height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    # width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    # fps = cap.get(cv2.CAP_PROP_FPS)
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # writer = cv2.VideoWriter(output_video_name, fourcc, int(fps), (int(width), int(height)))
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if not cap.isOpened():
        raise ValueError("Video open failed.")
        # return
    # status = True
    idx = 0
    while cap.isOpened():
        start_stamp = time.time()
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        read_frame_stamp = time.time()
        if ret:
            output =inference(frame,
                            conf_thresh,
                            iou_thresh=0.5,
                            input_shape=(260, 260),
                            draw_result=True)
            # print(output_infor)
            detected = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
            
            
            inference_stamp = time.time()
            # writer.write(img_raw)
            write_frame_stamp = time.time()
            idx += 1
            # print("%d of %d" % (idx, total_frames))
            # print("read_frame:%f, infer time:%f, write time:%f" % (read_frame_stamp - start_stamp,
            #                                                        inference_stamp - read_frame_stamp,
            #                                                        write_frame_stamp - inference_stamp))
            cv2.putText(detected, 'FPS: '+ str(int(1/(inference_stamp-start_stamp))), (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255))
            cv2.imshow('image', detected)
            if cv2.waitKey(1)&0xFF == 27:
                break
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Mask Detection")
    parser.add_argument('--img-mode', type=int, default=0, help='set 1 to run on image, 0 to run on video.')
    parser.add_argument('--img-path', type=str, help='path to your image.')
    parser.add_argument('--video-path', type=str, default='0', help='path to your video.')
    args = parser.parse_args()
    if args.img_mode:
        imgPath = args.img_path
        img = cv2.imread(imgPath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        inference(img, show_result=True, input_shape=(260,260))
    else:
        video_path = args.video_path
        if args.video_path == '0':
            video_path = 0
        run_on_video(video_path, '', conf_thresh=0.5)
