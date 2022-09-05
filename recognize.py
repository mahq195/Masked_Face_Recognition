from detect_face import inference
import cv2
from myutils.recognize_func import recognize_mask, recognize_nomask

# cam = cv2.VideoCapture(r'video\angela_merkel_clip.webm')
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT,480)


score = 0
name = 0

while cam.isOpened():
    ret, frame = cam.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        outputs = inference(frame,
                        conf_thresh= 0.5,
                        iou_thresh= 0.5,
                        input_shape= (260,260),
                        draw_result=True)
        for output in outputs:
        
            # print(outputs)
            # output = {[class_id, conf, xmin, ymin, xmax, ymax]]
            # detected = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
            xmin, ymin, xmax, ymax = output[2], output[3], output[4], output[5]
            mask = output[0] # class_id
            face = frame[ymin:ymax+1, xmin:xmax+1]
            face = cv2.resize(face, (160, 160))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            if mask == 1:
                score, name = recognize_nomask(face, threshold=1.2)
                color = (255,0,0)
            elif mask == 0:
                score, name = recognize_mask(face, threshold=1.2)
                color = (0,255,0)

            cv2.putText(frame, "%s: %.2f" % (name, score), (xmin + 2, ymin - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color)
        cv2.imshow('image', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        if cv2.waitKey(1)&0xFF == 27:
            break
        
            



