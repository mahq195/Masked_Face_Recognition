import cv2
import os
from detect_face import inference

def crop():
    print('Cropping images ....')
    dst_path = r'user_face'
    src_path = r'user_image'

    list_user = os.listdir(src_path)
    for user in list_user:
        user_path = os.path.join(src_path, user)
        user_images = os.listdir(user_path)
        for image in user_images[:2]:
            image = os.path.join(user_path, image)
            img = cv2.imread(image)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            outputs = inference(img, input_shape=(260, 260))
            output = outputs[0]
            
            usr_name = os.path.basename(os.path.dirname(image))
            print(usr_name)
            print(output)
            
            if output[0] == 0:
                USR_PATH = os.path.join(dst_path + '\\mask', usr_name)
            else:
                USR_PATH = os.path.join(dst_path + '\\no_mask', usr_name)

            if len(output) == 6:
                xmin, ymin, xmax, ymax = output[2], output[3], output[4], output[5]
                # mask = output[0] # class_id
                face = img[ymin:ymax+1, xmin:xmax+1]
                # cv2.imshow('face', face)
                face = cv2.resize(face, (160,160))
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

                if not os.path.exists(USR_PATH):
                        os.mkdir(USR_PATH)
                
                cv2.imwrite(os.path.join(USR_PATH, os.path.basename(image)), face)
                # print(  + ' done ' + str(id) + '\n')
    print('Cropping successfull!')

if __name__ == '__main__':
    # print('Cropping image ...')
    crop()