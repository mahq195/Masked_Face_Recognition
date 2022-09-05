# import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
import glob
import cv2

import torch
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1
from crop_face import crop

device =  torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
trans = transforms.Compose([transforms.ToPILImage(),
                            transforms.Resize((128,128)),
                            transforms.ToTensor()
                           ])


nomask_model = InceptionResnetV1(pretrained='vggface2').eval() # Model of facenet for normal face recognition
mask_model = torch.load('model_saved\InceptionResNetV1_ArcFace.pt', map_location='cpu') # our model for masked face recognition

# from detect_face_mask import inference
# from recognize import recognize
# from utils import *

db_path = r'database' 


def create_mask_ebd():
    embeddings = []
    names = []
    path = r'user_face\mask'

    ls_user = os.listdir(path)   
    for i, user in enumerate(ls_user):
        for img in glob.glob(os.path.join(path, user) + '\\*'):
            img = cv2.imread(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = trans(img).float()
            img = norm(img).unsqueeze(0)
            with torch.no_grad():
                embed = mask_model(img)['embeddings']
            # print(embed)
            embeddings.append(embed)
            names.append(user)

    torch.save(torch.stack(embeddings), db_path+"/2mask_ebd.pth")
    np.save(db_path+"/2mask_usernames", np.array(names))
    print('Done mask!')

def create_nomask_ebd():
    embeddings = []
    names = []
    path = r'user_face\no_mask'

    ls_user = os.listdir(path)   
    for i, user in enumerate(ls_user):
        for img in glob.glob(os.path.join(path, user) + '\\*'):
            img = cv2.imread(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img = trans(img).float()
            img = norm(img).unsqueeze(0)
            with torch.no_grad():
                embed = nomask_model(img)
            # print(embed)
            embeddings.append(embed)
            names.append(user)

    torch.save(torch.stack(embeddings), db_path+"/nomask_ebd.pth")
    np.save(db_path+"/nomask_usernames", np.array(names))
    print('Done nomask!')

if __name__ == '__main__':
    # crop()    
    # create_nomask_ebd()
    create_mask_ebd()

