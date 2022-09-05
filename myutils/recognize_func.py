# from threading import local
# from detect_face_mask import inference
# import cv2
import torch
import numpy as np
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1

device =  torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
trans = transforms.Compose([transforms.ToPILImage(),
                            transforms.Resize((128, 128)),
                            transforms.ToTensor()
                           ])

nomask_model = InceptionResnetV1(pretrained='vggface2').eval() # Model of facenet for normal face recognition
mask_model = torch.load('model_saved\InceptionResNetV1_ArcFace.pt', map_location='cpu') # our model for masked face recognition




def recognize_mask(face, threshold):
    local_embeds = torch.load(r'database\2mask_ebd.pth')
    # print(type(mask_embeds))
    names = np.load(r'database\2mask_usernames.npy')

    mask = trans(face).float()
    mask = norm(mask).unsqueeze(0)
    with torch.no_grad():
        mask_embed = mask_model(mask)['embeddings']
    # print(mask_embed)
    diff = mask_embed.flatten() - local_embeds.flatten(start_dim=1)
    norml2 = torch.sqrt(torch.sum(torch.pow(diff, 2), dim=1))
    min_dist, idx = torch.min(norml2, dim=0)

    if min_dist > threshold:
        name = 'Unknown'
        return min_dist, name 
    else:
        name = names[idx]
        return min_dist, name

def recognize_nomask(face, threshold):
    local_embeds = torch.load(r'database\2nomask_ebd.pth')
    # print(type(local_embeds))
    names = np.load(r'database\2nomask_usernames.npy')
    
    nomask = trans(face).float().unsqueeze(0)
    # nomask = norm(nomask)
    with torch.no_grad():
        embed = nomask_model(nomask)
    # print(embed)
    diff = embed.flatten() - local_embeds.flatten(start_dim=1)
    norml2 = torch.sqrt(torch.sum(torch.pow(diff, 2), dim=1))
    min_dist, idx = torch.min(norml2, dim=0)

    if min_dist > threshold:
        name = 'Unknown'
        return min_dist, name 
    else:
        name = names[idx]
        return min_dist, name