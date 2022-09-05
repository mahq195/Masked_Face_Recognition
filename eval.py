import os, glob, cv2
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1
from detect_face import inference

norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
trans = transforms.Compose([transforms.ToPILImage(),
                            transforms.Resize((128,128)),
                            transforms.ToTensor()
                           ])

nomask_model = InceptionResnetV1(pretrained='vggface2').eval() # Model of facenet for normal face recognition
mask_model = torch.load('model_saved\InceptionResNetV1_ArcFace.pt', map_location='cpu') # our model for masked face recognition


def save_mask_embeddings():
    data_embeddings = {'Name': [],
                        'Embedding': [],
                        'Ground_truth': []
                        }

    data_path = r'user_image'
    user_names = os.listdir(data_path)
    for i, name in enumerate(user_names): # name = user name
        user_path = os.path.join(data_path, name)
        img_ls = os.listdir(user_path)
        for j, img in enumerate(img_ls): #img = name of image
            
            img_path = os.path.join(user_path, img)
            
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            outputs = inference(image,
                        conf_thresh= 0.5,
                        iou_thresh= 0.5,
                        input_shape= (260,260),
                        draw_result=True)
            if outputs: 
                output = outputs[0]
                xmin, ymin, xmax, ymax = output[2], output[3], output[4], output[5]
                face = image[ymin:ymax+1, xmin:xmax+1]
                face = cv2.resize(face, (160, 160))
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                
                if output[0] == 0:
                    data_embeddings['Name'].append(img)
                    face = trans(face).float()
                    face = norm(face).unsqueeze(0)
                    with torch.no_grad():
                        embed = mask_model(face)['embeddings']
            
                    data_embeddings['Embedding'].append(embed)
                    data_embeddings['Ground_truth'].append(name)
                    print(i, 'done', name, j)

    torch.save(torch.stack(data_embeddings['Embedding']), 'database\\2data_embeddings.pth')
    np.save('database\\2data_names', np.array(data_embeddings['Ground_truth']))

def compute_acc(threshold):
    local_embeds = torch.load(r'database\2mask_ebd.pth')
    names = np.load(r'database\2mask_usernames.npy')
    
    ground_truth = np.load(r'database\2data_names.npy')
    mask_embeds = torch.load(r'database\2data_embeddings.pth')

    total = len(ground_truth)
    print(len(ground_truth))
    count = 0

    for i in range(len(mask_embeds)):
        mask_embed = mask_embeds[i]
        diff = mask_embed.flatten() - local_embeds.flatten(start_dim=1)
        norml2 = torch.sqrt(torch.sum(torch.pow(diff, 2), dim=1))
        min_dist, idx = torch.min(norml2, dim=0)
        print(min_dist)

        if min_dist > threshold:
            name = 'Unknown'
            # return min_dist, name 
        else:
            name = names[idx]
            # return min_dist, name
        print(ground_truth[i])
        if name == ground_truth[i]:
            count += 1  
            print(count)

    acc = int(count)/ int(total)
    print('Accuracy:', acc*100, '%')

if __name__ == "__main__":
    # save_mask_embeddings()
    compute_acc(1.1)
    


