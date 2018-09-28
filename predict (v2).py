import argparse
import numpy as np
import PIL
from PIL import Image
import json
import torch
from torch.nn import functional as F

def get_inp_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default='flowers/test/10/image_07090.jpg', 
                        help='path to single image for prediction')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], 
                        help='device on which to predict, either cuda gpu or cpu')
    parser.add_argument('--chpt_path', type=str, default='checkpoint.pth', 
                        help='path to checkpoint file')
    return parser.parse_args()
    

def read_json(file_path):
    with open(file_path, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name 

def load_checkpoint(path):
    
    checkpoint = torch.load(path)
    model = checkpoint['model']
    optimizer = checkpoint['optimizer_state']
    model.class_to_idx = checkpoint['class_to_idx']
    
    return checkpoint, optimizer, model.class_to_idx

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    img = Image.open(image_path)
    
    min_leng = 256
    ratio = min_leng/min(img.size)
    other_leng = int(max(img.size)*ratio)
    img = img.resize((min_leng,other_leng), PIL.Image.ANTIALIAS)
    
    #center cropping image to 224 X 224
    left = (min_leng - 224)/2
    top = (other_leng - 224)/2
    right = (min_leng + 224)/2
    bottom = (other_leng + 224)/2
    img = img.crop((left, top, right, bottom))
    
    #saves the new image replacing the original
    #img.save(image) 
    
    #converts the PIL image to a numpy array and scales the image to range [0,1]
    np_img = np.array(img)/np.max(img)
    
    #means and standard deviations of the red, green, blue channels, respectively
    means = np.array([0.485, 0.456, 0.406])
    stds = np.array([0.229, 0.224, 0.225])
    
    #normalizes the image
    np_img = (np_img - means)/stds
    
    #transposes the image so the color channels are first for pytorch format
    np_img = np.ndarray.transpose(np_img)
    
    py_tensor = torch.from_numpy(np_img)
    
    return py_tensor

def predict(image_path, checkpoint_path, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    img = process_image(image_path).float()
    img = img.unsqueeze_(0)
    
    checkpoint, _, class_to_idx = load_checkpoint(checkpoint_path)
    
    model = checkpoint['model']
    
    model.eval() #turns dropout off 
    
    with torch.no_grad():
        
        output = model.forward(img.to(device))
        probs, class_indices = F.softmax(output.data,dim=1).topk(5)
        
        probs = list(np.array(probs[0]))
        
        class_indices = list(np.array(class_indices[0]))
        
        classes = [k for k,v in checkpoint['class_to_idx'].items() if v in class_indices]
        
    return classes, probs

inp = get_inp_args()

classes, probs = predict(inp.image_path, inp.chpt_path, inp.device)

print()
print("Classes:", classes,'|', 'Probabilities:', probs)

    
#probs, classes = predict('flowers/test/10/image_07090.jpg')
#print(probs, classes)
#print()
cat_to_name = read_json('cat_to_name.json')
#print(cat_to_name)