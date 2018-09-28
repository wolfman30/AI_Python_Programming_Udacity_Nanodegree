import argparse
import matplotlib.pyplot as plt
from train import get_inp_args

inp = get_inp_args()

def load_checkpoint(path):
    
    checkpoint = torch.load(path)
    model = checkpoint['model']
    optimizer = checkpoint['optimizer_state']
    model.class_to_idx = checkpoint['class_to_idx']
    
    return checkpoint, optimizer, model.class_to_idx

checkpoint, optimizer, class_to_idx = load_checkpoint('checkpoint.pth')

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    img = PIL.Image.open(image)
    
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

def imshow(image_path, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    image = process_image(image_path)
    image = image.numpy() #converts pytorch tensor to numpy array
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    img = process_image(image_path).float()
    img = img.unsqueeze_(0)
    
    checkpoint, _, class_to_idx = load_checkpoint('checkpoint.pth')
    
    model = checkpoint['model']
    
    model.eval() #turns dropout off 
    
    with torch.no_grad():
        
        output = model.forward(img.to('cuda'))
        probs, class_indices = F.softmax(output.data,dim=1).topk(5)
        
        probs = list(np.array(probs[0]))
        
        class_indices = list(np.array(class_indices[0]))
        
        classes = [k for k,v in checkpoint['class_to_idx'].items() if v in class_indices]
        
    return probs, classes

def read_json(file_path):
    with open(file_path, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name 
    
def sanity_check(img_path):
    #shows the image and the top five classes predicted with their respective probabilities
    plt.figure(figsize = [15,5])

    #left plot
    #img_path = test_dir + '/101/image_07988.jpg'
    ax1 = imshow(img_path, ax=plt.subplot(1, 2, 1))
    ax1.set_title(cat_to_name['10'])

    probs, classes = predict(img_path)
    class_names = [cat_to_name[i] for i in classes]
    plt.subplot(1,2,2)
    sb.barplot(x=probs, y = class_names, color='blue')
    plt.xlabel('Probability')

    plt.show();


sanity_check(inp.test_dir + '/10/image_07090.jpg')