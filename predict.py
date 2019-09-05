import argparse 
import torch 
import numpy as np
import json
import sys

from torch import nn, optim
from torchvision import datasets, models, transforms
from PIL import Image


def resize_crop_image(image):
    img = Image.open(image)
    dx, dy = img.size
    aspect_ratio = max(dx, dy)/min(dx, dy)
    width = 256
    height = int(256 * aspect_ratio)
    if(dx > dy):
        width = height
        height = 256
    crop_size = 244
    img = img.resize([width, height]) 
    x1 = (width - crop_size)/2
    y1 = (height - crop_size)/2
    x2 = (width + crop_size)/2
    y2 = (height + crop_size)/2
    img = img.crop((x1, y1, x2, y2))
    cropped_image = np.array(img)
    cropped_image = cropped_image.astype('float64')
    cropped_image = cropped_image / [255,255,255]
    cropped_image = (cropped_image - [0.485, 0.456, 0.406])/ [0.229, 0.224, 0.225]
    cropped_image = cropped_image.transpose((2, 0, 1))
    return cropped_image

def load_model():
    model_info = torch.load(args.model_checkpoint)
    model = model_info['model']
    model.classifier = model_info['classifier']
    model.load_state_dict(model_info['state_dict'])
    model.class_to_idx = model_info['class_to_idx']
    return model

def image_predict(image_path, topk):
    with torch.no_grad():
        image = resize_crop_image(image_path)
        image = torch.from_numpy(image)
        image.unsqueeze_(0)
        image = image.float()
        model = load_model()
        if (args.gpu):
           image = image.cuda()
           model = model.cuda()
        else:
            image = image.cpu()
            model = model.cpu()
        outputs = model(image)
        probs_top, labs_top = torch.exp(outputs).topk(topk)
        probs_top, labs_top = probs_top[0].tolist(), labs_top[0].add(1).tolist()
        results = zip(probs_top,labs_top)
        return results


def read_categories():
    if (args.category_names is not None):
        cat_file = args.category_names 
        jfile = json.loads(open(cat_file).read())
        return jfile
    return None

def display_image(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    image = image.transpose((1, 2, 0))
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def show_prediction(image_path,topk):
    prediction = image_predict(image_path,topk)
    cat_file = read_categories()
    i = 0
    for p, c in prediction:
        i = i + 1
        p = str(round(p,4) * 100.) + '%'
        if (cat_file):
            c = cat_file.get(str(c),'None')
        else:
            c = ' class {}'.format(str(c))
        print("{}.{} ({})".format(i, c,p))
    return None
    
def parse():
    parser = argparse.ArgumentParser(description='Image Classifier')
    parser.add_argument('image_input', help='image file path(required)')
    parser.add_argument('model_checkpoint', help='Classification Model(required)')
    parser.add_argument('--top_k', help='Number of prediction categories [default 5].')
    parser.add_argument('--category_names', help='file with category names')
    parser.add_argument('--gpu', action='store_true', help='enable gpu mode')
    args = parser.parse_args()
    return args

def main():
    global args
    args = parse() 
    if (args.gpu and not torch.cuda.is_available()):
        raise Exception("--gpu option enabled...but GPU unavailable")
    top_k = args.top_k
    if (top_k is None):
        top_k = 5  
    image_path = args.image_input
    show_prediction(image_path,int(top_k))

main()
