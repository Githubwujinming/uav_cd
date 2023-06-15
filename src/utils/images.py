
import cv2
import numpy as np
import torch
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
import torchvision.transforms as T
from PIL import Image

def tensor2im(input_image, imtype=np.uint8,scale = 255, normalize=True, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))

        image_numpy = np.transpose(image_numpy, (1, 2, 0))
        if normalize:
            image_numpy = (image_numpy*std  + mean)  # post-processing: tranpose and scaling

    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return (image_numpy*scale).astype(imtype)


def find_centers(image, show_num=10, min_area=100):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold image
    # thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    # Find contours
    cnts = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    res = image.copy()
    centers = []
    # Iterate thorugh contours and draw rectangle around each one
    for i, c in enumerate(cnts):
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        M = cv2.moments(c)
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        if len(centers)<=show_num:
            centers.append({"中心坐标":(cx, cy),
                        "面积":area})
        res = cv2.circle(res, (cx, cy), 4, (255,0,0), -1)
    return res, centers

def preprocess_image(
        img: np.ndarray, mean=(0.485, 0.456, 0.406), 
                std=(0.229, 0.224, 0.225)) -> torch.Tensor:
        preprocessing = T.Compose([
            T.ToTensor(),
            T.Resize(size=(512,512)),
            T.Normalize(mean=mean, std=std),
        ])
        return preprocessing(img).unsqueeze(0)
    
def save_image(image_numpy, image_path):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)