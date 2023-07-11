from pdf2image import convert_from_path
from PIL import Image
import numpy as np
from numpy import asarray
import cv2

def Convert_PDF_Image(file_path):
  pages = []
  input_path = file_path
  images = convert_from_path(input_path, dpi=500, thread_count=4)
  for image in images:
    ndarr = Img_To_Ndarray(image)
    pages.append(ndarr)
  return pages

def Img_To_Ndarray(pil_img):
  cv2_img = np.array(pil_img)
  cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_RGB2BGR)  
  return cv2_img

def Ndarray_To_Img(cv2_img):
  cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
  pil_img = Image.fromarray(cv2_img)
  return pil_img
