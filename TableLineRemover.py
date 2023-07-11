import cv2
import numpy as np
from numpy import asarray
import Utils

class TABLELINESREMOVER:

    def __init__(self, npimage):
        self.image = npimage

    def Execute(self):
        self.Image_To_Grayscale()
        self.Threshold_Image()
        self.Invert_Image()
        self.Erode_Vertical_Lines()
        self.Erode_Horizontal_Lines()
        self.Combine_Eroded_Images()
        self.Dilate_Combined_Image()
        self.Subtract_Combined_And_Dilated_Image_From_Original_Image()
        self.Remove_Noise_With_Erode_And_Dilate()
        return self.image_without_lines_noise_removed

    def Image_To_Grayscale(self):
        self.grey = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def Threshold_Image(self):
        self.thresholded_image = cv2.threshold(self.grey, 127, 255, cv2.THRESH_BINARY)[1]

    def Invert_Image(self):
        self.inverted_image = cv2.bitwise_not(self.thresholded_image)

    def Erode_Vertical_Lines(self):
        hor = np.array([[1,1,1,1,1,1]])
        self.vertical_lines_eroded_image = cv2.erode(self.inverted_image, hor, iterations=10)
        self.vertical_lines_eroded_image = cv2.dilate(self.vertical_lines_eroded_image, hor, iterations=10)

    def Erode_Horizontal_Lines(self):
        ver = np.array([[1],
               [1],
               [1],
               [1],
               [1],
               [1],
               [1]])
        self.horizontal_lines_eroded_image = cv2.erode(self.inverted_image, ver, iterations=10)
        self.horizontal_lines_eroded_image = cv2.dilate(self.horizontal_lines_eroded_image, ver, iterations=10)

    def Combine_Eroded_Images(self):
        self.combined_image = cv2.add(self.vertical_lines_eroded_image, self.horizontal_lines_eroded_image)

    def Dilate_Combined_Image(self):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        self.combined_image_dilated = cv2.dilate(self.combined_image, kernel, iterations=5)

    def Subtract_Combined_And_Dilated_Image_From_Original_Image(self):
        self.image_without_lines = cv2.subtract(self.inverted_image, self.combined_image_dilated)

    def Remove_Noise_With_Erode_And_Dilate(self):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        self.image_without_lines_noise_removed = cv2.erode(self.image_without_lines, kernel, iterations=1)
        self.image_without_lines_noise_removed = cv2.dilate(self.image_without_lines_noise_removed, kernel, iterations=1)

