import cv2
import numpy as np
import Utils
import PIL
from PIL import Image
class TABLEEXTRACTOR:

    def __init__(self, npimage):
        self.npimage = npimage

    def Execute(self):
        self.Read_Image()
        self.Image_To_Grayscale()
        self.Threshold_Image()
        self.Invert_Image()
        self.Dilate_Image()
        self.Find_Contours()
        self.Filter_Contours()
        self.Find_Largest_Contour_By_Area()
        self.Order_Points_In_The_Contour_With_Max_Area()
        self.Calculate_New_Width_And_Height_Of_Image()
        self.Apply_Perspective_Transform()
        self.Add_10_Percent_Padding()
        self.store_process_image('corrected_image.jpg', self.perspective_corrected_image_with_padding)
        return self.perspective_corrected_image_with_padding

    def Read_Image(self):
        self.image = self.npimage

    def Image_To_Grayscale(self):
        self.grayscale_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def Blur_Image(self):
        self.blurred_image = cv2.blur(self.grayscale_image, (5, 5))

    def Threshold_Image(self):
        self.thresholded_image = cv2.threshold(self.grayscale_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    def Invert_Image(self):
        self.inverted_image = cv2.bitwise_not(self.thresholded_image)

    def Dilate_Image(self):
        self.dilated_image = cv2.dilate(self.inverted_image, None, iterations=5)

    def Find_Contours(self):
        self.contours, self.hierarchy = cv2.findContours(self.dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        self.image_with_all_contours = self.image.copy()
        cv2.drawContours(self.image_with_all_contours, self.contours, -1, (0, 255, 0), 3)

    def Filter_Contours(self):
        self.rectangular_contours = []
        count = 0
        peri_sum = 0
        for contour in self.contours:
            peri = 0.2*cv2.arcLength(contour, True)
            peri_sum += peri
            count += 1
        peri_mean = peri_sum/count
        for contour in self.contours:
            peri = 0.2*cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.05 * peri, True) 
            if peri > peri_mean/5 and len(approx) == 4:
                self.rectangular_contours.append(approx)
        self.image_with_only_rectangular_contours = self.image.copy()
        cv2.drawContours(self.image_with_only_rectangular_contours, self.rectangular_contours, -1, (0, 255, 0), 3)

    def Contour_To_Bounding_Boxes(self):
        self.bounding_boxes = []
        self.image_with_all_bounding_boxes = self.original_image.copy()
        for contour in self.rectangular_contours:
            x, y, w, h = cv2.boundingRect(contour)
            self.bounding_boxes.append((x, y, w, h))
            self.image_with_all_bounding_boxes = cv2.rectangle(self.image_with_all_bounding_boxes, (x, y), (x + w, y + h), (0, 255, 0), 5)

    def Find_Largest_Contour_By_Area(self):
        max_area = 0
        self.contour_with_max_area = None
        for contour in self.rectangular_contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                self.contour_with_max_area = contour
        self.image_with_contour_with_max_area = self.image.copy()
        cv2.drawContours(self.image_with_contour_with_max_area, [self.contour_with_max_area], -1, (0, 255, 0), 3)

    def Order_Points_In_The_Contour_With_Max_Area(self):
        self.contour_with_max_area_ordered = self.Order_Points(self.contour_with_max_area)
        self.image_with_points_plotted = self.image.copy()
        for point in self.contour_with_max_area_ordered:
            point_coordinates = (int(point[0]), int(point[1]))
            self.image_with_points_plotted = cv2.circle(self.image_with_points_plotted, point_coordinates, 10, (0, 0, 255), -1)

    def Calculate_New_Width_And_Height_Of_Image(self):
        existing_image_width = self.image.shape[1]
        existing_image_width_reduced_by_10_percent = int(existing_image_width * 0.9)
        
        distance_between_top_left_and_top_right = self.Calculate_Distance_Between_2_Points(self.contour_with_max_area_ordered[0], self.contour_with_max_area_ordered[1])
        distance_between_top_left_and_bottom_left = self.Calculate_Distance_Between_2_Points(self.contour_with_max_area_ordered[0], self.contour_with_max_area_ordered[3])

        aspect_ratio = distance_between_top_left_and_bottom_left / distance_between_top_left_and_top_right

        self.new_image_width = existing_image_width_reduced_by_10_percent
        self.new_image_height = int(self.new_image_width * aspect_ratio)

    def Apply_Perspective_Transform(self):
        pts1 = np.float32(self.contour_with_max_area_ordered)
        pts2 = np.float32([[0, 0], [self.new_image_width, 0], [self.new_image_width, self.new_image_height], [0, self.new_image_height]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        self.perspective_corrected_image = cv2.warpPerspective(self.image, matrix, (self.new_image_width, self.new_image_height))

    def Add_10_Percent_Padding(self):
        image_height = self.image.shape[0]
        padding = int(image_height * 0.1)
        self.perspective_corrected_image_with_padding = cv2.copyMakeBorder(self.perspective_corrected_image, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=[255, 255, 255])

    def Draw_Contours(self):
        self.image_with_contours = self.image.copy()
        cv2.drawContours(self.image_with_contours,  [ self.contour_with_max_area ], -1, (0, 255, 0), 1)

    def Calculate_Distance_Between_2_Points(self, p1, p2):
        dis = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
        return dis
    
    def Order_Points(self, pts):
        pts = pts.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        return rect
    
    def store_process_image(self, file_name, image):
        path = "./process_images/table_extractor/" + file_name
        cv2.imwrite(path, image)
