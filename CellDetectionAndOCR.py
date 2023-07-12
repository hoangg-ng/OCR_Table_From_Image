import cv2
import numpy as np
import subprocess
import json 
import csv
import SignatureDetection as sd
import pytesseract
import tesserocr
from tesserocr import PyTessBaseAPI, PSM, OEM
from PIL import Image
import Utils
import time
from time import process_time
import TableExtractor as te 

class OCR:

    def __init__(self, original_image, image_path):
        # self.thresholded_image = image
        self.original_image = original_image
        self.image_path = image_path

    def Execute(self):
        self.Convert_Image_To_Grayscale()
        self.Threshold_Image()
        self.Invert_Image()
        self.Dilate_Image()
        self.Erode_Vertical_Lines()
        self.Erode_Horizontal_Lines()
        self.Combine_Eroded_Images()
        self.Dilate_Combined_Image()
        self.Find_Contours()
        self.Filter_Contours()
        self.Convert_Contour_To_Bounding_Boxes()
        self.Mean_height = self.Get_Mean_Height_Of_Bounding_Boxes()
        self.Sort_Bounding_Boxes_By_Y_Coordinate()
        self.Add_Bounding_Boxes_To_Row()
        self.Sort_Rows_By_X()
        self.Crop_Bounding_Box_And_Ocr()
        self.Csv_Generator()
        self.Create_Json()
        self.store_process_image('bounding_box.jpg', self.image_with_all_bounding_boxes)
        self.store_process_image('find_contour.jpg',self.image_with_all_contours)
        self.store_process_image('rec_contour.jpg',self.image_with_only_rectangular_contours)

        # self.store_process_image('debug.jpg',self.dilated_image)
    def Convert_Image_To_Grayscale(self):
        self.grayscale_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)

    def Blur_Image(self):
        self.blurred_image = cv2.blur(self.grayscale_image, (5, 5))

    def Threshold_Image(self):
        self.thresholded_image = cv2.threshold(self.grayscale_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    def Invert_Image(self):
        self.inverted_image = cv2.bitwise_not(self.thresholded_image)

    def Dilate_Image(self):
        self.dilated_image = cv2.dilate(self.inverted_image, None, iterations=5)

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
        self.combined_image_dilated = cv2.dilate(self.combined_image, None , iterations=5)

    def Find_Contours(self):
        self.contours, self.hierarchy = cv2.findContours(self.combined_image_dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        self.image_with_all_contours = self.original_image.copy()
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
        del self.rectangular_contours[0]
        self.image_with_only_rectangular_contours = self.original_image.copy()
    
        cv2.drawContours(self.image_with_only_rectangular_contours, self.rectangular_contours, -1, (0, 255, 0), 3)

    def Convert_Contour_To_Bounding_Boxes(self):
        self.bounding_boxes = []
        self.image_with_all_bounding_boxes = self.original_image.copy()
        for contour in self.rectangular_contours:
            x, y, w, h = cv2.boundingRect(contour)
            self.bounding_boxes.append((x, y, w, h))
            self.image_with_all_bounding_boxes = cv2.rectangle(self.image_with_all_bounding_boxes, (x, y), (x + w, y + h), (0, 255, 0), 5)

    def Get_Mean_Height_Of_Bounding_Boxes(self):
        heights = []
        for bounding_box in self.bounding_boxes:
            x, y, w, h = bounding_box
            heights.append(h)
        return np.mean(heights)

    def Sort_Bounding_Boxes_By_Y_Coordinate(self):
        self.bounding_boxes = sorted(self.bounding_boxes, key=lambda x: x[1])

    def Add_Bounding_Boxes_To_Row(self):
        self.rows = []
        Half_Of_Mean_Height = self.Mean_height / 2
        Current_Row = [ self.bounding_boxes[0] ]
        for bounding_box in self.bounding_boxes[1:]:
            current_bounding_box_y = bounding_box[1]
            previous_bounding_box_y = Current_Row[-1][1]
            distance_between_bounding_boxes = abs(current_bounding_box_y - previous_bounding_box_y)
            if distance_between_bounding_boxes <= Half_Of_Mean_Height:
                Current_Row.append(bounding_box)
            else:
                self.rows.append(Current_Row)
                Current_Row = [ bounding_box ]
        self.rows.append(Current_Row)

    def Sort_Rows_By_X(self):
        for row in self.rows:
            row.sort(key=lambda x: x[0])

    def Crop_Bounding_Box_And_Ocr(self):
        self.table = []
        current_row = []
        table_content = []
        image_number = 0
        #data_field = ['Row', 'Fullname', 'Unit', 'ParentUnit', 'Code', 'Birth','Relation', 'Status']
        #a = len(data_field)-1
        #self.table.append(data_field)
        for row in self.rows:
            a = len(row)-1
            for bounding_box in row:
                x, y, w, h = bounding_box
                cropped_image = self.original_image[y:y+h, x:x+w]
                table_content.append(cropped_image)
                if int(image_number)%(a+1) == a and int(image_number > a):
                    string = ''
                    sliced_image = Utils.Ndarray_To_Img(cropped_image)
                    results_from_ocr = sd.SignDect(sliced_image)
                    string = str(results_from_ocr)
                else:
                    string = ''
                    sliced_image = Utils.Ndarray_To_Img(cropped_image)
                    results_from_ocr = self.Py_Tesseract(sliced_image)
                    for x in results_from_ocr:
                        if x == '\n':
                            string = string + ' '
                        elif x != '\n':
                            string = string + x
                    print(string)
                current_row.append(string)
                string = ''
                image_number += 1
            self.table.append(current_row)
            current_row = []
        print(len(table_content))
        for i in range(len(self.table)):
            print(self.table[i], "\n")

    def Tesseract(self, image_path):
        output = subprocess.getoutput('tesseract ' + image_path + ' - -l vie --oem 3 --psm 6 --dpi 256')
        # -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789().calmg* "
        output = output.strip()
        return output

    def Csv_Generator(self):
        with open("output.csv", "w") as f:
            for row in self.table:
                f.write(",".join(row) + "\n")

    def Create_Json(self):
        data = {}
        with open('output.csv', encoding='utf-8') as csvf:
            csvReader = csv.DictReader(csvf)
            for row in csvReader:
                key = row['STT']
                data[key] = row
    
        with open('output.json', 'a', encoding='utf-8') as jsonf:
            jsonf.write(json.dumps(data, indent=4, ensure_ascii=False))
    
    def Py_Tesseract(self, sliced_image):
        start = process_time()
        output = pytesseract.image_to_string(sliced_image,config='--oem 3 --psm 6 --dpi 128',lang='vie')
        output = output.strip()
        end = process_time()
        print ("OCR:", end - start)
        return output

    def store_process_image(self, file_name, image):
        path = "./process_images/ocr/" + file_name
        cv2.imwrite(path, image)


