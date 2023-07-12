import TableExtractor as te
import TableLinesRemover as tlr
import CellDetectionAndOCR as ocr
import cv2
import Utils as ut
from time import process_time
import os


pdf_path = 'images/HungHuy.pdf'
#ut.Convert_PDF_Image('image/test.pdf')
pages = ut.Convert_PDF_Image(pdf_path)
for page in pages:
    #page =pages[1]
    start = process_time()
    Table_Extractor = te.TABLEEXTRACTOR(page)
    Perspective_Corrected_Image = Table_Extractor.Execute()

    # Lines_Remover = tlr.TABLELINESREMOVER(Perspective_Corrected_Image)
    # Image_Without_Lines = Lines_Remover.Execute()

    Ocr_Result = ocr.OCR(Perspective_Corrected_Image,page)
    Ocr_Result.Execute()
    end = process_time()
    print('Timming the gioi:', end - start)


cv2.waitKey(0)
cv2.destroyAllWindows()
