import cv2

def segmentize (image_path, segment_width=2000, segment_height=2000):
    # Croping Formula ==> y:h, x:w
    idx, x_axis, x_width,  = 1, 0, segment_width
    y_axis, y_height = 0, segment_height
    img = cv2.imread(image_path)
    height, width, dept = img.shape

    while y_axis <= height:
        while x_axis <= width:
            crop = img[y_axis:y_height, x_axis:x_width]
            x_axis=x_width
            x_width+=segment_width
            cropped_image_path = "outputSeg/crop%d.tif" % idx
            cv2.imwrite(cropped_image_path, crop)
            idx+=1

        y_axis += segment_height
        y_height += segment_height
        x_axis, x_width = 0, segment_width


image_path = 'Satelite_cbers/CBERS_4A_WPM_20210121_205_139_L2_BAND1.tif'
segmentize(image_path)

#img = cv2.imread('tiffImages/CBERS_4A_WPM_20210121_205_139_L2_BAND4.tif')
# resolucion de => (14425, 14036, 3)