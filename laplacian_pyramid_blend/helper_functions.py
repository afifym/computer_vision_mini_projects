import cv2
import numpy as np
import matplotlib.pyplot as plt

BLUE, GREEN, RED = (255, 0, 0), (0, 255, 0), (0, 0, 255)

WHITE_LOWER, WHITE_UPPER = (200, 200, 200), (255, 255, 255)
BLACK_LOWER, BLACK_UPPER = (0, 0, 0), (10, 10, 10)


def create_trackbar():
	def nothing():
		pass

	cv2.namedWindow("Trackbar")
	cv2.createTrackbar("Hue (-)", "Trackbar", 0, 255, nothing)
	cv2.createTrackbar("Val (-)", "Trackbar", 0, 255, nothing)
	cv2.createTrackbar("Sat (-)", "Trackbar", 0, 255, nothing)
	cv2.createTrackbar("Hue (+)", "Trackbar", 255, 255, nothing)
	cv2.createTrackbar("Val (+)", "Trackbar", 255, 255, nothing)
	cv2.createTrackbar("Sat (+)", "Trackbar", 255, 255, nothing)


def update_trackbar():
	lower = cv2.getTrackbarPos("Hue (-)", "Trackbar"), \
			cv2.getTrackbarPos("Val (-)", "Trackbar"), \
			cv2.getTrackbarPos("Sat (-)", "Trackbar")

	upper = cv2.getTrackbarPos("Hue (+)", "Trackbar"), \
			cv2.getTrackbarPos("Val (+)", "Trackbar"), \
			cv2.getTrackbarPos("Sat (+)", "Trackbar")

	return lower, upper


def mask_from_color(image, lower, upper, erode=0, dilate=0):
    mask = cv2.inRange(image, lower, upper)

    if erode:
        mask = cv2.erode(mask, None, iterations=erode)
    if dilate:
        mask = cv2.dilate(mask, None, iterations=dilate)

    return mask


def resize_same(image, new_height=False, new_width=False):
    ar = image.shape[1]/image.shape[0]

    if new_height:
        new_width = int(ar * new_height)
    elif new_width:
        new_height = int(new_width/ar)

    return cv2.resize(image, dsize=(new_width, new_height))


def image_template_matching(image, template, maxCorr):
  
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF)
    (_, new_maxCorr, _, maxLoc) = cv2.minMaxLoc(result)

    if new_maxCorr > maxCorr:
        best_template, best_corr = template, new_maxCorr
        (tH, tW) = template.shape[:2]
        (xi, yi) = (int(maxLoc[0]), int(maxLoc[1]))
        (xf, yf) = (int((maxLoc[0] + tW)), int((maxLoc[1] + tH)))

    else:
        return None

    return (best_template, best_corr, xi, yi, xf, yf)


# def image_image_matching(image_1, image_2, maxCorr, edging=False)
#     if edged:
#         image_1 = cv2.Canny(image_1, 75, 200)
#         image_2 = cv2.Canny(image_2, 75, 200)
    
    
#     image_template_matching(image, template, maxCorr):
    
    
def image_center_region(image, region_width=False, region_height=False, ret_ranges=False):
    if region_width and region_height:
        xc, yc = image.shape[0]//2, image.shape[1]//2
        x_min, x_max = xc-region_width//2, xc+region_width//2
        y_min, y_max = yc-region_height//2, yc+region_height//2
        
        if ret_ranges:
            return image[x_min:x_max, y_min:y_max, :], x_min, x_max, y_min, y_max
            
        return image[x_min:x_max, y_min:y_max, :]
    
    ar = image.shape[1]/image.shape[0]
    
    if region_height:
        region_width = int(ar * region_height)
    elif region_width:
        region_height = int(region_width/ar)
        
    xc, yc = image.shape[0]//2, image.shape[1]//2
    x_min, x_max = xc-region_width//2, xc+region_width//2
    y_min, y_max = yc-region_height//2, yc+region_height//2
    
    if ret_ranges:
        return image[x_min:x_max, y_min:y_max, :], x_min, x_max, y_min, y_max
    return image[x_min:x_max, y_min:y_max, :]

def color_ranges_from_roi(roi):
    c0_min, c0_max = int(roi[:,:,0].min()), int(roi[:,:,0].max())
    c1_min, c1_max = int(roi[:,:,1].min()), int(roi[:,:,1].max())
    c2_min, c2_max = int(roi[:,:,2].min()), int(roi[:,:,2].max())
    
    return (c0_min, c1_min, c2_min), (c0_max, c1_max, c2_max)


def plt_imshow_bgr(image):
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	plt.imshow(image)


def plt_imshow_multi(*args, rows=2, cols=2):
	images = list(*args)
	for i in range(len(images)):
		plt.subplot(rows, cols, i+1)
		plt.imshow(images[i], 'gray')
		plt.title(str(i+1))
        

def crop_center(image, to_remove):
    return image[to_remove:-to_remove, to_remove:-to_remove,:]