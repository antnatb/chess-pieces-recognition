import cv2
import matplotlib.pyplot as plt
import numpy as np

# load image
image_path = "data\\train\\IMG_0294_JPG.rf.b7ae411904ff022e80b90f6267535f05.jpg"
img = cv2.imread(image_path)

# # Apply Gaussian blur
# blurred_img = cv2.GaussianBlur(img, (5, 5), 0)


# Apply thresholding
threshold = 40
thresholded_img = cv2.inRange(img, (0, 0, 0), (threshold, threshold, threshold))


# Apply opening
kernel = np.ones((5, 5), np.uint8)
opened_img = cv2.morphologyEx(thresholded_img, cv2.MORPH_OPEN, kernel)

# Apply dilation
dilated_img = cv2.dilate(opened_img, kernel, iterations=1)


# # Re-apply gaussian blur
# blurred_img = cv2.GaussianBlur(opened_img, (5, 5), 0)  
# plt.imshow(cv2.cvtColor(blurred_img, cv2.COLOR_BGR2RGB))
# plt.show()

# Show original vs final
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(dilated_img, cv2.COLOR_BGR2RGB))
plt.title("Processed Image")
plt.axis('off')
plt.show()
