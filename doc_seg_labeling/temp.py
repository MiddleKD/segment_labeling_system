import cv2
import numpy as np

# Read the image
img = cv2.imread('./data/image/0.jpg', cv2.IMREAD_GRAYSCALE)

# Threshold the image to get a binary image
_, binary_image = cv2.threshold(img, 140, 255, cv2.THRESH_BINARY_INV)

# Find connected components
num_labels, image_components = cv2.connectedComponents(binary_image)

# Create an output image to draw the components
output_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

# Loop through each label and assign a unique color
for label in range(1, num_labels):  # Start from 1 to ignore the background
    mask = image_components == label
    color = [int(j) for j in np.random.choice(range(256), size=3)]  # Random color
    output_image[mask] = color

# Overlay the components on the original image
overlay_image = cv2.addWeighted(img, 0.5, cv2.cvtColor(output_image, cv2.COLOR_BGR2GRAY), 0.5, 0)

# Display the image
cv2.imshow('Connected Components', overlay_image)
cv2.waitKey(0)
cv2.destroyAllWindows()