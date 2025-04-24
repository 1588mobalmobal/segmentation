import segment_model as seg
import cv2
import matplotlib.pyplot as plt
import os 

model = seg.init_model()

dir = os.path.dirname(__file__)
image_path = os.path.join(dir, 'temp_image.jpg')

cv_image = cv2.imread(image_path)
plt.imshow(cv_image)
# plt.show()

image_tensor, image_rgb = seg.preprocess_image(cv_image)
pred_mask = seg.predict_segmentation(model, image_tensor)

plt.imshow(pred_mask)
plt.show()