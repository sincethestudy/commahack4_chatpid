import cv2
import numpy as np

# Load the ENet model
model = cv2.dnn.readNet('enet-model.net')

# Load the input image
image = cv2.imread('crossfar.png')

# Preprocess the input image
blob = cv2.dnn.blobFromImage(image, scalefactor=1/255.0, size=(1024, 512), mean=(0, 0, 0), swapRB=True, crop=False)

# Pass the preprocessed image through the ENet model
model.setInput(blob)
output = model.forward()

# Postprocess the predicted segmentation mask
classes = open('enet-classes.txt').read().strip().split('\n')
colors = open('enet-colors.txt').read().strip().split('\n')
colors = [np.array(c.split(',')).astype('int') for c in colors]
output = output[0, :, :, :]
output = np.argmax(output, axis=0)
output = cv2.resize(output, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
output = colors[output]

# Display the labeled segmented image
cv2.imshow('Labeled Segmented Image', output)
cv2.waitKey(0)
cv2.destroyAllWindows()