import tensorflow as tf
from tensorflow import keras
class_names=['Jini', 'Lindi', 'Mora', 'Mothi', 'Sevardhana']
loaded_model=keras.models.load_model('ai.h5')
from tensorflow.keras.preprocessing import image
import numpy as np
image=keras.preprocessing.image.load_img('/home/rithvij/test1.jpeg',target_size=(32, 32))
image_array=keras.preprocessing.image.img_to_array(image)
normalized_image=image_array/255.0
import matplotlib.pyplot as plt
plt.imshow(normalized_image)
plt.title("Normalized Image")
plt.show()
predictions=loaded_model.predict(np.expand_dims(normalized_image, axis=0))
predicted_class_index=np.argmax(predictions)
predicted_class_label=class_names[predicted_class_index]
print(f"The predicted class is: {predicted_class_label}")
