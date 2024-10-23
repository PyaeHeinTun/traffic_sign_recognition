from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

# Total Generated number
total_number = 170

# data_gen = ImageDataGenerator(rescale=1. /255, shear_range=0.2, zoom_range=0.2, horizontal_flip=False,rotation_range=0,vertical_flip=False,brightness_range=(0.1,0.9),width_shift_range=0.3, fill_mode='nearest')
data_gen = ImageDataGenerator(rescale=1. /255,
                              shear_range=0.2,
                              zoom_range=[0.6,1.5],
                              channel_shift_range=150,
                              width_shift_range=0.08,
                              height_shift_range=0.08,
                              fill_mode="nearest",
                              rotation_range=5)

# Create image to tensor
img = image.image_utils.load_img("TrainDataset/1/1.png", grayscale=False)
arr = image.image_utils.img_to_array(img)
tensor_image = arr.reshape((1, ) + arr.shape)

for i, _ in enumerate(data_gen.flow(x=tensor_image,batch_size=1,save_to_dir="populated",save_prefix="00000_00001",save_format=".png")):
    if i > total_number:
        break

