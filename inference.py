import tensorflow as tf
import os
import matplotlib.pyplot as plt
from enet import ENet, ENet_arg_scope
from preprocessing import preprocess
from scipy.misc import imsave
import numpy as np
slim = tf.contrib.slim
import cv2

from label_loader import *

os.environ['CUDA_VISIBLE_DEVICES']='0'


image_dir = './dataset/testimg/'
images_list = sorted([os.path.join(image_dir, file) for file in os.listdir(image_dir) if file.endswith('.png')])

checkpoint_dir = "./weights/ade20k_combine_5"
checkpoint = tf.train.latest_checkpoint(checkpoint_dir)

num_initial_blocks = 1
skip_connections = False
stage_two_repeat = 2

load_label(os.path.abspath('./labels/camvid_label.mat'))

'''
#Labels to colours are obtained from here:
https://github.com/alexgkendall/SegNet-Tutorial/blob/c922cc4a4fcc7ce279dd998fb2d4a8703f34ebd7/Scripts/test_segmentation_camvid.py

However, the road_marking class is collapsed into the road class in the dataset provided.
'''

#Create the photo directory
photo_dir = checkpoint_dir + "/test_images"
if not os.path.exists(photo_dir):
    os.mkdir(photo_dir)

#Create a function to convert each pixel label to colour.
def grayscale_to_colour(image):
    print('Converting image...')
    image = image.reshape((360, 480, 1))
    image = np.repeat(image, 3, axis=-1)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            label = int(image[i,j,0])
            image[i,j] = np.array(label_to_colours[label])

    return image


with tf.Graph().as_default() as graph:
    images_tensor = tf.train.string_input_producer(images_list, shuffle=False)
    reader = tf.WholeFileReader()
    key, image_tensor = reader.read(images_tensor)
    image = tf.image.decode_png(image_tensor, channels=3)
    # image = tf.image.resize_image_with_crop_or_pad(image, 360, 480)
    # image = tf.cast(image, tf.float32)
    image = preprocess(image)
    images = tf.train.batch([image], batch_size=1, allow_smaller_final_batch=True)

    #Create the model inference
    with slim.arg_scope(ENet_arg_scope()):
        logits, probabilities = ENet(images,
                                     num_classes=12,
                                     batch_size=1,
                                     is_training=True,
                                     reuse=None,
                                     num_initial_blocks=num_initial_blocks,
                                     stage_two_repeat=stage_two_repeat,
                                     skip_connections=skip_connections)

    variables_to_restore = slim.get_variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)
    def restore_fn(sess):
        return saver.restore(sess, checkpoint)

    predictions = tf.argmax(probabilities, -1)
    predictions = tf.cast(predictions, tf.float32)

    sv = tf.train.Supervisor(logdir=None, init_fn=restore_fn)
    
    try:
        os.makedirs(photo_dir)
    except:
        pass

    with sv.managed_session() as sess:

        for i in range(len(images_list)):
            segmentations = sess.run(predictions)
            # print segmentations.shape

            

            converted_image = grayscale_to_colour(segmentations[0])
            print('Saving image {}/{}'.format(i+1, len(images_list)))
            converted_image = converted_image[...,::-1] #rgb to bgr
            filename = images_list[i].split('/')[-1]
            cv2.imwrite(os.path.join(photo_dir,filename), converted_image)
            output_label(segmentations[0], os.path.join(photo_dir, filename.split('.')[-2]+'_label.png'))
'''
            for j in xrange(segmentations.shape[0]):
                #Stop at the 233rd image as it's repeated
                if i*10 + j == 223:
                    break

                converted_image = grayscale_to_colour(segmentations[j])
                print 'Saving image %s/%s' %(i*10 + j, len(images_list))
                plt.axis('off')
                plt.imshow(converted_image)
                imsave(photo_dir + "/image_%s.png" %(i*10 + j), converted_image)
                plt.show()
'''
