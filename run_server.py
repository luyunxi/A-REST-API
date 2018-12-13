# USAGE
# The server is the application that uses deep learn models for the clients.
# author；Lu Yunxi
# Start the server:
# 	python run_keras_server.py
# Submit a request via cURL:
# 	curl -X POST -F image=@dog.jpg 'http://localhost:5000/predict'
# Submita a request via Python:
#	python simple_request.py

# import the necessary packages

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import vis

import deeplab_model
from utils import preprocessing
from utils import dataset_util

from tensorflow.python import debug as tf_debug
from keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import flask
import io
import tensorflow as tf
import datetime

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = None

absPath = os.getcwd()

_NUM_CLASSES = 2
parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', type=str, default='images',
                    help='The directory containing the image data.')

parser.add_argument('--output_dir', type=str, default='./inference_output',
                    help='Path to the directory to generate the inference results')

parser.add_argument('--infer_data_list', type=str, default='./time.txt',
                    help='Path to the file listing the inferring images.')

parser.add_argument('--model_dir', type=str, default='./model',
                    help="Base directory for the model. "
                         "Make sure 'model_checkpoint_path' given in 'checkpoint' file matches "
                         "with checkpoint name.")

parser.add_argument('--base_architecture', type=str, default='resnet_v2_101',
                    choices=['resnet_v2_50', 'resnet_v2_101'],
                    help='The architecture of base Resnet building block.')

parser.add_argument('--output_stride', type=int, default=16,
                    choices=[8, 16],
                    help='Output stride for DeepLab v3. Currently 8 or 16 is supported.')

parser.add_argument('--debug', action='store_true',
                    help='Whether to use debugger to track down bad values during training.')


def load_model():
    # load the pre-trained model. Here, we can load deffient modes for diffient tasks.
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

    # classify the input image and then initialize the list
    # of predictions to return to the client

    global model
    model = tf.estimator.Estimator(
        model_fn=deeplab_model.deeplabv3_plus_model_fn,
        model_dir=FLAGS.model_dir,
        params={
            'output_stride': FLAGS.output_stride,
            'batch_size': 1,  # Batch size must be 1 because the images' size may differ
            'base_architecture': FLAGS.base_architecture,
            'pre_trained_model': None,
            'batch_norm_decay': None,
            'num_classes': _NUM_CLASSES,
        })

    global graph
    graph = tf.get_default_graph()

    """
	global model
	model = ResNet50(weights="imagenet")
	global graph
	graph = tf.get_default_graph()
	"""


def prepare_image(image, target):
    # prepare images for usage.

    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize the input image and preprocess it
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    # image = imagenet_utils.preprocess_input(image)

    # return the processed image
    return image


@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # read the image in PIL format
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))

            # if necesary, we can perpare images ahead.
            # preprocess the image and prepare it for classification
            # image = prepare_image(image, target=(224, 224))
            # image = np.matrix(image, dtype=np.int32)

            # make file "time.txt" and write the path of the image get from clinet
            file_time = open("time.txt", "w+")
            fileName = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
            image.save('./images/' + fileName + '.jpg')
            file_time.write(absPath + '/images/' + fileName + '.jpg')
            file_time.close()

            pred_hooks = None
            if FLAGS.debug:
                debug_hook = tf_debug.LocalCLIDebugHook()
                pred_hooks = [debug_hook]

            examples = dataset_util.read_examples_list(FLAGS.infer_data_list)
            image_files = [os.path.join(FLAGS.data_dir, filename) for filename in examples]

            with graph.as_default():
                try:
                    global model
                    predictions = model.predict(input_fn=lambda: preprocessing.eval_input_fn(image_files),
                                                hooks=pred_hooks)
                except Exception as e:
                    raise TypeError("bad input") from e

            output_dir = FLAGS.output_dir
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # print(image_files)
            for pred_dict, image_path in zip(predictions, image_files):
                image_basename = os.path.splitext(os.path.basename(image_path))[0]
                output_filename = image_basename + '.png'
                path_to_output = os.path.join(output_dir, output_filename)
                orginalImage = np.array(Image.open(image_path))
                img = Image.fromarray(orginalImage)

                mask = pred_dict['decoded_labels']
                mask = Image.fromarray(mask)

                mask = mask.convert('L')
                threshold = 10
                table = []
                for i in range(256):
                    if i < threshold:
                        table.append(0)
                    else:
                        table.append(1)
                mask = mask.point(table, '1')
                mask = np.matrix(mask, dtype=np.int32)

                voc_palette = vis.make_palette(2)
                out_im = Image.fromarray(vis.color_seg(mask, voc_palette))
                (shotname, extension) = os.path.splitext(image_path)

                # mask images
                # out_im.save(shotname+'out.png')

                masked_im = Image.fromarray(vis.vis_seg(img, mask, voc_palette))
                # get vis in images/
                # masked_im.save(shotname+'vis.png')

                print("generating:", path_to_output)
                masked_im.save(path_to_output)

            # results = imagenet_utils.decode_predictions(preds)
            data["predictions"] = []

            # loop over the results and add them to the list of
            # returned predictions
            # for (imagenetID, label, prob) in 1:
            r = {"label": "apple", "probability": float(0.999)}
            data["predictions"].append(r)

            # indicate that the request was a success
            data["success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(data)


# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))
    FLAGS, unparsed = parser.parse_known_args()
    load_model()
    app.run()