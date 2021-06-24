# coding: utf-8
import tensorflow as tf
import io
import csv

# from PIL import Image
# import matplotlib.pyplot as plt


class TensorBoardLogger(object):
    def __init__(self, log_dir, session=None):
        self.log_dir = log_dir
        self.writer = tf.summary.FileWriter(self.log_dir)
        # print('TensorBoardLogger started. Run `tensorboard --logdir={}` to visualize'.format(self.log_dir))

        self.histograms = {}
        self.histogram_inputs = {}
        self.session = session or tf.get_default_session() or tf.Session()

    def tf_summary_image(self, image):
        size = image.get_size_inches() * image.dpi  # imageはMatplotlibのFigure形式
        height = int(size[0])
        width = int(size[1])
        channel = 1
        with io.BytesIO() as output:
            image.savefig(output, format="PNG")
            image_string = output.getvalue()
        return tf.Summary.Image(
            height=height,
            width=width,
            colorspace=channel,
            encoded_image_string=image_string,
        )

    def log(self, logs={}, histograms={}, images={}, epoch=0):
        # scalar logging
        for name, value in logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value
            summary_value.tag = name
            self.writer.add_summary(summary, epoch)

        # histograms
        for name, value in histograms.items():
            if name not in self.histograms:
                # make a tensor with no fixed shape
                self.histogram_inputs[name] = tf.Variable(value, validate_shape=False)
                # self.histograms[name] = tf.summary.histogram(
                #     name, self.histogram_inputs[name]
                # )
                self.histograms[name] = tf.compat.v1.summary.histogram(
                    name, self.histogram_inputs[name]
                )

            input_tensor = self.histogram_inputs[name]
            summary = self.histograms[name]
            summary_str = summary.eval(
                session=self.session, feed_dict={input_tensor.name: value}
            )
            self.writer.add_summary(summary_str, epoch)

        # images
        for name, image in images.items():
            tf_image = self.tf_summary_image(image)
            summary = tf.Summary(value=[tf.Summary.Value(tag=name, image=tf_image)])
            self.writer.add_summary(summary, epoch)

        self.writer.flush()


def Histograms(model, model_id, histograms):
    for layer in model:
        for i in range(len(layer.get_weights())):
            if "conv" in layer.name or "dense" in layer.name:
                name = model_id + "_" + layer.name + "/" + str(i)
                value = layer.get_weights()[i]
                histograms[name] = value
    return histograms
                

class RecordHistory:
    def __init__(self, csv_path, header):
        self.csv_path = csv_path
        self.header = header

    def generate_csv(self):
        with open(self.csv_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(self.header)

    def add_history(self, history):
        history_list = [history[key] for key in self.header]
        with open(self.csv_path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(history_list)

    def add_list(self, array):
        with open(self.csv_path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(array)