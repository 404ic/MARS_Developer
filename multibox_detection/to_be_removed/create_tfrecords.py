"""
Create the tfrecord files for a dataset.

A lot of this code comes from the tensorflow inception example, so here is their license:

# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

###added  path to tfrecords features

"""
from datetime import datetime
import numpy as np
import os
from multiprocessing import Queue
import sys
import tensorflow as tf
import threading
import random
import pdb

def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
  """Wrapper for inserting float features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(image_example, image_buffer, height, width):
  """Build an Example proto for an example.
  Args:
    image_example: dict, an image example
    image_buffer: string, JPEG encoding of RGB image
    height: integer, image height in pixels
    width: integer, image width in pixels
  Returns:
    Example proto
  """
  
  # Required
  filename = str(image_example['filename'])
  id = str(image_example['id'])
  
  # Class label for the whole image
  image_class = image_example.get('class', {})
  class_label = image_class.get('label', 0)
  class_text = str(image_class.get('text', ''))
  
  # Bounding Boxes
  image_objects = image_example.get('object', {})
  image_bboxes = image_objects.get('bbox', {})
  xmin = image_bboxes.get('xmin', [])
  xmax = image_bboxes.get('xmax', [])
  ymin = image_bboxes.get('ymin', [])
  ymax = image_bboxes.get('ymax', [])
  bbox_labels = image_bboxes.get('label', [])
  bbox_scores = image_bboxes.get('score', [])
  bbox_count = image_bboxes.get('count', 0)
  
  # Parts
  image_parts = image_objects.get('parts', {})
  parts_x = image_parts.get('x', [])
  parts_y = image_parts.get('y', [])
  parts_v = image_parts.get('v', [])
  
  # Areas
  object_areas = image_objects.get('area', [])
  
  # Ids
  object_ids = image_objects.get('id', [])
  
  colorspace = 'RGB'
  channels = 3
  image_format = 'JPEG'

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': _int64_feature(height),
      'image/width': _int64_feature(width),
      'image/colorspace': _bytes_feature(colorspace),
      'image/channels': _int64_feature(channels),
      'image/class/label': _int64_feature(class_label),
      'image/class/text': _bytes_feature(class_text),
      'image/object/bbox/xmin': _float_feature(xmin),
      'image/object/bbox/xmax': _float_feature(xmax),
      'image/object/bbox/ymin': _float_feature(ymin),
      'image/object/bbox/ymax': _float_feature(ymax),
      'image/object/bbox/label': _int64_feature(bbox_labels),
      'image/object/bbox/count' : _int64_feature(bbox_count),
      'image/object/bbox/score' : _float_feature(bbox_scores),
      'image/object/parts/x' : _float_feature(parts_x),
      'image/object/parts/y' : _float_feature(parts_y),
      'image/object/parts/v' : _int64_feature(parts_v),
      'image/object/parts/count' : _int64_feature(len(parts_v)),
      'image/object/area' : _float_feature(object_areas),
      'image/object/id' : _int64_feature(object_ids),
      'image/format': _bytes_feature(image_format),
      'image/filename': _bytes_feature(os.path.basename(filename)),
      'image/path': _bytes_feature(os.path.dirname(filename)),
      'image/id': _bytes_feature(str(id)),
      'image/encoded': _bytes_feature(image_buffer)}))
  return example


class ImageCoder(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Create a single Session to run all image coding calls.
    self._sess = tf.Session()

    # Initializes function that converts PNG to JPEG data.
    self._png_data = tf.placeholder(dtype=tf.string)
    image = tf.image.decode_png(self._png_data, channels=3)
    self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)

    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def png_to_jpeg(self, image_data):
    return self._sess.run(self._png_to_jpeg,
                          feed_dict={self._png_data: image_data})

  def decode_jpeg(self, image_data):
    image = self._sess.run(self._decode_jpeg,
                           feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image

def _is_png(filename):
  """Determine if a file contains a PNG format image.
  Args:
    filename: string, path of the image file.
  Returns:
    boolean indicating if the image is a PNG.
  """
  filepath, file_extension = os.path.splitext(filename)
  if file_extension == '.png':
    return True
  else:
    return False

def _process_image(filename, coder):
  """Process a single image file.
  Args:
    filename: string, path to an image file e.g., '/path/to/example.JPG'.
    coder: instance of ImageCoder to provide TensorFlow image coding utils.
  Returns:
    image_buffer: string, JPEG encoding of RGB image.
    height: integer, image height in pixels.
    width: integer, image width in pixels.
  """
  # Read the image file.
  image_data = tf.gfile.FastGFile(filename, 'r').read()

  # Clean the dirty data.
  if _is_png(filename):
    # 1 image is a PNG.
    #print('Converting PNG to JPEG for %s' % filename)
    image_data = coder.png_to_jpeg(image_data)

  # Decode the RGB JPEG.
  image = coder.decode_jpeg(image_data)
  # Check that image converted to RGB
  assert len(image.shape) == 3
  height = image.shape[0]
  width = image.shape[1]
  assert image.shape[2] == 3

  return image_data, height, width


def _process_image_files_batch(coder, thread_index, ranges, name, output_directory, dataset, num_shards, error_queue):
  """Processes and saves list of images as TFRecord in 1 thread.
  Args:
    coder: instance of ImageCoder to provide TensorFlow image coding utils.
    thread_index: integer, unique batch to run index is within [0, len(ranges)).
    ranges: list of pairs of integers specifying ranges of each batches to
      analyze in parallel.
    name: string, unique identifier specifying the data set (e.g. `train` or `test`)
    output_directory: string, file path to store the tfrecord files.
    dataset: list, a list of image example dicts
    num_shards: integer number of shards for this data set.
    error_queue: Queue, a queue to place image examples that failed.
  """
  # Each thread produces N shards where N = int(num_shards / num_threads).
  # For instance, if num_shards = 128, and the num_threads = 2, then the first
  # thread would produce shards [0, 64).
  num_threads = len(ranges)
  assert not num_shards % num_threads
  num_shards_per_batch = int(num_shards / num_threads)

  shard_ranges = np.linspace(ranges[thread_index][0],
                             ranges[thread_index][1],
                             num_shards_per_batch + 1).astype(int)
  num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

  counter = 0
  error_counter = 0
  for s in range(num_shards_per_batch):
    # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
    shard = thread_index * num_shards_per_batch + s
    output_filename = '%s-%.5d-of-%.5d' % (name, shard, num_shards)
    output_file = os.path.join(output_directory, output_filename)
    writer = tf.python_io.TFRecordWriter(output_file)

    shard_counter = 0
    files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
    for i in files_in_shard:
      
      image_example = dataset[i]
      
      filename = str(image_example['filename'])

      try:
        image_buffer, height, width = _process_image(filename, coder)

        example = _convert_to_example(image_example, image_buffer, height, width)
        writer.write(example.SerializeToString())
        shard_counter += 1
        counter += 1
      except Exception as e:
        raise
        error_counter += 1
        error_queue.put(image_example)

      # if not counter % 1000:
      #   print('%s [thread %d]: Processed %d of %d images in thread batch, with %d errors.' %
      #         (datetime.now(), thread_index, counter, num_files_in_thread, error_counter))
        sys.stdout.flush()

    # print('%s [thread %d]: Wrote %d images to %s, with %d errors.' %
    #       (datetime.now(), thread_index, shard_counter, output_file, error_counter))
    sys.stdout.flush()
    shard_counter = 0
    
  # print('%s [thread %d]: Wrote %d images to %d shards, with %d errors.' %
  #       (datetime.now(), thread_index, counter, num_files_in_thread, error_counter))
  sys.stdout.flush()
  

def create(dataset, dataset_name, output_directory, num_shards, num_threads, shuffle=True):
  """Create the tfrecord files to be used to train or test a model.
  
  Args:
    dataset : [{
      "filename" : <REQUIRED: path to the image file>, 
      "id" : <REQUIRED: id of the image>,
      "class" : {
        "label" : <[0, num_classes)>,
        "text" : <text description of class>
      },
      "object" : {
        "bbox" : {
          "xmin" : [],
          "xmax" : [],
          "ymin" : [],
          "ymax" : [],
          "label" : []
        }
      }
    }]
    
    dataset_name: a name for the dataset
    
    output_directory: path to a directory to write the tfrecord files
    
    num_shards: the number of tfrecord files to create
    
    num_threads: the number of threads to use 

    shuffle : bool, should the image examples be shuffled or not prior to creating the tfrecords.
  
  Returns:
    list : a list of image examples that failed to process.
  """
  
  # Images in the tfrecords set must be shuffled properly
  if shuffle:
    random.shuffle(dataset)

  # Break all images into batches with a [ranges[i][0], ranges[i][1]].
  spacing = np.linspace(0, len(dataset), num_threads + 1).astype(np.int)
  ranges = []
  threads = []
  for i in range(len(spacing) - 1):
    ranges.append([spacing[i], spacing[i+1]])

  # Launch a thread for each batch.
  # print('Launching %d threads for spacings: %s' % (num_threads, ranges))
  sys.stdout.flush()

  # Create a mechanism for monitoring when all threads are finished.
  coord = tf.train.Coordinator()

  # Create a generic TensorFlow-based utility for converting all image codings.
  coder = ImageCoder()
  
  # A Queue to hold the image examples that fail to process. 
  error_queue = Queue()  
  
  threads = []
  for thread_index in range(len(ranges)):
    args = (coder, thread_index, ranges, dataset_name, output_directory, dataset, num_shards, error_queue)
    t = threading.Thread(target=_process_image_files_batch, args=args)
    t.start()
    threads.append(t)

  # Wait for all the threads to terminate.
  coord.join(threads)
  # print('%s: Finished writing all %d images in data set.' %
  #       (datetime.now(), len(dataset)))
  
  # Collect the errors
  errors = []
  while not error_queue.empty():
    errors.append(error_queue.get())
  print ('%d examples failed.' % (len(errors),))
  
  return errors
