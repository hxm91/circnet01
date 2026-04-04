# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514

# import tensorflow as tf
import torch.utils.tensorboard as tb
# from torch.utils.tensorboard import SummaryWriter
import numpy as np
import scipy.misc
try:
  from StringIO import StringIO  # Python 2.7
except ImportError:
  from io import BytesIO         # Python 3.x


class Logger(object):

  def __init__(self, log_dir, num=20):
    """Create a summary writer logging to log_dir."""
    self.writer = tb.SummaryWriter(log_dir)
    #self.writer = tf.summary.create_file_writer(log_dir)
    # self.writer = tf.compat.v1.summary.FileWriter(log_dir)

    self.num = num
    # for i in range(self.num):
    #   writer_id_name = 'writer_%d'%i
    #   writer_id = tf.summary.FileWriter(log_dir)
    #   setattr(self, writer_id_name, writer_id)

  def scalar_summary(self, tag, value, step):
    """Log a scalar variable."""
    # summary = tb.Summary(value=[tb.Summary.Value(tag=tag, simple_value=value)])
    # self.writer.add_summary(summary, step)
    # summary = tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag=tag, simple_value=value)])
    self.writer.add_scalar(tag, value, step)
    self.writer.flush()
    '''
    with self.writer.as_default():
      tf.summary.scalar(tag,value, step=step)
      self.writer.flush()
    '''
  def image_summary(self, tag, images, step):
    """Log a list of images."""

    self.writer.add_images(tag, images, step)
    img_summaries = []
    for i, img in enumerate(images):
      # Write the image to a string
      try:
        s = StringIO()
      except:
        s = BytesIO()
      scipy.misc.toimage(img).save(s, format="png")
      # Create an Image object
      img_sum = tb.Summary.Image(encoded_image_string=s.getvalue(),
                                 height=img.shape[0],
                                 width=img.shape[1])
      # Create a Summary value
      img_summaries.append(tb.Summary.Value(
          tag='%s/%d' % (tag, i), image=img_sum))
      
    # Create and write Summary
    summary = tb.Summary(value=img_summaries)
    # self.writer.add_summary(summary, step)
    
    self.writer.flush()


  def histo_summary(self, tag, values, step, bins=1000):
    """Log a histogram of the tensor of values."""

    # Create a histogram using numpy
    counts, bin_edges = np.histogram(values, bins=bins)

    # Fill the fields of the histogram proto
    hist = tb.HistogramProto()
    hist.min = float(np.min(values))
    hist.max = float(np.max(values))
    hist.num = int(np.prod(values.shape))
    hist.sum = float(np.sum(values))
    hist.sum_squares = float(np.sum(values**2))

    # Drop the start of the first bin
    bin_edges = bin_edges[1:]

    # Add bin edges and counts
    for edge in bin_edges:
      hist.bucket_limit.append(edge)
    for c in counts:
      hist.bucket.append(c)

    # Create and write Summary
    with self.writer.as_default():
      self.writer.add_histogram(tag,hist, step=step)
      self.writer.flush()


  def multi_scalar_summary(self, tag, value, step):
    """Log multi scalar variable into one figure."""
    with self.writer.as_default():
      for i in range(self.num):
        tag=str(tag)+'_'+str(i)
        simple_value=value[i]
        tb.summary.scalar(tag,simple_value, step=step)
      self.writer.add_scalars(tag, value, step)
      self.writer.flush()
