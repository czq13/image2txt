# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys

import tensorflow as tf
import thulac
#import jieba

reload(sys)
sys.setdefaultencoding('utf8')


tf.flags.DEFINE_string("data_path", "./captions",
                       "Caption files directory")
tf.flags.DEFINE_string("parse_method", "default",
                       "Parsing method (default, thulac or jieba). default is letter by letter. jieba may have some encoding problem.")
tf.flags.DEFINE_string("word_id_output_file", "./captions/word_to_id.txt",
                       "Output vocabulary file of word to id pairs.")
tf.flags.DEFINE_string("word_count_output_file", "./captions/word_count.txt",
                       "Output vocabulary file of word count pairs.")

FLAGS = tf.flags.FLAGS

# parsing
def _read_words_with_thulac(sentence, thu):
  tokenized_caption = ["<S>"]
  temp = thu.cut(sentence, text=True).split(" ")
  tokenized_caption.extend(temp)
  tokenized_caption.append("</S>")
  return tokenized_caption


def _read_words_with_jieba(sentence):
  tokenized_caption = ["<S>"]
  temp=list(jieba.cut(sentence))
  temp=[a.encode('utf-8') for a in temp]
  tokenized_caption.extend(temp)
  tokenized_caption.append("</S>")
  return tokenized_caption


def _read_words(sentence):
  tokenized_caption = ["<S>"]
  temp=[sentence[i] for i in range(len(sentence))]
  tokenized_caption.extend(temp)
  tokenized_caption.append("</S>")
  return tokenized_caption


def _build_vocab(filename):
  with tf.gfile.GFile(filename, "r") as f:
    text = f.read().decode("utf-8").split("\r\n")
    image_caption_pairs=[]
    for line in text:
      # print(line, line.isdigit())
      if line.isdigit():
        image_id = int(line)
      else:
        image_caption_pairs.append((image_id, line))


    if FLAGS.parse_method == "thulac":
      thu1 = thulac.thulac(seg_only=True)
      data = [word for (_, sentence) in image_caption_pairs for word in _read_words(sentence, thu1)]
    elif FLAGS.parse_method == "jieba":
      data = [word for (_, sentence) in image_caption_pairs for word in _read_words_with_jieba(sentence)]  
    else:
      data = [word for (_, sentence) in image_caption_pairs for word in _read_words(sentence)]

    counter = collections.Counter(data)
    # print(counter)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    # print(count_pairs)

    # Remove Low-frequency word
    count_pairs = [(a,b) for (a,b) in count_pairs if b > 1]

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(1,1+len(words))))

    # add zero, begin, end, unknown words
    word_to_id['<UKW>'] = len(word_to_id)+1

    # Write out the word id file.
    with tf.gfile.FastGFile(FLAGS.word_id_output_file, "w") as f:
      f.write("\n".join(["%s %d" % (w, c) for (w, c) in word_to_id.items()]))
    print("Wrote vocabulary file:", FLAGS.word_id_output_file)
    # Write out the word count file.
    with tf.gfile.FastGFile(FLAGS.word_count_output_file, "w") as f:
      f.write("\n".join(["%s %d" % (w.encode('utf-8'), c) for (w, c) in count_pairs]))
    print("Wrote vocabulary file:", FLAGS.word_count_output_file)

    return word_to_id


if __name__ == '__main__':
  _build_vocab(os.path.join(FLAGS.data_path, "train.txt"))
