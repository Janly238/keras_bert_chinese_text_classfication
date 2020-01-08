import tensorflow as tf
import pandas as pd
import tensorflow_hub as hub
import os
import re
import numpy as np
from bert.tokenization import FullTokenizer
from tqdm import tqdm_notebook
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
from FM import *






os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# Initialize session
sess = tf.Session()
os.environ["TFHUB_CACHE_DIR"] = '../tf-hub'
# Params for bert model and tokenization
bert_path = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
max_seq_length = 256


# Load all files from a directory in a DataFrame.
def load_directory_data(directory):
    data = {}
    data["sentence"] = []
    data["sentiment"] = []
    for file_path in os.listdir(directory):
        with tf.gfile.GFile(os.path.join(directory, file_path), "r") as f:
            data["sentence"].append(f.read())
            data["sentiment"].append(re.match("\d+_(\d+)\.txt", file_path).group(1))
    return pd.DataFrame.from_dict(data)


# Merge positive and negative examples, add a polarity column and shuffle.
def load_dataset(directory):
    pos_df = load_directory_data(os.path.join(directory, "pos"))
    neg_df = load_directory_data(os.path.join(directory, "neg"))
    pos_df["polarity"] = 1
    neg_df["polarity"] = 0
    return pd.concat([pos_df, neg_df]).sample(frac=1).reset_index(drop=True)


# Download and process the dataset files.
def download_and_load_datasets(force_download=False):
    dataset = tf.keras.utils.get_file(
        fname="aclImdb.tar.gz",
        origin="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
        extract=True)

    train_df = load_dataset(os.path.join(os.path.dirname(dataset),
                                         "aclImdb", "train"))
    test_df = load_dataset(os.path.join(os.path.dirname(dataset),
                                        "aclImdb", "test"))

    return train_df, test_df


# Reduce logging output.
tf.logging.set_verbosity(tf.logging.ERROR)

train_df, test_df = download_and_load_datasets()
train_df.head()

# Create datasets (Only take up to max_seq_length words for memory)
train_text = train_df['sentence'].tolist()
train_text = [' '.join(t.split()[0:max_seq_length]) for t in train_text]
train_text = np.array(train_text, dtype=object)[:, np.newaxis]
train_label = train_df['polarity'].tolist()

test_text = test_df['sentence'].tolist()
test_text = [' '.join(t.split()[0:max_seq_length]) for t in test_text]
test_text = np.array(test_text, dtype=object)[:, np.newaxis]
test_label = test_df['polarity'].tolist()


class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.
  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size, because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.
  We use this class instead of `None` because treating `None` as padding
  battches could cause silent errors.
  """


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


def create_tokenizer_from_hub_module():
    """Get the vocab file and casing info from the Hub module."""
    bert_module = hub.Module(bert_path)
    tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
    vocab_file, do_lower_case = sess.run(
        [
            tokenization_info["vocab_file"],
            tokenization_info["do_lower_case"],
        ]
    )

    return FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)


def convert_single_example(tokenizer, example, max_seq_length=256):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    if isinstance(example, PaddingInputExample):
        input_ids = [0] * max_seq_length
        input_mask = [0] * max_seq_length
        segment_ids = [0] * max_seq_length
        label = 0
        return input_ids, input_mask, segment_ids, label

    tokens_a = tokenizer.tokenize(example.text_a)
    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[0: (max_seq_length - 2)]

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    return input_ids, input_mask, segment_ids, example.label


def convert_examples_to_features(tokenizer, examples, max_seq_length=256):
    """Convert a set of `InputExample`s to a list of `InputFeatures`."""

    input_ids, input_masks, segment_ids, labels = [], [], [], []
    for example in tqdm_notebook(examples, desc="Converting examples to features"):
        input_id, input_mask, segment_id, label = convert_single_example(
            tokenizer, example, max_seq_length
        )
        input_ids.append(input_id)
        input_masks.append(input_mask)
        segment_ids.append(segment_id)
        labels.append(label)
    return (
        np.array(input_ids),
        np.array(input_masks),
        np.array(segment_ids),
        np.array(labels).reshape(-1, 1),
    )


def convert_text_to_examples(texts, labels):
    """Create InputExamples"""
    InputExamples = []
    for text, label in zip(texts, labels):
        InputExamples.append(
            InputExample(guid=None, text_a=" ".join(text), text_b=None, label=label)
        )
    return InputExamples


# Instantiate tokenizer
tokenizer = create_tokenizer_from_hub_module()

# Convert data to InputExample format
# train_text [['aa','bb','cc','dd'], ['ee','ff','gg','hh']]
# train_label = [4.3,2.2,...,5.0]
# 这里是输入数据

import pickle as p

df = p.load(open('../data/exp_all_data.pkl', 'rb'))
y = df.review_rating
x = []
num = []
for i,j in zip(df.review,df.description):
    x.append(str(i)+str(j))

for i, j in zip(df.numDownloads, df.genre):
    i = list(i)
    j = list(j)
    num.append(i + j)

num = np.array(num)
print(num.shape)
train_text, test_text, y_train, y_test = train_test_split(x[0:20000], y[0:20000], test_size=0.3, random_state=2019)
train_input_fm, test_input_fm, y_train_fm, y_test_fm = train_test_split(num[0:20000], y[0:20000], test_size=0.3,
                                                                        random_state=2019)

train_examples = convert_text_to_examples(train_text, y_train)
test_examples = convert_text_to_examples(test_text, y_test)

# Convert to features
(train_input_ids, train_input_masks, train_segment_ids, train_labels
 ) = convert_examples_to_features(tokenizer, train_examples, max_seq_length=max_seq_length)
(test_input_ids, test_input_masks, test_segment_ids, test_labels
 ) = convert_examples_to_features(tokenizer, test_examples, max_seq_length=max_seq_length)


class BertLayer(tf.layers.Layer):
    def __init__(self, n_fine_tune_layers=10, **kwargs):
        self.n_fine_tune_layers = n_fine_tune_layers
        self.trainable = True
        self.output_size = 768
        super(BertLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.bert = hub.Module(
            bert_path,
            trainable=self.trainable,
            name="{}_module".format(self.name)
        )

        trainable_vars = self.bert.variables

        # Remove unused layers
        trainable_vars = [var for var in trainable_vars if not "/cls/" in var.name]

        # Select how many layers to fine tune
        trainable_vars = trainable_vars[-self.n_fine_tune_layers:]

        # Add to trainable weights
        for var in trainable_vars:
            self._trainable_weights.append(var)

        for var in self.bert.variables:
            if var not in self._trainable_weights:
                self._non_trainable_weights.append(var)

        super(BertLayer, self).build(input_shape)

    def call(self, inputs):
        inputs = [K.cast(x, dtype="int32") for x in inputs]
        input_ids, input_mask, segment_ids = inputs
        bert_inputs = dict(
            input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids
        )
        result = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)[
            "pooled_output"
        ]
        return result

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_size)


# Build model
def build_model(max_seq_length, feature_dim):
    in_id = tf.keras.layers.Input(shape=(max_seq_length,), name="input_ids")
    in_mask = tf.keras.layers.Input(shape=(max_seq_length,), name="input_masks")
    in_segment = tf.keras.layers.Input(shape=(max_seq_length,), name="segment_ids")
    inputs_fm = tf.keras.Input(shape=(feature_dim,), name='fm_input')

    bert_inputs = [in_id, in_mask, in_segment]
    bert_output = BertLayer(n_fine_tune_layers=3)(bert_inputs)
    dense = tf.keras.layers.Dense(256, activation='relu')(bert_output)
    pred = tf.keras.layers.Dense(1, activation='sigmoid')(dense)
    pred_bert = tf.keras.layers.Dense(1)(pred)

    liner = tf.keras.layers.Dense(units=1,
                                  bias_regularizer=tf.keras.regularizers.l2(0.01),
                                  kernel_regularizer=tf.keras.regularizers.l1(0.02),
                                  )(inputs_fm)
    cross = MyLayer(feature_dim)(inputs_fm)

    add = tf.keras.layers.Add()([liner, cross])
    pred_fm = tf.keras.layers.Activation('sigmoid')(add)
    pred_fm = tf.keras.layers.Dense(1)(pred_fm)

    pred_all = tf.keras.layers.Add(name='pred_all')([pred_bert, pred_fm])
    print('333')

    def rmse(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

    model = tf.keras.models.Model(inputs=[in_id, in_mask, in_segment, inputs_fm], outputs=pred_all)
    model.compile(loss=rmse, optimizer='sgd', metrics=[rmse, 'mae'])
    # model.summary()
    return model


def initialize_vars(sess):
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    K.set_session(sess)


model = build_model(max_seq_length, 65)

# Instantiate variables
initialize_vars(sess)

model.fit(x={'input_ids': train_input_ids,
             'input_masks': train_input_masks,
             'segment_ids': train_segment_ids,
             'fm_input': train_input_fm},
          y={'pred_all': train_labels},
          validation_data=({'input_ids': test_input_ids[0:100],
                            'input_masks': test_input_masks[0:100],
                            'segment_ids': test_segment_ids[0:100],
                            'fm_input': test_input_fm[0:100]}, {'pred_all': test_labels[0:100]}),
          epochs=10,
          batch_size=128
          )

model.save('BertModel.h5')
model.predict(x={'input_ids': test_input_ids,
                 'input_masks': test_input_masks,
                 'segment_ids': test_segment_ids,
                 'fm_input': test_input_fm}
              )  # predictions after we clear and reload model

# # Clear and load model
# model = None
# model = build_model(max_seq_length, 65)
# initialize_vars(sess)
# model.load_weights('BertModel.h5')
#
# model.predict(x={'input_ids': test_input_ids,
#                                   'input_masks': test_input_masks,
#                                   'segment_ids': test_segment_ids,
#                                   'fm_input': test_input_fm}
#                                 )  # predictions after we clear and reload model
