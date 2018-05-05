import os
import sys
import time
from datetime import timedelta
import tensorflow.contrib.keras as kr
import numpy as np
import tensorflow as tf
from sklearn import metrics
import pickle
import argparse
from classification_preprocess import cat_to_id
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file


save_dir = './checkpoints/textcnn'
save_path = os.path.join(save_dir, 'best_validation')
param_saving_path = '../data/param-classify.dat'
tensorboard_dir = './tensorboard/textcnn'
validation_rate = 0.1

class TCNNConfig(object):
    """CNN param"""
    embedding_dim = 64  # word vector dimension
    seq_length = 800  # sequense length
    num_classes = 3  # class number
    num_filters = 256  # kernel number
    kernel_size = 5  # kernel size
    vocab_size = 5000  # vocab size

    hidden_dim = 128  # fully connected neuro number

    dropout_keep_prob = 0.5  # dropout keeping rate
    learning_rate = 1e-3  # learning rate

    batch_size = 64  # batch size
    num_epochs = 10  # total epoch number

    print_per_batch = 10  # output iterations
    save_per_batch = 10  # save tensorboard iterations


class TextCNN(object):
    """text classification，CNN model"""

    def __init__(self, config):
        self.config = config

        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.cnn()

    def cnn(self):
        """CNN model"""
        # word embedding
        with tf.device('/cpu:0'):
            embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim])
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)

        with tf.name_scope("cnn"):
            # CNN layer
            conv = tf.layers.conv1d(embedding_inputs, self.config.num_filters, self.config.kernel_size, name='conv')
            # global max pooling layer
            gmp = tf.reduce_max(conv, reduction_indices=[1], name='gmp')

        with tf.name_scope("score"):
            # fully connected layer，with dropout and ReLU
            fc = tf.layers.dense(gmp, self.config.hidden_dim, name='fc1')
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)

            # classifier
            self.logits = tf.layers.dense(fc, self.config.num_classes, name='fc2')
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)  # predictor

        with tf.name_scope("optimize"):
            # loss function，cross entropy
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
            # optimizor
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # accuracy
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

def get_time_dif(start_time):
    """get time"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

def batch_iter(x, y, batch_size=64):
    """generate batchsize data"""
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]

def feed_data(model, x_batch, y_batch, keep_prob):
    feed_dict = {
        model.input_x: x_batch,
        model.input_y: y_batch,
        model.keep_prob: keep_prob
    }
    return feed_dict


def evaluate(model, sess, x_, y_):
    """evaluate the loss and accuracy"""
    data_len = len(x_)
    batch_eval = batch_iter(x_, y_, 128)
    total_loss = 0.0
    total_acc = 0.0
    for x_batch, y_batch in batch_eval:
        batch_len = len(x_batch)
        feed_dict = feed_data(model, x_batch, y_batch, 1.0)
        loss, acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)
        total_loss += loss * batch_len
        total_acc += acc * batch_len

    return total_loss / data_len, total_acc / data_len

def train(filename):
    config = TCNNConfig()
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    x = data['X']
    y = data['Y']
    print(len(x))
    P = np.random.permutation(len(x))
    x = x[P]
    y = y[P]

    wordToID = data['wordToID']
    seq_length = data['seq_length']
    config.vocab_size = len(wordToID)
    config.seq_length = seq_length

    model = TextCNN(config)

    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("accuracy", model.acc)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)
    
    
    saver = tf.train.Saver()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    idx = int(x.shape[0] * validation_rate)
    x_train = x[idx:]
    x_val = x[:idx]
    y_train = y[idx:]
    y_val = y[:idx]
    
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    writer.add_graph(session.graph)
    
    print('Training and evaluating...')
    start_time = time.time()
    total_batch = 0  # total batch number
    best_acc_val = 0.0  # best validation accuracy
    last_improved = 0  # last improving
    require_improvement = 1000  # if not improving after 1000 iterations, end early
    
    flag = False
    for epoch in range(config.num_epochs):
        print('Epoch:', epoch + 1)
        batch_train = batch_iter(x_train, y_train, config.batch_size)
        for x_batch, y_batch in batch_train:
            feed_dict = feed_data(model, x_batch, y_batch, config.dropout_keep_prob)
            
            if total_batch % config.save_per_batch == 0:
                # save to tensorboard scalar
                s = session.run(merged_summary, feed_dict=feed_dict)
                writer.add_summary(s, total_batch)


            if total_batch % config.print_per_batch == 0:
                # get the loss and accuracy on training set and validation set
                feed_dict[model.keep_prob] = 1.0
                loss_train, acc_train = session.run([model.loss, model.acc], feed_dict=feed_dict)
                loss_val, acc_val = evaluate(model, session, x_val, y_val)  # todo

                if acc_val > best_acc_val:
                    # save the best result
                    best_acc_val = acc_val
                    last_improved = total_batch
                    saver.save(sess=session, save_path=save_path)
                    print("Save model!")
                    improved_str = '*'
                else:
                    improved_str = ''

                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6}, Train Loss: {1:>4.4}, Train Acc: {2:>5.2%},' \
                      + ' Val Loss: {3:>4.4}, Val Acc: {4:>5.2%}, Time: {5} {6}'
                print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, time_dif, improved_str))

            session.run(model.optim, feed_dict=feed_dict)  # 运行优化
            total_batch += 1

            if total_batch - last_improved > require_improvement:
                # early end
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break 
        if flag:
            break


def test(text, filename, genre, model_path=save_dir):

    config = TCNNConfig()
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    wordToID = data['wordToID']
    seq_length = data['seq_length']
    config.vocab_size = len(wordToID)
    config.seq_length = seq_length

    model = TextCNN(config)

    text_ids = [[wordToID[word] for word in text.split(" ") if word in wordToID]]
    # print(text_ids)
    y = np.array([cat_to_id[genre]])

    x_pad = kr.preprocessing.sequence.pad_sequences(text_ids, seq_length)
    y_pad = kr.utils.to_categorical(y, num_classes=len(cat_to_id)) 

    with tf.Session() as session:

    # session = tf.Session()
        session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        # saver.restore(sess=session, save_path=save_path) 

        checkPoint = tf.train.get_checkpoint_state(model_path)
        # if have checkPoint, restore checkPoint
        if checkPoint and checkPoint.model_checkpoint_path:
            # print_tensors_in_checkpoint_file(file_name=checkPoint.model_checkpoint_path, 
            #     tensor_name='',
            #     all_tensors=False,
            #     all_tensor_names=True)
            print(checkPoint.model_checkpoint_path)
            # print([n.name for n in tf.get_default_graph().as_graph_def().node])

            saver.restore(session, checkPoint.model_checkpoint_path)
            print("restored %s" % checkPoint.model_checkpoint_path)
            print("\n\n")
        else:
            print("no checkpoint found!")
            exit(0)

        

        print('Testing...')

        feed_dict = feed_data(model, x_pad, y_pad, 1.0)
        y_pred = session.run(model.y_pred_cls, feed_dict=feed_dict)
        return list(cat_to_id)[y_pred[0]]

# text = "i wonder to where darkness . where i want . i want to know my good . feeling like you . i remember the girl . and you and i . only the pain of that my love . a memory . and that keep me enough . . im always in vain . the gypsy burning and supper . hide behind my feeling . let spend the night in bloom we will play . i just breathe , breathe me . dont see me cry again "
# print(test(text, "../data/param-classify-test.dat", "rock"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", help = "select mode by 'train' or test",
        choices = ["train", "test"], default = "test")
    
    args = parser.parse_args()

    filename = "./generate-param/param-classify-test.dat"

    if args.mode == 'test':
        text = "i wonder to where darkness . where i want . i want to know my good . feeling like you . i remember the girl . and you and i . only the pain of that my love . a memory . and that keep me enough . . im always in vain . the gypsy burning and supper . hide behind my feeling . let spend the night in bloom we will play . i just breathe , breathe me . dont see me cry again "
        print(test(text,filename , "rock"))
    else:
        train(filename)

if __name__ == "__main__":
    main()
