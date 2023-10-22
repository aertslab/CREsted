import os
import sys
from random import choice
import pickle
import heapq
import weblogo
import subprocess
import operator
from collections import OrderedDict
import matplotlib
import model_zoo as mz
import numpy as np
import pandas as pd
import pybedtools
from pybedtools import BedTool, Interval
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from matplotlib import image as mpimg
import tensorflow as tf
#tf.enable_eager_execution()
from tensorflow.keras import layers, Model, Input
#import tensorflow_probability as tfp
from tqdm import tqdm
import shap
import re
matplotlib.use('pdf')


def set_pybedtool_temp(foldername):
    pybedtools.set_tempdir(foldername)


def set_seed(seed):
    np.random.seed(seed)


def get_genome_sizes_and_fasta_filename(genome_file):
    return 'resources/{0:s}.fa'.format(genome_file), 'resources/{0:s}.chrom.sizes'.format(genome_file)


def get_selected_topic_from_text(selected_topic_textfile):
    with open(selected_topic_textfile, 'r') as rf:
        for line in rf:
            return np.array([int(nums) for nums in line.strip().strip(',').split(',')]) - 1


def get_merged_topic_from_text(merge_topic_textfile):
    merge_topic_list = []
    with open(merge_topic_textfile, 'r') as rf:
        for line in rf:
            merge_topic_list.append(np.array([int(nums) for nums in line.strip().strip(',').split(',')]) - 1)
    return merge_topic_list


def one_hot_encode_along_row_axis(sequence):
    to_return = np.zeros((1, len(sequence), 4), dtype=np.int8)
    seq_to_one_hot_fill_in_array(zeros_array=to_return[0], sequence=sequence, one_hot_axis=1)
    return to_return


def seq_to_one_hot_fill_in_array(zeros_array, sequence, one_hot_axis):
    assert one_hot_axis == 0 or one_hot_axis == 1
    if one_hot_axis == 0:
        assert zeros_array.shape[1] == len(sequence)
    elif one_hot_axis == 1:
        assert zeros_array.shape[0] == len(sequence)
    for (i, char) in enumerate(sequence):
        if char == "A" or char == "a":
            char_idx = 0
        elif char == "C" or char == "c":
            char_idx = 1
        elif char == "G" or char == "g":
            char_idx = 2
        elif char == "T" or char == "t":
            char_idx = 3
        elif char == "N" or char == "n":
            continue
        else:
            raise RuntimeError("Unsupported character: " + str(char))
        if one_hot_axis == 0:
            zeros_array[char_idx, i] = 1
        elif one_hot_axis == 1:
            zeros_array[i, char_idx] = 1


def readfile_wolabel(filename):
    ids = []
    ids_d = {}
    seqs = {}
    f = open(filename, 'r')
    lines = f.readlines()
    f.close()
    seq = []
    for line in lines:
        if line[0] == '>':
            ids.append(line[1:].rstrip('\n'))
            id_line = line[1:].rstrip('\n').split('_')[0]
            if id_line not in seqs:
                seqs[id_line] = []
            if id_line not in ids_d:
                ids_d[id_line] = id_line
            if seq:
                seqs[ids[-2].split('_')[0]] = ("".join(seq))
            seq = []
        else:
            seq.append(line.rstrip('\n').upper())
    if seq:
        seqs[ids[-1].split('_')[0]] = ("".join(seq))

    return ids, ids_d, seqs


def readfile_withlabel(filename, num_topics):
    ids = []
    ids_d = {}
    seqs = {}
    classes = {}
    f = open(filename, 'r')
    lines = f.readlines()
    f.close()
    seq = []
    for line in lines:
        if line[0] == '>':
            ids.append(line[1:].rstrip('\n'))
            if line[1:].rstrip('\n').split('_')[0] not in seqs:
                seqs[line[1:].rstrip('\n').split('_')[0]] = []
            if line[1:].rstrip('\n').split('_')[0] not in ids_d:
                ids_d[line[1:].rstrip('\n').split('_')[0]] = line[1:].rstrip('\n').split('_')[0]
            if line[1:].rstrip('\n').split('_')[0] not in classes:
                classes[line[1:].rstrip('\n').split('_')[0]] = np.zeros(num_topics)

            classes[line[1:].rstrip('\n').split('_')[0]][int(line[1:].rstrip('\n').split('_')[1]) - 1] = 1
            if seq != []: seqs[ids[-2].split('_')[0]] = ("".join(seq))
            seq = []
        else:
            seq.append(line.rstrip('\n').upper())
    if seq != []:
        seqs[ids[-1].split('_')[0]] = ("".join(seq))

    return ids, ids_d, seqs, classes


def prepare_data_from_summits(filename, L, stride=10, augment=False, isfilepath=True, chr_size_file=None):
    if isfilepath:
        bedfile = BedTool(filename)
    else:
        bedfile = filename
    if augment:
        centered_bed = centerization(size=L + 200, name='', input_bed=bedfile, chr_size_file=chr_size_file)
        augmented_bed = augmentation(size=L, stride=stride, name='', input_bed=centered_bed)
        return augmented_bed
    else:
        centered_bed = centerization(size=L, name='', input_bed=bedfile, chr_size_file=chr_size_file)
        return centered_bed


def prepare_data_from_topics(output_dir, foldername, augmented_bed, genome, name=""):
    #augmented_bed.saveas(os.path.join(output_dir, name))
    # topic_list = []
    # for i in os.listdir(foldername):
    #     i_ = os.path.join(foldername, i)
    #     if os.path.isfile(i_) and i_.endswith(".bed") and not(i_.startswith(".")):
    #         topic_list.append(i_)
    command = "{shfile} {output_name} {topic_folder} {summit} {genome}"
    bashCommand = command.format(shfile="/staging/leuven/stg_00002/lcb/itask/programs/summit_topic_intersection.sh",
                                 output_name=os.path.join(output_dir, "summit_to_topic_"+name),
                                 topic_folder=foldername,
                                 summit=os.path.join(output_dir, name),
                                 genome=genome)
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    _output, _error = process.communicate()
    # a=[]
    # for file_to_read in topic_list:
    #     a.append(BedTool(file_to_read).fn)
    # int_bed = augmented_bed.intersect(i=a,
    #                                   wa=True, wb=True, f=0.6, names=names, )
    # for region in int_bed:
    #     print(region)
    # # for region in int_bed:
    # #     print(Interval(region.chrom, region.start, region.end,
    # #                               name=region.chrom + ":" + str(region.start) + "-" + str(
    # #                                   region.end) + "_" + name),file=fr)
    # output_bed = BedTool(os.path.join(output_dir, "summit_to_topic_"+name))
    if name.startswith("nonAugmented"):
        output_bed = BedTool(os.path.join(output_dir, "summit_to_topic_" + name))
        return output_bed
    else:
        return 0


def valid_test_split_wrapper(bed, valid_chroms, test_chroms):
    bed_train = []
    bed_valid = []
    bed_test = []
    if valid_chroms[0].startswith("chr"):
        for interval in bed:
            chrom = interval.chrom
            if chrom in test_chroms:
                bed_test.append(interval)
            elif chrom in valid_chroms:
                bed_valid.append(interval)
            else:
                bed_train.append(interval)
    else:
        train_indeces, rest_indices = train_test_split(
            np.arange(len(bed)), test_size=(int(valid_chroms[0])+int(test_chroms[0]))/100, shuffle=True)
        valid_indeces, test_indices = train_test_split(
            rest_indices, test_size=int(test_chroms[0])/(int(valid_chroms[0])+int(test_chroms[0])), shuffle=True)
        for index, interval in enumerate(bed):
            if index in test_indices:
                bed_test.append(interval)
            elif index in valid_indeces:
                bed_valid.append(interval)
            else:
                bed_train.append(interval)
    bed_train = BedTool(bed_train)
    bed_valid = BedTool(bed_valid)
    bed_test = BedTool(bed_test)
    return bed_train, bed_valid, bed_test


def prepare_CV_file(outputdirc, inputtopics, consensuspeaks, seqlen, stride, genome, chrSize, selected_topics,
                    numTopics, mergetopic):
    centered_bed = prepare_data_from_summits(
        consensuspeaks, seqlen, stride=stride, augment=False, chr_size_file=chrSize)

    kf = KFold(n_splits=10, shuffle=True)
    indexes = np.arange(len(centered_bed))
    kf.get_n_splits(indexes)
    indexes_10_fold = []
    for train_index, test_index in kf.split(indexes):
        indexes_10_fold.append(test_index)
    bed_10_fold = [[] for _i in range(10)]
    for index, interval in enumerate(centered_bed):
        for i in range(10):
            if index in indexes_10_fold[i]:
                bed_10_fold[i].append(interval)

    centered_bed_withName = [[] for _i in range(10)]
    augmented_bed = [[] for _i in range(10)]
    data_aug = [[] for _i in range(10)]
    y_aug = [[] for _i in range(10)]
    ids_aug = [[] for _i in range(10)]
    data_naug = [[] for _i in range(10)]
    y_naug = [[] for _i in range(10)]
    ids_naug = [[] for _i in range(10)]

    for i in range(10):
        bed_10_fold[i] = BedTool(bed_10_fold[i])
        centered_bed_withName[i] = prepare_data_from_topics(outputdirc, inputtopics, bed_10_fold[i], genome,
                                                            name="nonAugmented_fold" + str(i + 1) + ".bed")
        printFasta(genome, centered_bed_withName[i],
                   os.path.join(outputdirc, 'nonAugmented_fold' + str(i + 1) + '.fa'))
        augmented_bed[i] = prepare_data_from_summits(centered_bed_withName[i], seqlen, stride=stride,
                                                     augment=True, isfilepath=False, chr_size_file=chrSize)
        _ = prepare_data_from_topics(outputdirc, inputtopics, augmented_bed[i], genome,
                                     name="augmented_fold" + str(i + 1) + ".bed")
        data_aug[i], y_aug[i], ids_aug[i] = prepare_data_withlabel(
            os.path.join(outputdirc, 'summit_to_topic_augmented_fold' + str(i + 1) + '.fa'), selected_topics,
            numTopics, mergetopic)
        data_naug[i], y_naug[i], ids_naug[i] = prepare_data_withlabel(
            os.path.join(outputdirc, 'summit_to_topic_nonAugmented_fold' + str(i + 1) + '.fa'), selected_topics,
            numTopics, mergetopic)

    augmented_data_dict = {"data": data_aug, "y": y_aug, "ids": ids_aug}
    f = open(os.path.join(outputdirc, 'augmented_data_dict_10fold.pkl'), "wb")
    pickle.dump(augmented_data_dict, f)
    f.close()

    for fold in range(9):
        train_data = np.concatenate([x for i, x in enumerate(data_aug[0:9]) if i != fold])
        y_train = np.concatenate([x for i, x in enumerate(y_aug[0:9]) if i != fold])
        ids_train = np.concatenate([x for i, x in enumerate(ids_aug[0:9]) if i != fold])
        valid_data = data_aug[fold]
        y_valid = y_aug[fold]
        ids_valid = ids_aug[fold]
        test_data = data_aug[9]
        y_test = y_aug[9]
        ids_test = ids_aug[9]
        augmented_data_dict = {"train_data": train_data, "y_train": y_train, "ids_train": ids_train,
                               "valid_data": valid_data, "y_valid": y_valid, "ids_valid": ids_valid,
                               "test_data": test_data, "y_test": y_test, "ids_test": ids_test}
        f = open(os.path.join(outputdirc, 'augmented_data_dict_fold' + str(fold + 1) + '.pkl'), "wb")
        pickle.dump(augmented_data_dict, f)

    non_augmented_data_dict = {"data": data_naug, "y": y_naug, "ids":ids_naug}
    f = open(os.path.join(outputdirc, 'nonAugmented_data_dict_10fold.pkl'), "wb")
    pickle.dump(non_augmented_data_dict, f)
    f.close()

    for fold in range(9):
        train_data = np.concatenate([x for i, x in enumerate(data_naug[0:9]) if i != fold])
        y_train = np.concatenate([x for i, x in enumerate(y_naug[0:9]) if i != fold])
        ids_train = np.concatenate([x for i, x in enumerate(ids_naug[0:9]) if i != fold])
        valid_data = data_naug[fold]
        y_valid = y_naug[fold]
        ids_valid = ids_naug[fold]
        test_data = data_naug[9]
        y_test = y_naug[9]
        ids_test = ids_naug[9]
        nonAugmented_data = np.concatenate(data_naug)
        nonAugmented_y = np.concatenate(y_naug)
        nonAugmented_ids = np.concatenate(ids_naug)
        nonAugmented_data_dict = {"nonAugmented_data": nonAugmented_data, "nonAugmented_y": nonAugmented_y,
                                  "nonAugmented_ids": nonAugmented_ids,
                                  "train_data": train_data, "y_train": y_train, "ids_train": ids_train,
                                  "valid_data": valid_data, "y_valid": y_valid, "ids_valid": ids_valid,
                                  "test_data": test_data, "y_test": y_test, "ids_test": ids_test}
        f = open(os.path.join(outputdirc, 'nonAugmented_data_dict_fold' + str(fold + 1) + '.pkl'), "wb")
        pickle.dump(nonAugmented_data_dict, f)


def prepare_data_withlabel(filename, selected_topics, num_topics, mergetopic):
    ids, ids_d, seqs, classes = readfile_withlabel(filename, num_topics)
    X = np.array([one_hot_encode_along_row_axis(seqs[id_]) for id_ in ids_d]).squeeze(axis=1)
    y = np.array([classes[id_] for id_ in ids_d])
    ids_ = np.array([id_ for id_ in ids_d])
    if mergetopic != "False":
        new_y = np.zeros((y.shape[0], len(mergetopic) + 1))
        for i in range(len(mergetopic)):
            result = y[:, mergetopic[i][0]]
            for j in range(len(mergetopic[i])):
                result = np.logical_or(result, y[:, mergetopic[i][j]])
            new_y[:, i] = result
        new_y[:, -1][np.sum(new_y, axis=1) == 0] = 1
        X = X[new_y.sum(axis=1) > 0]
        ids_ = ids_[new_y.sum(axis=1) > 0]
        new_y = new_y[new_y.sum(axis=1) > 0]
        # X_rc = X[:, ::-1, ::-1]
        # data = [X, X_rc]
        data = X
        return data, new_y, ids_
    else:
        y = y[:, selected_topics]
        X = X[y.sum(axis=1) > 0]
        ids_ = ids_[y.sum(axis=1) > 0]
        y = y[y.sum(axis=1) > 0]
        # X_rc = X[:, ::-1, ::-1]
        # data = [X, X_rc]
        data = X
        return data, y, ids_


def prepare_data_wolabel(filename):
    ids, ids_d, seqs, = readfile_wolabel(filename)
    X = np.array([one_hot_encode_along_row_axis(seqs[id_]) for id_ in ids_d]).squeeze(axis=1)
    # X_rc = X[:, ::-1, ::-1]
    # data = [X, X_rc]
    data = X
    return data, ids


def prepare_data_wolabel_difsize(filename):
    ids, ids_d, seqs = readfile_wolabel(filename)
    X = ([one_hot_encode_along_row_axis(seqs[id]) for id in ids_d])
    ids = np.array([id for id in ids_d])
    result = {'ids': ids, 'X': X}
    return result

def get_sample_weights(output_dir, regs, logfc_df):
    print('Calculating sample weights...')
    sample_weights = np.zeros(len(regs))
    df = pd.read_pickle(logfc_df)
    for i, reg in tqdm(enumerate(regs)):
        if(i%5==0):
            chrom = reg.split(':')[0]
            start = int(reg.split(':')[1].split('-')[0])
            end = int(reg.split('-')[1])
            start += 100
            end -= 100
            reg_id=chrom+':'+str(start)+'-'+str(end)
            if(len(df.loc[reg_id])==1):
                res = df.loc[reg_id]['Log2FC']
            else:  
                res = np.mean(df.loc[reg_id]['Log2FC'].values)
            sample_weights[i:i+5] = res
    return sample_weights
        


def create_plots(history, output_dir):
    plt.plot(history.history['auPR'])
    plt.plot(history.history['val_auPR'])
    plt.title('model area under PR')
    plt.ylabel('auPR')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(os.path.join(output_dir, 'accuracy.png'))
    plt.clf()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(os.path.join(output_dir, 'loss.png'))
    plt.clf()


reverse_lambda_ax2 = layers.Lambda(lambda x: tf.reverse(x, [2]))
reverse_lambda_ax1 = layers.Lambda(lambda x: tf.reverse(x, [1]))


def get_output(input_layer, hidden_layers):
    output = input_layer
    for hidden_layer in hidden_layers:
        output = hidden_layer(output)
    return output


def get_additive_output(input_layer, hidden_layers):
    output = input_layer
    for hidden_layer in hidden_layers:
        output = hidden_layer(output)
    return layers.add([input_layer, output])

from tensorflow.keras import backend as K

# https://github.com/WenYanger/Keras_Metrics/blob/master/PearsonCorr.py
def pearson_r(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = K.mean(x, axis=0)
    my = K.mean(y, axis=0)
    xm, ym = x - mx, y - my
    r_num = K.sum(xm * ym)
    x_square_sum = K.sum(xm * xm)
    y_square_sum = K.sum(ym * ym)
    r_den = K.sqrt(x_square_sum * y_square_sum)
    r = r_num / r_den
    return K.mean(r)

import tensorflow as tf

class PearsonCorrelation(tf.keras.metrics.Metric):
    def __init__(self, name='pearson_correlation', **kwargs):
        super(PearsonCorrelation, self).__init__(name=name, **kwargs)
        self.y_true_sum = self.add_weight(name='y_true_sum', initializer='zeros')
        self.y_pred_sum = self.add_weight(name='y_pred_sum', initializer='zeros')
        self.y_true_squared_sum = self.add_weight(name='y_true_squared_sum', initializer='zeros')
        self.y_pred_squared_sum = self.add_weight(name='y_pred_squared_sum', initializer='zeros')
        self.y_true_y_pred_sum = self.add_weight(name='y_true_y_pred_sum', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        self.y_true_sum.assign_add(tf.reduce_sum(y_true))
        self.y_pred_sum.assign_add(tf.reduce_sum(y_pred))
        self.y_true_squared_sum.assign_add(tf.reduce_sum(tf.square(y_true)))
        self.y_pred_squared_sum.assign_add(tf.reduce_sum(tf.square(y_pred)))
        self.y_true_y_pred_sum.assign_add(tf.reduce_sum(y_true * y_pred))
        self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))

    def result(self):
        mean_true = self.y_true_sum / self.count
        mean_pred = self.y_pred_sum / self.count

        numerator = self.count * self.y_true_y_pred_sum - self.y_true_sum * self.y_pred_sum
        denominator = tf.sqrt((self.count * self.y_true_squared_sum - tf.square(self.y_true_sum)) * 
                              (self.count * self.y_pred_squared_sum - tf.square(self.y_pred_sum)))

        return numerator / (denominator + tf.keras.backend.epsilon())

    def reset_states(self):
        self.y_true_sum.assign(0.0)
        self.y_pred_sum.assign(0.0)
        self.y_true_squared_sum.assign(0.0)
        self.y_pred_squared_sum.assign(0.0)
        self.y_true_y_pred_sum.assign(0.0)
        self.count.assign(0.0)



#from https://github.com/kundajelab/basepair/blob/cda0875571066343cdf90aed031f7c51714d991a/basepair/losses.py#L87
def multinomial_nll(true_counts, logits):
    """Compute the multinomial negative log-likelihood
    Args:
      true_counts: observed count values
      logits: predicted logit values
    """
    counts_per_example = tf.reduce_sum(true_counts, axis=-1)
    #dist = tfp.distributions.Multinomial(total_count=counts_per_example, logits=logits)
    dist = tf.raw_ops.Multinomial(num_samples=counts_per_example, logits=logits)
    return (-tf.reduce_sum(dist.log_prob(true_counts)) / 
            tf.cast(tf.shape(true_counts)[0], dtype=tf.float32))

from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend as K
def custom_loss(y_true, y_pred):
    y_true1 = nn.l2_normalize(y_true, axis=-1)
    y_pred1 = nn.l2_normalize(y_pred, axis=-1)
    y_pred2 = ops.convert_to_tensor_v2_with_dispatch(y_pred)
    y_true2 = math_ops.cast(y_true, y_pred.dtype)
    return -math_ops.reduce_sum(y_true1 * y_pred1, axis=-1) + K.mean(math_ops.squared_difference(y_pred2, y_true2), axis=-1)


def build_model(num_classes, filter_size=17, num_filters=1024, pool_size=4, num_dense=1024, activation='gelu',
                learningrate=1e-3, seq_shape=(500,4), insertmotif="False", conv_l2=1e-6, dense_l2=1e-3, conv_do=0.15, dense_do=0.5, 
                use_transformer = False, model_type="main", main_model=False, y = None):
    
    input_, output = mz.select(num_classes, filter_size, num_filters, pool_size, num_dense, activation,
                learningrate, seq_shape, insertmotif, conv_l2, dense_l2, conv_do, dense_do, 
                use_transformer, model_type, main_model)

    model = Model(inputs=input_, outputs=output)

    if insertmotif != "False" and model_type =='main':
        filter_size = filter_size 
        motif_dict = OrderedDict()
        with open(insertmotif, 'r') as cbfile:
            for line in cbfile:
                if line.startswith(">"):
                    motif_name = line.split("/")[0].strip(">").strip()
                    motif_dict[motif_name] = []
                    continue
                nts = line.strip().split('\t')
                summ = float(nts[0]) + float(nts[1]) + float(nts[2]) + float(nts[3])
                motif_dict[motif_name].append(
                    [float(nts[0]) / summ, float(nts[1]) / summ, float(nts[2]) / summ, float(nts[3]) / summ])
        for motif_name in motif_dict:
            motif_dict[motif_name] = (np.array(motif_dict[motif_name]) - 0.25)

        number_of_motifs = len(motif_dict)
        conv_weights = model.layers[1].get_weights()
        # conv_weights[0][:, :, :] = conv_weights[0][:, :, :] * 0.1
        for i, name in enumerate(motif_dict):
            if number_of_motifs >= num_motifs == i + 1:
                break
            motif_size = len(motif_dict[name])
            if filter_size >= motif_size:
                start = int((filter_size - motif_size) / 2)
                conv_weights[0][start:start + motif_size, :, i] = motif_dict[name]
            else:
                start = int((motif_size - filter_size) / 2)
                conv_weights[0][:, :, i] = motif_dict[name][start:start + filter_size, :]
        model.layers[1].set_weights(conv_weights)
    

    optimizer = tf.keras.optimizers.Adam(lr=learningrate)
    lr_metric = get_lr_metric(optimizer)
    print('test')
    if(int(tf.version.VERSION[0])==2):
        model.compile(optimizer=optimizer,
                      #loss=['cosine_similarity'],#'mse'
                      loss=[custom_loss],
                      metrics=[tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.CosineSimilarity(axis=1), PearsonCorrelation(), lr_metric])
    else:
        model.compile(optimizer=optimizer,
                      loss='mse',
                      metrics=['accuracy',lr_metric])
    return model

def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer._decayed_lr(tf.float32) # I use ._decayed_lr method instead of .lr
    return lr

def build_transferlearning_model_deeppeak(model, num_classes, seq_shape=(500,4)):
    model_tl = model

    freeze=False
    if(freeze):
        # Define the layer name where you want to stop freezing
        stop_freezing_at = 'conv1d_4'
        
        # Initialize a flag
        should_train = False
        
        # Loop through all layers in the model
        for layer in model_tl.layers:
            # If this is the layer where you want to stop freezing, set the flag to True
            if layer.name == stop_freezing_at:
                should_train = True
            
            # If the layer is a Batch Normalization layer, always make it trainable
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = True
            else:
                layer.trainable = should_train
            
    optimizer = tf.keras.optimizers.Adam(lr=1e-5)
    lr_metric = get_lr_metric(optimizer)
    print('test')
    if(int(tf.version.VERSION[0])==2):
        model_tl.compile(optimizer=optimizer,
                      #loss=['cosine_similarity'],#'mse'
                      loss=[custom_loss],
                      metrics=[tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.CosineSimilarity(axis=1), PearsonCorrelation(), lr_metric])
    return model_tl
def build_transferlearning_model(model, num_classes, dense_units=1024, do=0.5, seq_shape=(500,4)):
    layer = model.layers[-9] # from topic model
    base_model = tf.keras.Model(inputs=model.input, outputs=layer.output)
    base_model.trainable = False
    inputs = tf.keras.Input(shape=seq_shape)
    x = base_model(inputs, training=False)
    x = tf.keras.layers.Flatten(name='flatt')(x)
    x = tf.keras.layers.Dropout(do, name = "intermediate_dropout")(x)
    x = tf.keras.layers.Dense(dense_units, activation="linear",name="intermediate_dense", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(name = "batchnorm")(x)
    x = tf.keras.layers.Activation('relu', name = "intermediate_relu")(x)
    x = tf.keras.layers.Dropout(0.2,name = "intermediate_dropout2")(x)

    outputs = tf.keras.layers.Dense(num_classes, activation="sigmoid", name= "classification_layer")(x)
    
    model_tl = tf.keras.Model(inputs, outputs)
    optimizer = tf.keras.optimizers.Adam(lr=1e-4)
    model_tl.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=[
                      [tf.keras.metrics.AUC(num_thresholds=200, curve='ROC', summation_method='interpolation', name='auROC', thresholds=None, multi_label=True, label_weights=None),
                       tf.keras.metrics.AUC(num_thresholds=200, curve='PR', summation_method='interpolation', name='auPR', thresholds=None, multi_label=True, label_weights=None), 'categorical_accuracy']
                  ])
    return model_tl


def class_weights(y):
    class_sum = (np.sum(y) / np.sum(y, axis=0)) / np.max(np.sum(y) / np.sum(y, axis=0))
    class_weight_dict = {}
    for i in range(y.shape[1]):
        class_weight_dict[i] = class_sum[i]
    return class_weight_dict


def augmentation(size, stride, name, input_bed):
    bd = input_bed
    intervals = []
    shift_number = 1
    start_point = 0
    for region in bd:
        if region.length > size:
            start_point = region.start
            shift_number = int((region.length - size) / stride) + 1
        if region.length < size:
            start_point = region.start - (size - region.length)
            shift_number = int((size - region.length) / stride) + 1
        if region.length == size:
            start_point = region.start
            shift_number = 1
        for shift in range(shift_number):
            if name == '4COL':
                intervals.append(
                    Interval(region.chrom, start_point + stride * shift, start_point + stride * shift + size,
                             name=region.name + "-" + str(shift * stride) + "_"))
            else:
                intervals.append(
                    Interval(region.chrom, start_point + stride * shift, start_point + stride * shift + size,
                             name=region.chrom + ":" + str(region.start) + "-" + str(region.end) + "-" + str(
                                 shift * stride) + "_" + name))

    output_bed = BedTool(intervals)
    return output_bed


def centerization(size, name, input_bed, chr_size_file=None):
    bd = input_bed
    intervals = []

    if chr_size_file:
        chr_size_dict = {}
        with open(chr_size_file, 'r') as chr_s:
            for line in chr_s:
                chr_size_dict[line.strip().split('\t')[0]] = int(line.strip().split('\t')[1])
        for region in bd:
            centre_point = int((region.start + region.end) / 2)
            if size % 2 == 0:
                upstream_point = centre_point - int(size / 2)
                downstream_point = centre_point + int(size / 2)
            else:
                upstream_point = centre_point - int(size / 2)
                downstream_point = centre_point + int(size / 2) + 1

            if downstream_point > chr_size_dict[region.chrom]:
                print(region.chrom, region.start, region.end, ">", chr_size_dict[region.chrom])
            if upstream_point > -1 and \
                    name != "4COL" and name != "4COL+" and downstream_point <= chr_size_dict[region.chrom]:
                intervals.append(Interval(region.chrom, upstream_point, downstream_point,
                                          name=region.chrom + ":" + str(region.start) + "-" + str(
                                              region.end) + "_" + name))
            if upstream_point > -1 and name == "4COL" and downstream_point <= chr_size_dict[region.chrom]:
                intervals.append(Interval(region.chrom, upstream_point, downstream_point, name=region.name))
            if upstream_point > -1 and name == "4COL+" and downstream_point <= chr_size_dict[region.chrom]:
                intervals.append(Interval(region.chrom, upstream_point, downstream_point,
                                          name=region.chrom + ":" + str(region.start) + "-" + str(
                                              region.end) + "-" + region.name + "_"))
    else:
        for region in bd:
            centre_point = int((region.start + region.end) / 2)
            if size % 2 == 0:
                upstream_point = centre_point - int(size / 2)
                downstream_point = centre_point + int(size / 2)
            else:
                upstream_point = centre_point - int(size / 2)
                downstream_point = centre_point + int(size / 2) + 1
            if upstream_point > -1 and name != "4COL" and name != "4COL+":
                intervals.append(Interval(region.chrom, upstream_point, downstream_point,
                                          name=region.chrom + ":" + str(region.start) + "-" + str(
                                              region.end) + "_" + name))
            if upstream_point > -1 and name == "4COL":
                intervals.append(Interval(region.chrom, upstream_point, downstream_point, name=region.name))
            if upstream_point > -1 and name == "4COL+":
                intervals.append(Interval(region.chrom, upstream_point, downstream_point,
                                          name=region.chrom + ":" + str(region.start) + "-" + str(
                                              region.end) + "-" + region.name + "_"))
    output_bed = BedTool(intervals)
    return output_bed


def printFasta(genome, bedfile, outfile):
    bedfile = bedfile.sequence(fi=genome, name=True)
    with open(outfile, 'w') as wf:
        print(open(bedfile.seqfn).read(), end="", file=wf)


def load_model(output_dir, name_hdf5):
    model_json_file = open(os.path.join(output_dir, 'model.json'))
    model_json = model_json_file.read()
    model = tf.keras.models.model_from_json(model_json)
    model.load_weights(os.path.join(output_dir, name_hdf5))
    return model


def shuffle_label(label):
    for i in range(len(label.T)):
        label.T[i] = shuffle(label.T[i])
    return label


def calculate_roc_pr(score, label):
    output = np.zeros((len(label.T), 2))
    for i in range(len(label.T)):
        roc_ = roc_auc_score(label.T[i], score.T[i])
        pr_ = average_precision_score(label.T[i], score.T[i])
        output[i] = [roc_, pr_]
    return output


def generate_random_short_seqs(len_char):
    X_all = []
    allchar = "ACTG"
    for i in range(2000000):
        password = "".join(choice(allchar) for _x in range(len_char))
        X_all.append(password)
    X_all = list(set(X_all))
    return X_all


def analyze_rocpr(model, nonAugmented_data_dict, output_dir, num_classes, topic_dict_onlynum, topic_dict_onlyname):
    roc_pr_dict = {"train": {}, "valid": {}, "test": {}, "shuffle": {}}
    roc_pr_dict["train"]["score"] = model.predict(nonAugmented_data_dict["train_data"])
    roc_pr_dict["train"]["label"] = nonAugmented_data_dict["y_train"]
    roc_pr_dict["valid"]["score"] = model.predict(nonAugmented_data_dict["valid_data"])
    roc_pr_dict["valid"]["label"] = nonAugmented_data_dict["y_valid"]
    roc_pr_dict["test"]["score"] = model.predict(nonAugmented_data_dict["test_data"])
    roc_pr_dict["test"]["label"] = nonAugmented_data_dict["y_test"]
    roc_pr_dict["shuffle"]["score"] = np.array(roc_pr_dict["train"]["score"], copy=True)
    nonAugmented_y_train_shuffle = np.array(nonAugmented_data_dict["y_train"], copy=True)
    roc_pr_dict["shuffle"]["label"] = shuffle_label(nonAugmented_y_train_shuffle)
    for sets in roc_pr_dict:
        roc_pr_dict[sets]["roc_pr"] = calculate_roc_pr(roc_pr_dict[sets]["score"], roc_pr_dict[sets]["label"])
    np.savez(os.path.join(output_dir, 'roc_pr.npz'), roc_pr=roc_pr_dict)

    matplotlib.style.use('default')
    fig = plt.figure(figsize=(50, 20))
    ax = fig.add_subplot(3, 1, 1)
    ax.set_ylabel('auROC')
    ax.scatter(list(range(num_classes)), roc_pr_dict["train"]["roc_pr"].T[0], color='red', label='TRAIN')
    ax.scatter(list(range(num_classes)), roc_pr_dict["valid"]["roc_pr"].T[0], color='green',
               label='VALIDATION')
    ax.scatter(list(range(num_classes)), roc_pr_dict["test"]["roc_pr"].T[0], color='blue', label='TEST')
    ax.scatter(list(range(num_classes)), roc_pr_dict["shuffle"]["roc_pr"].T[0], color='gray', label='SHUFFLED')
    ax.set_ylim([0, 1])
    _ = ax.set_xticks(range(num_classes))
    _ = ax.set_xticklabels(list(topic_dict_onlynum.values()))
    ax.legend()

    ax = fig.add_subplot(3, 1, 2)
    ax.set_ylabel('auPR')
    ax.scatter(list(range(num_classes)), roc_pr_dict["train"]["roc_pr"].T[1], color='red', label='TRAIN')
    ax.scatter(list(range(num_classes)), roc_pr_dict["valid"]["roc_pr"].T[1], color='green',
               label='VALIDATION')
    ax.scatter(list(range(num_classes)), roc_pr_dict["test"]["roc_pr"].T[1], color='blue', label='TEST')
    ax.scatter(list(range(num_classes)), roc_pr_dict["shuffle"]["roc_pr"].T[1], color='gray', label='SHUFFLED')
    ax.set_ylim([0, 1])
    _ = ax.set_xticks(range(num_classes))
    _ = ax.set_xticklabels(list(topic_dict_onlyname.values()), rotation=90)

    fig.savefig(os.path.join(output_dir, "all_roc_pr.png"))
    np.savez(os.path.join(output_dir, 'roc_pr_dict.npz'), X=roc_pr_dict)
    with open(os.path.join(output_dir, "roc_pr_dict.txt"), 'w') as file_to_write:
        print('ROC_train', *(roc_pr_dict["train"]["roc_pr"].T[0]), sep='\t', file=file_to_write)
        print('ROC_valid', *(roc_pr_dict["valid"]["roc_pr"].T[0]), sep='\t', file=file_to_write)
        print('ROC_test', *(roc_pr_dict["test"]["roc_pr"].T[0]), sep='\t', file=file_to_write)
        print('ROC_shuffle', *(roc_pr_dict["shuffle"]["roc_pr"].T[0]), sep='\t', file=file_to_write)
        print('PR_train', *(roc_pr_dict["train"]["roc_pr"].T[1]), sep='\t', file=file_to_write)
        print('PR_valid', *(roc_pr_dict["valid"]["roc_pr"].T[1]), sep='\t', file=file_to_write)
        print('PR_test', *(roc_pr_dict["test"]["roc_pr"].T[1]), sep='\t', file=file_to_write)
        print('PR_shuffle', *(roc_pr_dict["shuffle"]["roc_pr"].T[1]), sep='\t', file=file_to_write)


def analyze_motif(model, motifwidth, output_dir):
    gen_seq_motifs = generate_random_short_seqs(motifwidth)
    X_all_oc = np.array([one_hot_encode_along_row_axis(i) for i in gen_seq_motifs]).squeeze(axis=1)
    weights = model.get_weights()[0]
    filter_manual_valid = tf.nn.conv1d(X_all_oc.astype('float32'), weights, 1, padding='VALID')
    filter_manual_valid_numpy = filter_manual_valid.numpy()

    counts_100 = []
    nsite_true = 100
    motif_numbers = range(weights.shape[2])
    for motif in tqdm(motif_numbers):
        top100 = heapq.nlargest(nsite_true, enumerate(filter_manual_valid_numpy[:, :, motif]),
                                key=operator.itemgetter(1))
        cont = np.zeros((motifwidth, 4))
        for i in range(len(top100)):
            cont += X_all_oc[top100[i][0]]
        counts_100.append(cont / nsite_true)
        print(motif, end=',')
    np.savez(os.path.join(output_dir, 'motifs_top100.npz'), X=counts_100)

    with open(os.path.join(output_dir, "motifs_top100.txt"), "w") as file_to_write:
        for j in range(weights.shape[2]):
            print(' ', file=file_to_write)
            for i in counts_100[j]:
                print(i[0], i[1], i[2], i[3], file=file_to_write)

    if not os.path.exists(os.path.join(output_dir, 'tomtom')):
        os.makedirs(os.path.join(output_dir, 'tomtom'))

    command = "{shfile} {output_dir} "
    bashCommand = command.format(shfile="/staging/leuven/stg_00002/lcb/nkemp/tools/tomtom_run.sh",
                                 output_dir=output_dir)
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    _output, _error = process.communicate()

    if not os.path.exists(os.path.join(output_dir, 'motif_cb_x100')):
        os.makedirs(os.path.join(output_dir, 'motif_cb_x100'))

    for j in tqdm(range(weights.shape[2])):
        with open(os.path.join(output_dir, "motif_cb_x100/motif_" + str(j + 1) + ".cb"), 'w') as file_to_write:
            print('>motif_' + str(j + 1), file=file_to_write)
            for i in counts_100[j]:
                print("\t".join(str(int(round(e * 100))) for e in i), file=file_to_write)

    if not os.path.exists(os.path.join(output_dir, 'found_motifs')):
        os.makedirs(os.path.join(output_dir, 'found_motifs'))

    for i in tqdm(range(len(counts_100))):
        data = weblogo.LogoData.from_counts('ACGT', counts_100[i])
        options = weblogo.LogoOptions(fineprint="DeepTopic",
                                      logo_title="motif_" + str(i + 1),
                                      color_scheme=weblogo.classic,
                                      stack_width=weblogo.std_sizes["large"])
        logo_format = weblogo.LogoFormat(data, options)
        a = weblogo.png_formatter(data, logo_format)
        with open(os.path.join(output_dir, "found_motifs/motif_" + str(i + 1) + ".png"), "wb") as png:
            png.write(a)
    print('finished')


def analyze_print_importance(topic_dict, loss, loss_real, output_dir):

    if not os.path.exists(os.path.join(output_dir, 'motif_importance')):
        os.makedirs(os.path.join(output_dir, 'motif_importance'))

    for topcc in range(len(topic_dict)):
        first_n = 15
        matrx = (loss - loss_real)
        for i in range(len(topic_dict)):
            matrx.T[i] = matrx.T[i] / max(matrx.T[i])
        sorting_index = [i[0] for i in sorted(enumerate(matrx.T[topcc].T), key=lambda x: x[1])]
        objects = [i + 1 for i in sorting_index[::-1][0:first_n]][::-1]
        y_pos = np.arange(len(matrx.T[topcc].T[sorting_index[::-1]][0:first_n]))
        performance = matrx.T[topcc].T[sorting_index[::-1]][0:first_n][::-1]
        fig = plt.figure(figsize=(15, 30), facecolor='white')
        ax = fig.gca()
        ax.set_title(1)
        row_s = 100
        col_s = 10
        ax = plt.subplot2grid((row_s, col_s), (0, 5), rowspan=row_s - 7, colspan=2)
        ax.plot(performance, y_pos, '--', color='gray', linewidth=3)
        ax.scatter(performance, y_pos, color='red', linewidths=8)
        ax.set_yticks(y_pos, )
        ax.set_yticklabels(objects)
        ax.set_ylabel('Filter Number', rotation=270)
        ax.yaxis.set_label_position('right')
        ax.set_xlabel('Filter Importance (Normalized)', )
        ax.xaxis.set_label_position('top')
        ax.tick_params(labeltop=True, labelright=True, labelleft=False)
        ax.set_xlim([-0.05, 1.05])
        ax.grid(True)
        ax = plt.subplot2grid((row_s, col_s), (0, 0), rowspan=1, colspan=5)
        ax.grid(False)
        ax.axis('off')
        ax.set_title(topic_dict[topcc], fontsize=30)
        for i, j in enumerate(reversed(objects)):
            ax = plt.subplot2grid((row_s, col_s), (i * int(row_s / len(objects)) + 1, 0),
                                  rowspan=int(row_s / len(objects)), colspan=5)
            img_ = mpimg.imread(os.path.join(output_dir, "found_motifs/motif_" + str(int(j)) + ".png"))
            ax.imshow(img_)
            ax.grid(False)
            ax.axis("off")
        fig.savefig(os.path.join(output_dir, "motif_importance/" + topic_dict[topcc].split(' ')[0] + ".png"))
        plt.close(fig)
        print('finished')


def my_print(text):
    sys.stdout.write(str(text))
    sys.stdout.flush()
    
@tf.function
def contribution_input_grad(model, input_sequence,
                              target_mask, output_head='mouse'):
    input_sequence = input_sequence

    target_mask_mass = tf.reduce_sum(target_mask)
    with tf.GradientTape() as tape:
        tape.watch(input_sequence)
        prediction = tf.reduce_sum(
          target_mask *
          model(input_sequence)) / target_mask_mass

    input_grad = tape.gradient(prediction, input_sequence) * input_sequence
    input_grad = tf.squeeze(input_grad, axis=0)
    res = tf.reduce_sum(input_grad, axis=-1)
    return tf.reduce_sum(input_grad, axis=-1)
    
def get_contribution_scores(model_, seq, topic_nr, seq_len=500):
    seq = np.expand_dims(seq, axis=0)
    predictions = model_.predict(seq)[0]
    #print(predictions)
    target_mask = np.zeros_like(predictions)
    topic_nr = topic_nr - 1 
    target_mask[topic_nr] = 1

    # This will take some time since tf.function needs to get compiled.
    contribution_scores = contribution_input_grad(model_,seq.astype(np.float32), target_mask).numpy()
    contribution_scores = np.repeat(contribution_scores,4)
    contribution_scores = np.reshape(contribution_scores, (500,4))
    return contribution_scores


def generateShapValues(model, selectedtopics_shap, nonAugmented_data_dict, outputdirc):
    rn = np.random.choice(nonAugmented_data_dict["nonAugmented_data"].shape[0], 500, replace=False)
    explainer = shap.DeepExplainer(model, nonAugmented_data_dict["nonAugmented_data"][rn])
    
    #seqs = np.load("/staging/leuven/stg_00002/lcb/nkemp/mouse/predictions/OPC_scplus_seqsonehot.npy")
    predictions = model.predict(nonAugmented_data_dict["nonAugmented_data"])
    #predictions = model.predict(seqs)

    top_n = 1000
    index = []
    for topic__ in selectedtopics_shap:
        sorted_indices = np.argsort(predictions[:, topic__])
        top500_indices = sorted_indices[::-1][:top_n]
        index = index + list(top500_indices)
    index = np.array(list(set(index)))
    regions = nonAugmented_data_dict["nonAugmented_data"][index]
    #regions = seqs[index]
    
    shap_dict = {}
    tasks = []
    for topic__ in selectedtopics_shap:
        my_print("Topic_" + str(topic__ + 1) + "\n")
        task = "Topic_" + str(topic__ + 1)
        tasks.append(task)
        shap_dict[task] = {}
        for i in tqdm(range(len(regions))):
            #if i % 50 == 0:
                #my_print(str(i) + "_")
            shap_values_, indexes_ = explainer.shap_values(
                regions[i:i + 1],
                output_rank_order=str(topic__),
                ranked_outputs=1,
                check_additivity=False)
            shap_dict[task][i] = [shap_values_]


    task_to_scores = OrderedDict()
    task_to_hyp_scores = OrderedDict()
    for task in tasks:
        task_to_scores[task] = [(shap_dict[task][ids_][0][0] * (regions[ids_])).squeeze() for ids_ in shap_dict[task]]
        task_to_hyp_scores[task] = [shap_dict[task][ids_][0][0].squeeze() for ids_ in shap_dict[task]]
    onehot_data = regions

    print(task_to_hyp_scores[tasks[0]][0].shape)
    print(onehot_data[0].shape)
    print(task_to_scores[tasks[0]][0].shape)
    save_name = ""
    for topic__ in selectedtopics_shap:
        save_name = save_name + "_" + str(topic__+1)

    f = open(os.path.join(outputdirc, "deepExpVal_Topic"+save_name+".pkl"), "wb")
    pickle.dump(tasks, f)
    pickle.dump(task_to_scores, f)
    pickle.dump(task_to_hyp_scores, f)
    pickle.dump(onehot_data, f)
    f.close()


def plot_a(ax, base, left_edge, height, color):
    a_polygon_coords = [
        np.array([
            [0.0, 0.0],
            [0.5, 1.0],
            [0.5, 0.8],
            [0.2, 0.0],
        ]),
        np.array([
            [1.0, 0.0],
            [0.5, 1.0],
            [0.5, 0.8],
            [0.8, 0.0],
        ]),
        np.array([
            [0.225, 0.45],
            [0.775, 0.45],
            [0.85, 0.3],
            [0.15, 0.3],
        ])
    ]
    for polygon_coords in a_polygon_coords:
        ax.add_patch(matplotlib.patches.Polygon((np.array([1, height])[None, :] * polygon_coords
                                                 + np.array([left_edge, base])[None, :]),
                                                facecolor=color, edgecolor=color))


def plot_c(ax, base, left_edge, height, color):
    ax.add_patch(matplotlib.patches.Ellipse(xy=[left_edge + 0.65, base + 0.5 * height], width=1.3, height=height,
                                            facecolor=color, edgecolor=color))
    ax.add_patch(
        matplotlib.patches.Ellipse(xy=[left_edge + 0.65, base + 0.5 * height], width=0.7 * 1.3, height=0.7 * height,
                                   facecolor='white', edgecolor='white'))
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge + 1, base], width=1.0, height=height,
                                              facecolor='white', edgecolor='white', fill=True))


def plot_g(ax, base, left_edge, height, color):
    ax.add_patch(matplotlib.patches.Ellipse(xy=[left_edge + 0.65, base + 0.5 * height], width=1.3, height=height,
                                            facecolor=color, edgecolor=color))
    ax.add_patch(
        matplotlib.patches.Ellipse(xy=[left_edge + 0.65, base + 0.5 * height], width=0.7 * 1.3, height=0.7 * height,
                                   facecolor='white', edgecolor='white'))
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge + 1, base], width=1.0, height=height,
                                              facecolor='white', edgecolor='white', fill=True))
    ax.add_patch(
        matplotlib.patches.Rectangle(xy=[left_edge + 0.825, base + 0.085 * height], width=0.174, height=0.415 * height,
                                     facecolor=color, edgecolor=color, fill=True))
    ax.add_patch(
        matplotlib.patches.Rectangle(xy=[left_edge + 0.625, base + 0.35 * height], width=0.374, height=0.15 * height,
                                     facecolor=color, edgecolor=color, fill=True))


def plot_t(ax, base, left_edge, height, color):
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge + 0.4, base],
                                              width=0.2, height=height, facecolor=color, edgecolor=color, fill=True))
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge, base + 0.8 * height],
                                              width=1.0, height=0.2 * height, facecolor=color, edgecolor=color,
                                              fill=True))


default_colors = {0: 'green', 1: 'blue', 2: 'orange', 3: 'red'}
default_plot_funcs = {0: plot_a, 1: plot_c, 2: plot_g, 3: plot_t}


def plot_weights_given_ax(ax, array,
                          height_padding_factor,
                          length_padding,
                          subticks_frequency,
                          highlight,
                          colors=default_colors,
                          plot_funcs=default_plot_funcs):
    if len(array.shape) == 3:
        array = np.squeeze(array)
    assert len(array.shape) == 2, array.shape
    if array.shape[0] == 4 and array.shape[1] != 4:
        array = array.transpose(1, 0)
    assert array.shape[1] == 4
    max_pos_height = 0.0
    min_neg_height = 0.0
    heights_at_positions = []
    depths_at_positions = []
    for i in range(array.shape[0]):
        # sort from smallest to highest magnitude
        acgt_vals = sorted(enumerate(array[i, :]), key=lambda x: abs(x[1]))
        positive_height_so_far = 0.0
        negative_height_so_far = 0.0
        for letter in acgt_vals:
            plot_func = plot_funcs[letter[0]]
            color = colors[letter[0]]
            if letter[1] > 0:
                height_so_far = positive_height_so_far
                positive_height_so_far += letter[1]
            else:
                height_so_far = negative_height_so_far
                negative_height_so_far += letter[1]
            plot_func(ax=ax, base=height_so_far, left_edge=i, height=letter[1], color=color)
        max_pos_height = max(max_pos_height, positive_height_so_far)
        min_neg_height = min(min_neg_height, negative_height_so_far)
        heights_at_positions.append(positive_height_so_far)
        depths_at_positions.append(negative_height_so_far)

    # now highlight any desired positions; the key of
    # the highlight dict should be the color
    for color in highlight:
        for start_pos, end_pos in highlight[color]:
            assert start_pos >= 0.0 and end_pos <= array.shape[0]
            min_depth = np.min(depths_at_positions[start_pos:end_pos])
            max_height = np.max(heights_at_positions[start_pos:end_pos])
            ax.add_patch(
                matplotlib.patches.Rectangle(xy=[start_pos, min_depth],
                                             width=end_pos - start_pos,
                                             height=max_height - min_depth,
                                             edgecolor=color, fill=False))

    ax.set_xlim(-length_padding, array.shape[0] + length_padding)
    ax.xaxis.set_ticks(np.arange(0.0, array.shape[0] + 1, subticks_frequency))
    height_padding = max(abs(min_neg_height) * (height_padding_factor),
                         abs(max_pos_height) * (height_padding_factor))
    ax.set_ylim(min_neg_height - height_padding, max_pos_height + height_padding)
    return ax


def plot_weights(array, fig, n, n1, n2, zoom, title='', ylab='',
                 height_padding_factor=0.2,
                 length_padding=1.0,
                 subticks_frequency=20,
                 colors=default_colors,
                 plot_funcs=default_plot_funcs,
                 highlight={}):
    ax = fig.add_subplot(n, n1, n2)
    ax.set_title(title)
    ax.set_ylabel(ylab)
    diff = array.shape[1]-zoom
    start=0+int(diff/2)
    end=array.shape[1]-int(diff/2)
    y = plot_weights_given_ax(ax=ax, array=array[0,start:end],
                              height_padding_factor=height_padding_factor,
                              length_padding=length_padding,
                              subticks_frequency=subticks_frequency,
                              colors=colors,
                              plot_funcs=plot_funcs,
                              highlight=highlight)
    return fig, ax


def labelLine(line,x,label=None,align=True,**kwargs):

    ax = line.axes
    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if (x < xdata[0]) or (x > xdata[-1]):
        print('x label location is outside data range!')
        return

    # Find corresponding y co-ordinate and angle of the line
    ip = 1
    for i in range(len(xdata)):
        if x < xdata[i]:
            ip = i
            break

    y = ydata[ip-1] + (ydata[ip]-ydata[ip-1])*(x-xdata[ip-1])/(xdata[ip]-xdata[ip-1])

    if not label:
        label = line.get_label()

    if align:
        # Compute the slope
        dx = xdata[ip] - xdata[ip-1]
        dy = ydata[ip] - ydata[ip-1]
        ang = degrees(atan2(dy,dx))

        # Transform to screen co-ordinates
        pt = np.array([x,y]).reshape((1,2))
        trans_angle = ax.transData.transform_angles(np.array((ang,)),pt)[0]

    else:
        trans_angle = 0

    # Set a bunch of keyword arguments
    if 'color' not in kwargs:
        kwargs['color'] = line.get_color()

    if ('horizontalalignment' not in kwargs) and ('ha' not in kwargs):
        kwargs['ha'] = 'center'

    if ('verticalalignment' not in kwargs) and ('va' not in kwargs):
        kwargs['va'] = 'center'

    if 'backgroundcolor' not in kwargs:
        kwargs['backgroundcolor'] = ax.get_facecolor()

    if 'clip_on' not in kwargs:
        kwargs['clip_on'] = True

    if 'zorder' not in kwargs:
        kwargs['zorder'] = 2.5

    ax.text(x, y, label, rotation=trans_angle, **kwargs)


def labelLines(lines, align=True, xvals=None, **kwargs):

    ax = lines[0].axes
    labLines = []
    labels = []

    # Take only the lines which have labels other than the default ones
    for line in lines:
        label = line.get_label()
        if "_line" not in label:
            labLines.append(line)
            labels.append(label)

    if xvals is None:
        xmin, xmax = ax.get_xlim()
        xvals = np.linspace(xmin, xmax, len(labLines)+2)[1:-1]

    for line, x, label in zip(labLines, xvals, labels):
        labelLine(line, x, label, align, **kwargs)


def plot_prediction_givenax(model, fig, ntrack, track_no, seq_onehot):
    NUM_CLASSES = model.output_shape[1]
    real_score = model.predict(seq_onehot)[0]
    ax = fig.add_subplot(ntrack, 2, track_no*2-1)
    ax.margins(x=0)
    ax.set_ylabel('Prediction', color='red')
    ax.plot(real_score, '--', color='gray', linewidth=3)
    ax.scatter(range(NUM_CLASSES), real_score, marker='o', color='red', linewidth=11)
    ax.tick_params(axis='y', labelcolor='red')
    ax.set_xticks(range(NUM_CLASSES),)
    ax.set_xticklabels(range(1, NUM_CLASSES+1))
    ax.grid(True)
    return ax


def plot_deepexplainer_givenax(explainer, fig, ntrack, track_no, seq_onehot, topic, zoom=500):
    topic = topic - 1
    shap_values_, indexes_ = explainer.shap_values(seq_onehot,
                                                   output_rank_order=str(topic),
                                                   ranked_outputs=1,
                                                   check_additivity=False)
    _, ax1 = plot_weights(shap_values_[0][0]*seq_onehot,
                          fig, ntrack, 1, track_no,
                          title="Topic_" + str(topic+1), subticks_frequency=10, ylab="DeepExplainer", zoom=zoom)
    return ax1


def plot_mutagenesis_givenax(model, fig, ntrack, track_no, seq_onehot, topic, zoom=500):
    seq_shape = model.input_shape[1:]
    NUM_CLASSES = model.output_shape[1]
    topic = topic-1
    arrr_A = np.zeros((NUM_CLASSES, seq_shape[0]))
    arrr_C = np.zeros((NUM_CLASSES, seq_shape[0]))
    arrr_G = np.zeros((NUM_CLASSES, seq_shape[0]))
    arrr_T = np.zeros((NUM_CLASSES, seq_shape[0]))
    real_score = model.predict(seq_onehot)[0]
    for mutloc in range(seq_shape[0]):
        new_X = np.copy(seq_onehot)
        if new_X[0][mutloc, :][0] == 0:
            new_X[0][mutloc, :] = np.array([1, 0, 0, 0], dtype='int8')
            arrr_A[:, mutloc] = (real_score - model.predict(new_X)[0])
        if new_X[0][mutloc, :][1] == 0:
            new_X[0][mutloc, :] = np.array([0, 1, 0, 0], dtype='int8')
            arrr_C[:, mutloc] = (real_score - model.predict(new_X)[0])
        if new_X[0][mutloc, :][2] == 0:
            new_X[0][mutloc, :] = np.array([0, 0, 1, 0], dtype='int8')
            arrr_G[:, mutloc] = (real_score - model.predict(new_X)[0])
        if new_X[0][mutloc, :][3] == 0:
            new_X[0][mutloc, :] = np.array([0, 0, 0, 1], dtype='int8')
            arrr_T[:, mutloc] = (real_score - model.predict(new_X)[0])
    arrr_A[arrr_A == 0] = None
    arrr_C[arrr_C == 0] = None
    arrr_G[arrr_G == 0] = None
    arrr_T[arrr_T == 0] = None

    diff = seq_shape[0]-zoom
    start=0+int(diff/2)
    end=seq_shape[0]-int(diff/2)
    ax = fig.add_subplot(ntrack, 1, track_no)
    ax.set_ylabel('In silico\nMutagenesis\nTopic_'+str(topic+1))
    ax.set_title("Topic_" + str(topic+1))
    ax.scatter(range(zoom), -1*arrr_A[topic][start:end], label='A', color='green')
    ax.scatter(range(zoom), -1*arrr_C[topic][start:end], label='C', color='blue')
    ax.scatter(range(zoom), -1*arrr_G[topic][start:end], label='G', color='orange')
    ax.scatter(range(zoom), -1*arrr_T[topic][start:end], label='T', color='red')
    ax.legend()
    ax.axhline(y=0, linestyle='--', color='gray')
    ax.set_xlim((0, zoom))
    _ = ax.set_xticks(np.arange(0, zoom+1, 10))
    
    return ax, arrr_A[topic],arrr_C[topic],arrr_G[topic],arrr_T[topic]

###################
# simple function for getting sequencs from FASTA
#
# returns: dictionary = { header: sequence }
#
##################
def simple_read_fastas(list_fastas):

    dict_seqs = {}
    
    for fastaf in list_fastas:
        head = ""
        with open(fastaf, "r") as f:
            for line in f:
                line = line.rstrip()           
                m = re.search(r'^>(.*)', line)
                
                if m is not None:
                    head = m.group(1)
                else:
                    uline = line.upper()
                    if head in dict_seqs.keys():
                        dict_seqs[head] = dict_seqs[head] + uline
                    else:
                        dict_seqs[head] = uline
    return dict_seqs

####################################
# function for computing Shap values from FASTA file input
# 
####################################
def generateShapValuesFromOneHot(model, selectedtopics_shap, onehot_input, onehot_background, outputdirc, n_pick_top=5000, seqids_input=None):

    my_print("Running DeepExplainer (background) ...\n")
    explainer = shap.DeepExplainer(model, onehot_background)

    # getting predictions
    my_print("Running predictions ...\n")
    predictions = model.predict(onehot_input)
    my_print("done.\n")
    
    # picking top 5000 regions
    if n_pick_top >= onehot_input.shape[0]:
        n_pick_top = onehot_input.shape[0]-1
    
    index = []
    for topic__ in selectedtopics_shap:
        sorted_indices = np.argsort(predictions[:, topic__])
        top_indices = sorted_indices[::-1][:n_pick_top]
        index = index + list(top_indices)
    index = np.array(list(set(index)))
    regions = onehot_input[index]

    my_print("Computing Shap values ...\n")
    shap_dict = {}
    tasks = []
    for topic__ in selectedtopics_shap:
        my_print("Topic_" + str(topic__ + 1) + "\n")
        task = "Topic_" + str(topic__ + 1)
        tasks.append(task)
        shap_dict[task] = {}
        for i in range(len(regions)):
            if i % 50 == 0:
                my_print(str(i) + "_")
            shap_values_, indexes_ = explainer.shap_values(
                regions[i:i + 1],
                output_rank_order=str(topic__),
                ranked_outputs=1,
                check_additivity=False)
            shap_dict[task][i] = [shap_values_]

    task_to_scores = OrderedDict()
    task_to_hyp_scores = OrderedDict()
    for task in tasks:
        task_to_scores[task] = [(shap_dict[task][ids_][0][0] * (regions[ids_])).squeeze() for ids_ in shap_dict[task]]
        task_to_hyp_scores[task] = [shap_dict[task][ids_][0][0].squeeze() for ids_ in shap_dict[task]]
    onehot_data = regions

    print(task_to_hyp_scores[tasks[0]][0].shape)
    print(onehot_data[0].shape)
    print(task_to_scores[tasks[0]][0].shape)
    save_name = ""
    for topic__ in selectedtopics_shap:
        save_name = save_name + "_" + str(topic__+1)

    f = open(os.path.join(outputdirc, "deepExpVal_Topic"+save_name+"_fromOneHot.pkl"), "wb")
    pickle.dump(tasks, f)
    pickle.dump(task_to_scores, f)
    pickle.dump(task_to_hyp_scores, f)
    pickle.dump(onehot_data, f)
    # if specified store names of regions
    if seqids_input is not None:
        pickle.dump([seqids_input[idx] for idx in index], f)
    f.close()
