import os
import click
import numpy as np
import tensorflow as tf

import argparse
import errno
import os
import sys
import time
import matplotlib
import utils
import pandas as pd
import pickle
import numpy as np
import tensorflow as tf
from tqdm import tqdm
matplotlib.use('pdf')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def make_argument_parser():
    parser = argparse.ArgumentParser(
        description="Train model.",)
    parser.add_argument('--inputtopics', '-i', type=str, required=True,
                        help='Folders containing all topic.bed files.')
    parser.add_argument('--numTopics', '-nt', type=int, required=True,
                        help='Number of topics resulted from cisTopic.')
    parser.add_argument('--consensuspeaks', '-su', type=str, required=True,
                        help='Consensus peaks bed file.')
    parser.add_argument('--genome', '-g', type=str, required=True,
                        help='Path to genome.fa file')
    parser.add_argument('--chrSize', '-c', type=str, required=True,
                        help='Path to chrom.sizes file')
    parser.add_argument('--validchroms', '-v', type=str, required=True, nargs='+',
                        help='Chromosome(s) to set aside for validation (ex: chr11).')
    parser.add_argument('--testchroms', '-t', type=str, required=True, nargs='+',
                        help='Chromosome(s) to set aside for testing (ex: chr2).')
    parser.add_argument('--selectedtopics', '-st', type=str, required=False,
                        default="all",
                        help='Txt file containing selected topic id.')
    parser.add_argument('--epochs', '-e', type=int, required=False,
                        default=100,
                        help='Epochs to train (default: 100).')
    parser.add_argument('--stride', '-strd', type=int, required=False,
                        default=10,
                        help='Stride size.')
    parser.add_argument('--patience', '-ep', type=int, required=False,
                        default=6,
                        help='Number of epochs with no improvement after which training will be stopped (default: 6).')
    parser.add_argument('--learningrate', '-lr', type=float, required=False,
                        default=0.001,
                        help='Learning rate (default: 0.001).')
    parser.add_argument('--activation', '-act', type=str, required=False,
                        default="gelu",
                        help='Activation function of first convolutional layer')
    parser.add_argument('--conv_l2', '-cl2', type=float, required=False,
                        default=1e-6,
                        help='Convolution l2 regularization (default: 1e-6).')
    parser.add_argument('--dense_l2', '-dl2', type=float, required=False,
                        default=1e-3,
                        help='Dense l2 regularization (default: 1e-3).')
    parser.add_argument('--convDO', '-cdo', type=float, required=False,
                        default=0.15,
                        help='Dropout after convolution (default: 0.5).')
    parser.add_argument('--denseDO', '-ddo', type=float, required=False,
                        default=0.5,
                        help='Dropout of LSTM layer (default: 0.2).')
    parser.add_argument('--seqlen', '-L', type=int, required=False,
                        default=500,
                        help='Length of sequence input (default: 500).')
    parser.add_argument('--batchsize', '-batch', type=int, required=False,
                        default=512,
                        help='Batch size.')
    parser.add_argument('--motifwidth', '-w', type=int, required=False,
                        default=17,
                        help='Width of the convolutional kernels (default: 17).')
    parser.add_argument('--numkernels', '-k', type=int, required=False,
                        default=1024,
                        help='Number of kernels in first convolutional layer (default: 1024).')
    parser.add_argument('--insertmotif', '-im', type=str, required=False,
                        default="False",
                        help='Path to motif cb file (default: False).')
    parser.add_argument('--maxpoolsize', '-mps', type=int, required=False,
                        default=4,
                        help='Maxpooling window size (default: 4).')
    parser.add_argument('--numdense', '-d', type=int, required=False,
                        default=1024,
                        help='Number of dense units in model (default: 1024).')
    parser.add_argument('--seed', '-s', type=int, required=False,
                        default=777,
                        help='Random seed for consistency (default: 777).')
    parser.add_argument('--transferlearn', '-tf', type=str, required=False,
                        default="False",
                        help='Path to weights')
    parser.add_argument('--doubleDATArc', '-ddrc', type=str, required=False,
                        default="False",
                        help='Augment the data by using reverse compliment')
    parser.add_argument('--wandbname', '-wb', type=str, required=False,
                        default="False",
                        help='Name of the project')
    parser.add_argument('--wandbUser', '-wbu', type=str, required=False,
                        default="False",
                        help='lcb or username')
    parser.add_argument('--mergetopic', '-mt', type=str, required=False,
                        default="False",
                        help='Merging Topics')
    parser.add_argument('--usetransformer', '-rs', type=str, required=False,
                        default="False",
                        help='Use transformer layers after convolution.')
    parser.add_argument('--useclassweight', '-ucw', type=str, required=False,
                        default="True",
                        help='use class weight')
    parser.add_argument('--gpuname', '-gpu', type=str, required=False,
                        default="False",
                        help='Gpu-id')
    parser.add_argument('--useCreatedData', '-ucf', type=str, required=False,
                        default="False",
                        help='nonAugmented or Augmented')
    parser.add_argument('--namehdf5', '-nhdf5', type=str, required=False,
                        default="False",
                        help='Model name hdf5 file')
    parser.add_argument('--runType', '-rt', type=str, required=False,
                        default="train",
                        help='train, analyze, analyzeRocpr, analyzeGenerateMotif, analyzeGenerateShapValues')
    parser.add_argument('--selectedtopicsShap', '-sts', type=str, required=False,
                        default="False",
                        help='Txt file containing selected topic id for SHAP.')
    parser.add_argument('--model', '-mdl', type=str, required=False,
                        default="main",
                        help='Model selection.')
    parser.add_argument('--shapRegions', type=str, required=False,
                        default="False",
                        help='Txt file containing path to files used for Shap value computation.')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-oc', '--outputdirc', type=str,
                       help='The output directory. Will overwrite if directory already exists.')
    return parser

def main():
    parser = make_argument_parser()
    args = parser.parse_args()

    inputtopics = args.inputtopics
    consensuspeaks = args.consensuspeaks
    genome = args.genome
    chrSize = args.chrSize
    validchroms = args.validchroms
    testchroms = args.testchroms
    transferlearn = args.transferlearn
    doubleDATArc = args.doubleDATArc
    wandbname = args.wandbname
    wandbUser = args.wandbUser
    mergetopic = args.mergetopic
    useclassweight = args.useclassweight
    useCreatedData = args.useCreatedData
    gpuname = args.gpuname
    numTopics = args.numTopics
    epochs = args.epochs
    patience = args.patience
    learningrate = args.learningrate
    activation = args.activation
    conv_l2 = args.conv_l2
    dense_l2 = args.dense_l2
    conv_do = args.convDO
    dense_do = args.denseDO
    seed = args.seed
    utils.set_seed(seed)
    seqlen = args.seqlen
    batchsize = args.batchsize
    stride = args.stride
    motifwidth = args.motifwidth
    numkernels = args.numkernels
    maxpoolsize = args.maxpoolsize
    numdense = args.numdense
    insertmotif = args.insertmotif
    usetransformer = True if args.usetransformer == "True" else False
    selectedtopics = args.selectedtopics
    selectedtopicsShap = args.selectedtopicsShap
    runType = args.runType
    namehdf5 = args.namehdf5
    outputdirc = args.outputdirc
    clobber = True
    model_type = args.model
    shapRegions = args.shapRegions

    if wandbname != "False":
        import wandb
        from wandb.keras import WandbCallback

        hyperparameter_defaults = dict(
            numTopics=numTopics,
            epochs=epochs,
            patience=patience,
            activation=activation,
            usetransformer=usetransformer,
            learningrate=learningrate,
            conv_l2=conv_l2,
            seed=seed,
            seqlen=seqlen,
            motifwidth=motifwidth,
            numkernels=numkernels,
            maxpoolsize=maxpoolsize,
            numdense=numdense,
            batchsize=batchsize,
            inputtopics=inputtopics,
            consensuspeaks=consensuspeaks,
            genome=genome,
            chrSize=chrSize,
            validchroms=validchroms,
            testchroms=testchroms,
            transferlearn=transferlearn,
            doubleDATArc=doubleDATArc,
            wandbUser=wandbUser,
            wandbname=wandbname,
            mergetopic=mergetopic,
            useclassweight=useclassweight,
            useCreatedData=useCreatedData,
            gpuname=gpuname,
            stride=stride,
            insertmotif=insertmotif,
            selectedtopics=selectedtopics,
            selectedtopicsShap=selectedtopicsShap,
            namehdf5=namehdf5,
            outputdirc=outputdirc,
            runType=runType
        )
        if wandbUser != False:
            #wandb.init(settings=wandb.Settings(start_method='fork'))
            wandb.init(project=wandbname, entity=wandbUser, config=hyperparameter_defaults)
        else:
            wandb.init(project=wandbname, config=hyperparameter_defaults)
        config = wandb.config

    selected_topics = np.array(list(range(numTopics)))


    if gpuname != "False":
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # The GPU id to use, usually either "0" or "1";
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpuname)
        # Do other imports now...

    topic_dict = {}
    num_classes = len(selected_topics)
    for i in range(len(selected_topics)):
        topic_dict[i] = 'Topic_' + str(selected_topics[i] + 1)

    try:
        # OUTPUT=WORKDIR
        os.makedirs(outputdirc)
    except OSError as exc:
        if exc.errno == errno.EEXIST:
            if not clobber:
                print('output directory ({0:s}) already exists but you specified not to clobber it'.format(outputdirc),
                      file=sys.stderr)
                sys.exit(1)
            else:
                print('output directory ({0:s}) already exists so it will be clobbered'.format(outputdirc),
                      file=sys.stderr)

    utils.set_pybedtool_temp(outputdirc)

    topic_dict_onlynum = {}
    topic_dict_onlyname = {}
    for key in topic_dict:
        topic_dict_onlynum[key] = topic_dict[key].split(' ')[0].split('_')[1]
        topic_dict_onlyname[key] = topic_dict[key].split(' ')[0]
    topic_list = []
    for i in range(len(topic_dict)):
        topic_list.append(topic_dict[i])

    PATH_TO_SAVE_ARC = os.path.join(outputdirc, 'model.json')
    # PATH_TO_SAVE_BEST_LOST_WEIGHTS = os.path.join(output_dir, 'model_best_loss')
    # PATH_TO_SAVE_BEST_ACC_WEIGHTS = os.path.join(output_dir, 'model_best_acc')
    PATH_TO_SAVE_EVERY_EPOCH_WEIGHTS = os.path.join(outputdirc, 'model_epoch')

    seq_shape = (seqlen, 4)

    # useCreatedData
    f = open(os.path.join(useCreatedData+'nonAugmented_data_dict.pkl'), "rb")
    nonAugmented_data_dict = pickle.load(f)
    augmented_data_dict= nonAugmented_data_dict
    f.close()

    print("Compile model...")
    model = utils.build_model(num_classes, motifwidth, numkernels, maxpoolsize, numdense, activation,
                learningrate, seq_shape, insertmotif, conv_l2, dense_l2, conv_do, dense_do, 
                usetransformer, model_type)

    if runType == "train":
        callbacks_list = []

        # Save model architecture (model type)
        with open(PATH_TO_SAVE_ARC, "w") as json_file:
            json_file.write(model.to_json())
        checkpoint = tf.keras.callbacks.ModelCheckpoint(PATH_TO_SAVE_EVERY_EPOCH_WEIGHTS + '_{epoch:02d}.hdf5', save_freq='epoch')
        
        callbacks_list.append(checkpoint)
        
        # Early stopping
        early_stop_metric = 'val_pearson_correlation' if wandbname != "False" else 'val_mean_absolute_error'
        early_stop = tf.keras.callbacks.EarlyStopping(monitor=early_stop_metric, patience=patience, mode='max')
        callbacks_list.append(early_stop)
        
        # Learning rate adjustment
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor=early_stop_metric, factor=0.25, patience=3, min_lr=0.000001, mode='max')
        callbacks_list.append(reduce_lr)
        
        if wandbname != "False":
            callbacks_list.append(WandbCallback())

        # Double (augmentation)
        if doubleDATArc == "True":
            for key in ["train", "valid", "test"]:
                augmented_data_dict[key+"_data"] = np.concatenate((augmented_data_dict[key+"_data"],
                                                                   augmented_data_dict[key+"_data"][:, ::-1, ::-1]),
                                                                  axis=0)
                augmented_data_dict["y_"+key] = np.concatenate((augmented_data_dict["y_"+key],
                                                                augmented_data_dict["y_"+key]),
                                                               axis=0)

        print("Training model...")
        print(augmented_data_dict["train_data"].shape)
        print(augmented_data_dict["y_train"].shape)
        
        model.summary()
        class_weights = utils.class_weights(augmented_data_dict["y_train"]) if useclassweight != "False" else None 
        sw = False

        sample_weights= None
        history = model.fit(augmented_data_dict["train_data"], augmented_data_dict["y_train"],
                            epochs=epochs, batch_size=batchsize, shuffle=True,
                            validation_data=(nonAugmented_data_dict["valid_data"], nonAugmented_data_dict["y_valid"]),
                            verbose=1, callbacks=callbacks_list, class_weight=class_weights, sample_weight = sample_weights)
    
        try:
            utils.create_plots(history, outputdirc)
        except:
            print("Unexpected error on plot")

        print("\nEvaluating test samples")
        n_epochs = len(history.history['loss'])
        if(n_epochs - patience < 10):
            ep_str = 'model_epoch_0'+str(n_epochs-patience)+'.hdf5'
        else:
            ep_str = 'model_epoch_'+str(n_epochs-patience)+'.hdf5'
        model = utils.load_model(outputdirc, ep_str)
        optimizer = tf.keras.optimizers.Adam(lr=learningrate)
        model.compile(optimizer=optimizer,
                              loss='binary_crossentropy',
                              metrics=[[tf.keras.metrics.AUC(num_thresholds=200, curve='ROC', summation_method='interpolation', name='auROC', thresholds=None, multi_label=True, label_weights=None), tf.keras.metrics.AUC(num_thresholds=200, curve='PR', summation_method='interpolation', name='auPR', thresholds=None, multi_label=True, label_weights=None)]])
        loss, roc, pr = model.evaluate(
            nonAugmented_data_dict["test_data"], nonAugmented_data_dict["y_test"], batch_size=128)
        print("Loss on test samples: ", loss)
        print("auROC on test samples: ", roc)
        print("auPR on test samples: ", pr)
        print("Successfully trained model ヽ(^o^)ノ")


if __name__ == "__main__":
    main()
