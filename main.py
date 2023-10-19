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

    if selectedtopics == "False":
        selected_topics = np.array(list(range(numTopics)))
    else:
        selected_topics = utils.get_selected_topic_from_text(selectedtopics)

    if gpuname != "False":
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # The GPU id to use, usually either "0" or "1";
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpuname)
        # Do other imports now...

    topic_dict = {}
    if mergetopic != "False":
        mergetopic = utils.get_merged_topic_from_text(mergetopic)
        num_classes = len(mergetopic)+1
        for i in range(len(mergetopic)+1):
            topic_dict[i] = 'Topic_' + str(i + 1)
    else:
        num_classes = len(selected_topics)
        for i in range(len(selected_topics)):
            topic_dict[i] = 'Topic_' + str(selected_topics[i] + 1)

    try:
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
    if useCreatedData != "False":
        f = open(os.path.join(useCreatedData+'nonAugmented_data_dict.pkl'), "rb")
        nonAugmented_data_dict = pickle.load(f)
        augmented_data_dict= nonAugmented_data_dict
        f.close()
        
    else:
        print('Prepare data from summits...')
        centered_bed = utils.prepare_data_from_summits(
            consensuspeaks, seqlen, stride=stride, augment=False, chr_size_file=chrSize)
        centered_bed_train, centered_bed_valid, centered_bed_test = utils.valid_test_split_wrapper(centered_bed,
                                                                                                   validchroms, testchroms)
        print('Prepare data from topics...')                                  
        centered_bed_withName = utils.prepare_data_from_topics(
            outputdirc, inputtopics, centered_bed, genome, name="nonAugmented.bed")
        centered_bed_train_withName = utils.prepare_data_from_topics(
            outputdirc, inputtopics, centered_bed_train, genome, name="nonAugmented_train.bed")
        centered_bed_valid_withName = utils.prepare_data_from_topics(
            outputdirc, inputtopics, centered_bed_valid, genome, name="nonAugmented_valid.bed")
        centered_bed_test_withName = utils.prepare_data_from_topics(
            outputdirc, inputtopics, centered_bed_test, genome, name="nonAugmented_test.bed")

        print('Printing fasta...')
        utils.printFasta(genome, centered_bed_withName, os.path.join(
            outputdirc, 'nonAugmented.fa'))
        utils.printFasta(genome, centered_bed_train_withName, os.path.join(
            outputdirc, 'nonAugmented_train.fa'))
        utils.printFasta(genome, centered_bed_valid_withName, os.path.join(
            outputdirc, 'nonAugmented_valid.fa'))
        utils.printFasta(genome, centered_bed_test_withName, os.path.join(
            outputdirc, 'nonAugmented_test.fa'))

        print('Prepare data from summits p2...')
        augmented_bed_train = utils.prepare_data_from_summits(
            centered_bed_train_withName, seqlen,
            stride=stride, augment=True, isfilepath=False, chr_size_file=chrSize)
        augmented_bed_valid = utils.prepare_data_from_summits(
            centered_bed_valid_withName, seqlen,
            stride=stride, augment=True, isfilepath=False, chr_size_file=chrSize)
        augmented_bed_test = utils.prepare_data_from_summits(
            centered_bed_test_withName, seqlen,
            stride=stride, augment=True, isfilepath=False, chr_size_file=chrSize)
        print('Prepare data from topics p2...')
        _ = utils.prepare_data_from_topics(
            outputdirc, inputtopics, augmented_bed_train, genome, name="augmented_train.bed")
        _ = utils.prepare_data_from_topics(
            outputdirc, inputtopics, augmented_bed_valid, genome, name="augmented_valid.bed")
        _ = utils.prepare_data_from_topics(
            outputdirc, inputtopics, augmented_bed_test, genome, name="augmented_test.bed")

        print('Prepare data with labels (nonAugmented)...')
        nonAugmented_data, nonAugmented_y, nonAugmented_ids = utils.prepare_data_withlabel(
            os.path.join(outputdirc, 'summit_to_topic_nonAugmented.fa'), selected_topics, numTopics, mergetopic)
        train_data, y_train, ids_train = utils.prepare_data_withlabel(
            os.path.join(outputdirc, 'summit_to_topic_nonAugmented_train.fa'), selected_topics, numTopics, mergetopic)
        valid_data, y_valid, ids_valid = utils.prepare_data_withlabel(
            os.path.join(outputdirc, 'summit_to_topic_nonAugmented_valid.fa'), selected_topics, numTopics, mergetopic)
        test_data, y_test, ids_test = utils.prepare_data_withlabel(
            os.path.join(outputdirc, 'summit_to_topic_nonAugmented_test.fa'), selected_topics, numTopics, mergetopic)

        nonAugmented_data_dict = {"nonAugmented_data": nonAugmented_data, "nonAugmented_y": nonAugmented_y,
                                  "nonAugmented_ids": nonAugmented_ids,
                                  "train_data": train_data, "y_train": y_train, "ids_train": ids_train,
                                  "valid_data": valid_data, "y_valid": y_valid, "ids_valid": ids_valid,
                                  "test_data": test_data, "y_test": y_test, "ids_test": ids_test}
        f = open(os.path.join(outputdirc, 'nonAugmented_data_dict.pkl'), "wb")
        pickle.dump(nonAugmented_data_dict, f)
        f.close()


        print('Prepare data with labels (augmented, train)...')
        train_data, y_train, ids_train = utils.prepare_data_withlabel(
            os.path.join(outputdirc, 'summit_to_topic_augmented_train.fa'), selected_topics, numTopics, mergetopic)
        print('Prepare data with labels (augmented, valid)...')
        valid_data, y_valid, ids_valid = utils.prepare_data_withlabel(
            os.path.join(outputdirc, 'summit_to_topic_augmented_valid.fa'), selected_topics, numTopics, mergetopic)
        print('Prepare data with labels (augmented, test)...')
        test_data, y_test, ids_test = utils.prepare_data_withlabel(
            os.path.join(outputdirc, 'summit_to_topic_augmented_test.fa'), selected_topics, numTopics, mergetopic)


        print('Making dictionary')
        augmented_data_dict = {"train_data": train_data, "y_train": y_train, "ids_train": ids_train,
                               "valid_data": valid_data, "y_valid": y_valid, "ids_valid": ids_valid,
                               "test_data": test_data, "y_test": y_test, "ids_test": ids_test}
        f = open(os.path.join(outputdirc, 'augmented_data_dict.pkl'), "wb")
        pickle.dump(augmented_data_dict, f)
        f.close()

    print("Compile model...")
    model = utils.build_model(num_classes, motifwidth, numkernels, maxpoolsize, numdense, activation,
                learningrate, seq_shape, insertmotif, conv_l2, dense_l2, conv_do, dense_do, 
                usetransformer, model_type)

    if runType == "train":
        if transferlearn != "False":
            base_model_output_dir = transferlearn[:-19] #Temporary solution, hardcoded sadly
            model_json_file = open(os.path.join(base_model_output_dir, 'model.json'))
            model_json = model_json_file.read()
            base_model = tf.keras.models.model_from_json(model_json)
            base_model.load_weights(transferlearn)
            model = utils.build_transferlearning_model_deeppeak(base_model, num_classes, seq_shape=(2114,4))
            patience=3
            print("\n\n\n============================\n\n\n Transfer learning from topic model \n\n\n============================\n")
        
        callbacks_list = []

        if model_type == "enformer":
            checkpoint = tf.keras.callbacks.ModelCheckpoint(PATH_TO_SAVE_EVERY_EPOCH_WEIGHTS + '_{epoch:02d}.h5', save_freq='epoch')
        else:
            # Save model architecture
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

        if transferlearn != "False":
            callbacks_list = [checkpoint, early_stop, WandbCallback()]
            #model.load_weights(transferlearn)
            
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
        if sw:
            sample_weights = utils.get_sample_weights(outputdirc, augmented_data_dict['ids_train'], '/staging/leuven/stg_00002/lcb/nkemp/mouse/exc/logFC.pkl')
            if doubleDATArc == "True":
                sample_weights = np.concatenate((sample_weights,  sample_weights), axis=0)
            sample_weights = pd.Series(sample_weights)
        else:
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

    elif runType == "prepare_CV_file":
        utils.prepare_CV_file(outputdirc, inputtopics, consensuspeaks, seqlen, stride, genome, chrSize, selected_topics,
                              numTopics, mergetopic)

    elif runType == "analyze":

        # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        print('Loading model...')
        model = utils.load_model(outputdirc, namehdf5)

        print('Loading data...')
        f = open(os.path.join(useCreatedData), "rb")
        nonAugmented_data_dict = pickle.load(f)
        f.close()

        print('Calculating ROC and PR...')
        #utils.analyze_rocpr(
            #model, nonAugmented_data_dict, outputdirc, num_classes, topic_dict_onlynum, topic_dict_onlyname)

        print('Generating Motifs...')
        #utils.analyze_motif(model, motifwidth, outputdirc)

        #os.environ["CUDA_VISIBLE_DEVICES"] = str(gpuname)

        tf.keras.backend.clear_session()
        #tf.disable_eager_execution()
        #tf.compat.v1.disable_eager_execution()
        print('Loading model...')
        model = utils.load_model(outputdirc, namehdf5)

        model_rest = utils.build_model(
            num_classes, motifwidth, numkernels, maxpoolsize, numdense, activation,
            learningrate, seq_shape, insertmotif, conv_l2, dense_l2, conv_do, dense_do, 
            usetransformer, model_type="without_conv_maxp_v2", main_model=model)
        #model_rest.summary()
        #model.summary()
        #model.layers[1].summary()
        #model_rest.set_weights(model.get_weights()[6:])
        #print(model_rest.layers[23])
        #for i, layer in enumerate(model_rest.layers[23:28]):
            #print(layer)
            #print((layer.get_weights()[0].shape))
            #print(model.layers[2+i])#.get_weights()[6+i])
            #print((model.layers[2+i].get_weights()[0].shape))
            #model_rest.layers[23:28] = layer.set_weights(model.layers[2+i].get_weights())
        model_rest.layers[1:23] = [layer.set_weights(model.layers[1].layers[6+i].get_weights()) for i, layer in (enumerate(model_rest.layers[1:23]))]
        #model_rest.layers[:23].set_weights(model.layers[1].get_weights()[6:])
        model_rest.layers[23:-1] = [layer.set_weights(model.layers[2+i].get_weights()) for i, layer in enumerate(model_rest.layers[23:-1])]

        model_conv = utils.build_model(
            num_classes, motifwidth, numkernels, maxpoolsize, numdense, activation,
            learningrate, seq_shape, insertmotif, conv_l2, dense_l2, conv_do, dense_do, 
            usetransformer, model_type="conv_maxp_v2", main_model=model)
        #model_conv.summary()
        #print(model.layers[1].layers[1])
        model_conv.layers[1:] = [layer.set_weights(model.layers[1].layers[i+1].get_weights()) for i, layer in (enumerate(model_conv.layers[1:]))]
        #model_conv.set_weights(model.layers[1].get_weights()[:6])

        model_rest.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        #model_rest.set_weights(model.get_weights()[6:])
        #print(model_rest.layers[:22])
        #model_rest.layers[:23].set_weights(model.layers[1].get_weights()[6:])
        #model_rest.layers[23:].set_weights(model.layers[2:])

        intermediate_output_ = model_conv.predict(nonAugmented_data_dict["test_data"])

        loss_real = np.zeros(num_classes)
        new_intermediate_output_ = np.array(intermediate_output_, copy=True)
        predictions = model_rest.predict(new_intermediate_output_)

        print('calculate real...')
        for i in range(num_classes):
            print(i, end=',')
            sys.stdout.flush()
            y_true = tf.keras.backend.variable(nonAugmented_data_dict["y_test"][:, i])
            y_pred = tf.keras.backend.variable(predictions[:, i])
            loss_real[i] = tf.keras.backend.eval(tf.keras.losses.binary_crossentropy(y_true, y_pred))
            tf.keras.backend.clear_session()
        print('finished')

        print('calculate change...')
        loss = np.zeros((numkernels, num_classes))
        for i in tqdm(range(numkernels)):
            start_load_time = time.perf_counter()
            model = utils.load_model(outputdirc, namehdf5)
            model_rest = utils.build_model(
                num_classes, motifwidth, numkernels, maxpoolsize, numdense, activation,
                learningrate, seq_shape, insertmotif, conv_l2, dense_l2, conv_do, dense_do, 
                usetransformer, model_type="without_conv_maxp_v2", main_model=model)
            model_rest.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            model_rest.layers[1:23] = [layer.set_weights(model.layers[1].layers[6+i].get_weights()) for i, layer in (enumerate(model_rest.layers[1:23]))]
            model_rest.layers[23:-1] = [layer.set_weights(model.layers[2+i].get_weights()) for i, layer in enumerate(model_rest.layers[23:-1])]
            start_pred_time = time.perf_counter()
            print(i, end='---\n')
            sys.stdout.flush()
            new_intermediate_output_ = np.array(intermediate_output_, copy=True)
            new_intermediate_output_[:, :, i] = np.ones(new_intermediate_output_[:, :, i].shape) * np.mean(
                intermediate_output_[:, :, i])
            predictions = model_rest.predict(new_intermediate_output_)
            end_pred_time = time.perf_counter()
            for j in range(num_classes):
                print(j, end=',')
                sys.stdout.flush()
                tf.keras.backend.clear_session()
                y_true = tf.keras.backend.variable(nonAugmented_data_dict["y_test"][:, j])
                y_pred = tf.keras.backend.variable(predictions[:, j])
                loss[i, j] = tf.keras.backend.eval(tf.keras.losses.binary_crossentropy(y_true, y_pred))
            end_eval_time = time.perf_counter()
            print("time: ",
                  start_pred_time - start_load_time,
                  end_pred_time - start_pred_time,
                  end_eval_time - end_pred_time)
            sys.stdout.flush()
            tf.keras.backend.clear_session()
        np.savez(os.path.join(outputdirc, 'loss.npz'), loss_real=loss_real, loss=loss)
        print('finished')

        print('plot motif importance...')
        utils.analyze_print_importance(topic_dict, loss, loss_real, outputdirc)

    elif runType == "analyzeRocpr":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpuname)
        print('Loading model...')
        model = utils.load_model(outputdirc, namehdf5)

        print('Loading data...')
        f = open(os.path.join(useCreatedData), "rb")
        nonAugmented_data_dict = pickle.load(f)
        f.close()

        print('Calculating ROC and PR...')
        utils.analyze_rocpr(
            model, nonAugmented_data_dict, outputdirc, num_classes, topic_dict_onlynum, topic_dict_onlyname)

    elif runType == "analyzeGenerateMotif":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        print('Generating Motifs...')
        model = utils.load_model(outputdirc, namehdf5)
        utils.analyze_motif(model, motifwidth, outputdirc)

    elif runType == "analyzeGenerateShapValues":
        print('Calculate DeepExplainer Values...')
        tf.keras.backend.clear_session()
        #tf.disable_eager_execution()
        tf.compat.v1.disable_v2_behavior()
        tf.compat.v1.disable_eager_execution()
        model = utils.load_model(outputdirc, namehdf5)
        selectedtopics_shap = utils.get_selected_topic_from_text(selectedtopicsShap)
        #f = open(os.path.join(useCreatedData), "rb")
        #nonAugmented_data_dict = pickle.load(f)
        #f.close()
        utils.generateShapValues(model, selectedtopics_shap, nonAugmented_data_dict, outputdirc)
        
    #RunType = analyzeGenerateShapValuesFromFASTA -> same as before, but now you can specify the regions the model needs to use to calculate SHAP on, instead of using the top n regions from the entire dataset (thanks Nikolai)
    elif runType == "analyzeGenerateShapValuesFromFASTA":
        print('Calculate DeepExplainer Values...')
        tf.keras.backend.clear_session()
        #tf.disable_eager_execution()
        tf.compat.v1.disable_v2_behavior()
        tf.compat.v1.disable_eager_execution()
        model = utils.load_model(outputdirc, namehdf5)
        selectedtopics_shap = utils.get_selected_topic_from_text(selectedtopicsShap)

        # read fasta files
        list_fastafiles = []
        with open(shapRegions, "r") as f:
            line = f.readline().rstrip()
            list_fastafiles.append(line)

        print("Reading input fasta sequences ...\n")
        dict_shap_seqs = utils.simple_read_fastas(list_fastafiles)

        # encode as one-hot
        # also trim sequences to length
        lenSeq=2114
        print("Encoding sequences as onehot ...\n")
        onehot_input = np.array([utils.one_hot_encode_along_row_axis(seq[0:lenSeq]) for seq in dict_shap_seqs.values()]).squeeze(axis=1)
        print("done.\n")

        # generate background from nonAugmented data
        f = open(os.path.join(useCreatedData, 'nonAugmented_data_dict_2114.pkl'), "rb")
        nonAugmented_data_dict = pickle.load(f)
        f.close()

        # pick random background subset from nonAugmented data
        # background is estimated from intial input data (change ?)
        n_random_bg = 250
        rn = np.random.choice(nonAugmented_data_dict["valid_data"].shape[0], n_random_bg, replace=False)
        onehot_background = nonAugmented_data_dict["valid_data"][rn]

        n_pick=2500
        utils.generateShapValuesFromOneHot(model, selectedtopics_shap, onehot_input, onehot_background, outputdirc, n_pick, list(dict_shap_seqs.keys()) )

    else:
        print("Nothing selected")


if __name__ == "__main__":
    main()
