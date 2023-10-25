import os
import argparse
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import utils

def make_argument_parser():
    parser = argparse.ArgumentParser(description="Train model.")
    parser.add_argument('--inputtopics', '-i', type=str, required=True, help='Folders containing all topic.bed files.')
    parser.add_argument('--numTopics', '-nt', type=int, required=True, help='Number of topics resulted from cisTopic.')
    # Add other relevant arguments here...

    return parser

def build_model(input_shape, num_classes, activation, learning_rate, conv_l2, dense_l2, conv_dropout, dense_dropout):
    model = models.Sequential()
    # Add your model architecture here...
    # Example:
    model.add(layers.Conv1D(64, 3, activation=activation, input_shape=input_shape, kernel_regularizer=tf.keras.regularizers.l2(conv_l2)))
    model.add(layers.MaxPooling1D(2))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation=activation, kernel_regularizer=tf.keras.regularizers.l2(dense_l2)))
    model.add(layers.Dropout(dense_dropout))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model

def main():
    parser = make_argument_parser()
    args = parser.parse_args()

    # Extract arguments
    inputtopics = args.inputtopics
    numTopics = args.numTopics
    # Extract other arguments...

    # Set random seed for reproducibility
    utils.set_seed(args.seed)

    # Load and preprocess your data here...

    # Build the model
    input_shape = (args.seqlen, 4)
    num_classes = numTopics
    model = build_model(input_shape, num_classes, args.activation, args.learningrate, args.conv_l2, args.dense_l2, args.convDO, args.denseDO)

    # Compile the model
    optimizer = tf.keras.optimizers.Adam(lr=args.learningrate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    # Define callbacks
    callbacks_list = [
        ModelCheckpoint(filepath='model_epoch_{epoch:02d}.hdf5', save_freq='epoch'),
        EarlyStopping(monitor='val_loss', patience=args.patience, mode='min'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.25, patience=3, min_lr=0.000001, mode='min'),
    ]

    # Train the model
    history = model.fit(
        x=your_training_data,
        y=your_training_labels,
        epochs=args.epochs,
        batch_size=args.batchsize,
        validation_data=(your_validation_data, your_validation_labels),
        callbacks=callbacks_list,
    )

    # Evaluate the model on test data
    loss, accuracy = model.evaluate(x=your_test_data, y=your_test_labels, batch_size=args.batchsize)
    print("Loss on test samples:", loss)
    print("Accuracy on test samples:", accuracy)

if __name__ == "__main__":
    main()
