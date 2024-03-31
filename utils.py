import os
import pickle
import glob
import json
import shutil
import pathlib
import numpy as np 
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, Flatten

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, classification_report

import warnings
warnings.filterwarnings("ignore")
np.random.seed(42)
tf.random.set_seed(42)

def analyse_incorrect(actual, pred, results_df, data_dir):
    """
    Function to display and analyze incorrect predictions.

    Inputs:
    - actual: The actual label of the incorrectly predicted image.
    - pred: The predicted label for the image.
    - results_df: DataFrame containing the results (actuals and predictions).
    - data_dir: Directory where the image files are located.
    """
    # Filter out incorrect results for the given actual and predicted labels
    incorrect_res = results_df[(results_df['Actuals'] == actual) & (results_df['Preds'] == pred)]

    # Iterate over each file name in the filtered incorrect results
    for file_name in incorrect_res["File"].values:
        print(file_name)
        plt.imshow(cv2.imread(data_dir+file_name)) # Load image using OpenCV
        plt.show()

def draw_confusion_matrix(actual, pred):
    """
    Function to plot the confusion matrix and print the F1 score.

    Parameters:
    - actual: The actual labels.
    - pred: The predicted labels.
    """
    cm = confusion_matrix(actual, pred)
    plt.figure(figsize=(2, 2))
    sns.heatmap(cm, annot=True, vmin=0, fmt='g', cmap='Reds', cbar=False)       
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
    print(f1_score(actual, pred, average="micro"))
    

def del_aug_data(data_dir):
    """Deletes Augmented Data. Used to start with original data only."""
    path = pathlib.Path(data_dir)
    shutil.rmtree(path, ignore_errors=True)

def get_data(train_path, test_path, val_path, target_labels):
    """Function Used to read the data from the path and returns them as Dataframes"""
    train_data = glob.glob("**/*.png", root_dir=train_path, recursive=True)
    test_data = glob.glob("**/*.png", root_dir=test_path, recursive=True)
    val_data = glob.glob("**/*.png", root_dir=val_path, recursive=True)
    train_df = pd.DataFrame({"images":train_data})
    train_df['target_label'] = train_df.images.apply(lambda x: target_labels[int(x.split('\\')[0])])
    test_df = pd.DataFrame({"images":test_data})
    test_df['target_label'] = test_df.images.apply(lambda x: target_labels[int(x.split('\\')[0])])
    val_df = pd.DataFrame({"images":val_data})
    val_df['target_label'] = val_df.images.apply(lambda x: target_labels[int(x.split('\\')[0])])
    
    print(f"Number of Images in train {train_df.shape[0]}")
    print(f"Target Class Distribution in train {train_df.target_label.value_counts()} \n")    
    print(f"Number of Images in test {test_df.shape[0]}")
    print(f"Target Class Distribution in test {test_df.target_label.value_counts()} \n")    
    print(f"Number of Images in val {val_df.shape[0]}")
    print(f"Target Class Distribution in val {val_df.target_label.value_counts()} \n")  
    
    return train_df, test_df, val_df


def augment_dataset(data_df, data_dir, image_size, thresh):
    """
    Selective Augmentation function, used to add new augmented images based on the threshold.

    Parameters:
    - data_df: DataFrame containing image information, including target labels.
    - data_dir: Directory where the image files are located.
    - image_size: Tuple specifying the target size of the images (e.g., (224, 224)).
    - thresh: Threshold indicating the minimum number of images required for augmentation.
    """

    ## Keras Image Data Generator
    generator = ImageDataGenerator(
        rotation_range=10,
        height_shift_range=0.15,
        horizontal_flip=True,
        vertical_flip=False,
        brightness_range=(0.2,0.5),
        fill_mode='nearest')
    
    ## Grouping by Target Label
    target_groups = data_df.groupby('target_label')
    target_label_counts = list(data_df['target_label'].value_counts().items())
    target_label_counts.sort()
    max_count = target_label_counts[0][1]
    
    ## Loop through each target group and check if the number of images are atleast the given threshold.
    ## If not, required number of images are added.
    for target, count in target_label_counts[1:]:
        group = target_groups.get_group(target)
        req_count = int(max_count*thresh) - count
        aug_dir = data_dir+"/"+target.split(":")[0]+"/aug"
        if not os.path.exists(aug_dir): os.mkdir(aug_dir)
        augmented_image_gen = generator.flow_from_dataframe(group,directory=data_dir, x_col='images', y_col=None, 
                                                            target_size=image_size,
                                                        class_mode=None, batch_size=1, shuffle=False, 
                                                        save_to_dir=aug_dir, save_prefix='aug', color_mode='rgb',
                                                        save_format='png')
        print(f"Adding {req_count} images to {aug_dir}")
        # Generate and save augmented images until the required count is reached
        while req_count > 0:
            images=next(augmented_image_gen)            
            req_count -= len(images)
        max_count = int(max_count*thresh)


def create_data_gen(data_df, data_dir, data_gen, image_size, batch_size, shuffle):
    """Function to create a Data Generator"""
    data = data_gen.flow_from_dataframe(data_df,data_dir, 
                                          x_col='images', y_col="target_label",
                                          target_size=image_size,
                                          batch_size=batch_size,
                                          class_mode='categorical',
                                          color_mode='rgb',
                                          shuffle=shuffle,
                                          seed = 123)
    return data


def get_cm_plot(y_true, y_pred, title, target_labels, figsize=(8, 8)):
    """Function to return Confusion Matrix pplot, without plt.show()"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, vmin=0, fmt='g', cmap='Reds', xticklabels=target_labels, yticklabels=target_labels, cbar=False)   
    plt.title(title)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    return plt


def run_prediction_save_metrics(model, data_gen, title, plot_metrics=False, return_preds=True):
    """Fucntion get the predictions and Save the Evaluation Metrics."""
    evaluation_dict = {}
    y_pred_scores = model.predict(data_gen)
    y_pred = y_pred_scores.argmax(axis=1)
    y_true = data_gen.labels
    targets = data_gen.class_indices.keys()

    if plot_metrics:
        plt = get_cm_plot(y_true, y_pred, title, targets, figsize=(3,3))
        plt.show()
        plt.close()
    
    
    # Computing the Evaulation metric scores.
    curr_f1_score = round(f1_score(y_pred, y_true, average='weighted'),3)
    curr_precision_score = round(precision_score(y_pred, y_true, average='weighted'), 3)
    curr_recall_score = round(recall_score(y_pred, y_true, average='weighted'), 3)
    curr_accuracy_score = round(accuracy_score(y_pred, y_true), 3)

    # Display the computed Evaulation metric scores.
    print(f"F1 Score {curr_f1_score}")
    print(f"Precision Score {curr_precision_score}")
    print(f"Recall Score {curr_recall_score}")
    print(f"Accuracy Score {curr_accuracy_score}")
    print(f"Classification Metrics \n{classification_report(y_pred, y_true)}")

    # Return the computed Evaulation metric scores.
    evaluation_dict.update({"f1_score" : curr_f1_score,
            "precision_score" : curr_precision_score,
            "recall_score" : curr_recall_score,
            "accuracy_score": curr_accuracy_score})
    
    if return_preds: 
        evaluation_dict["y_pred"] = y_pred
        evaluation_dict["y_true"] = y_true
    evaluation_dict["title"] = title
        
    return evaluation_dict


def create_model(transfer_model, image_size, num_classes, new_layers_list, freeze_layers=True, chkp_weights=None):
    """
    Function to create a model using transfer learning and additional layers.

    Parameters:
    - transfer_model: Transfer learning model (e.g., ResNet, VGG, etc.).
    - image_size: Tuple specifying the input image size (e.g., (224, 224)).
    - num_classes: Number of classes for classification.
    - new_layers_list: List of new layers to add after the transfer model's output.
    - freeze_layers: Boolean indicating whether to freeze the layers of the transfer model.
    - chkp_weights: Path to checkpoint weights for initializing the model.

    Returns:
    - model: Compiled Keras model for image classification.
    """

    # Create the base model using the transfer learning model
    base_model = transfer_model(input_shape=tuple(image_size + [3]),
                            include_top=False,
                            weights='imagenet'
                           )

    # Freeze the layers of the base model if specified
    if freeze_layers:
        for layer in base_model.layers:
            layer.trainable = False  

    # Define input and output layers for the model
    input_layer = base_model.input
    x = Flatten()(base_model.output)
    
    # Add new layers from the new_layers_list
    for each_layer in new_layers_list:
        x = each_layer(x)

    # Add a final output layer for classification
    output_layer = Dense(num_classes, activation='softmax')(x)
    model = Model(input_layer, output_layer)
    
    # Load checkpoint weights if provided
    if chkp_weights: model.load_weights(chkp_weights)

    # Compile the model with Adam optimizer and categorical crossentropy loss
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def fit_model(model, tr_data, vl_data, epochs, callbacks, iter_training = False, class_weight=None):
    """Function to Fit the model"""
    hist = model.fit(tr_data,
          epochs=epochs,
          validation_data=vl_data,
          callbacks=callbacks,
          verbose=1,class_weight=class_weight)
    return model,hist


def fit_model_iterative(model, tr_data, vl_data, ts_data, epochs_per_iters, new_layers_count, layers_to_unfreeze, callbacks):
    """
    Function to fit the model iteratively, unfreezing layers progressively.

    Parameters:
    - model: Keras model to be trained.
    - tr_data: Training data (e.g., ImageDataGenerator or DataLoader).
    - vl_data: Validation data (e.g., ImageDataGenerator or DataLoader).
    - ts_data: Test data (e.g., ImageDataGenerator or DataLoader).
    - epochs_per_iters: Number of epochs to train per iteration.
    - new_layers_count: Number of new layers added after transfer learning.
    - layers_to_unfreeze: Number of layers to unfreeze iteratively.
    - callbacks: List of Keras callbacks for training.

    Returns:
    - model: Trained Keras model.
    - iters: Dictionary containing metrics and history for each iteration.
    """

    iters = {}
    
    ## First Layer to Unfreeze, It will be the last layer in transfer learning block.
    start_layer = -new_layers_count-3
    while layers_to_unfreeze >= 0:
        # Check if the layers to unfreeze threshold is reached.
        print(layers_to_unfreeze)
        print(f"Checking {model.layers[start_layer].name} Layer")
        if model.layers[start_layer].count_params() != 0:
            ## Check if layer has params, We want to unfreeze only trainable layers.
            print(f"Releasing {model.layers[start_layer].name} Layer")
            model.layers[start_layer].trainable = True
            hist = model.fit(tr_data,
                            epochs=epochs_per_iters,
                            validation_data=vl_data,
                            callbacks=callbacks,
                            verbose=2)
            
              
            ## Check the Metrics
            train_preds = model.predict(tr_data)
            val_preds = model.predict(vl_data)
            test_preds = model.predict(ts_data)
            
            train_f1_score = round(f1_score(tr_data.labels,train_preds.argmax(axis=1),average='weighted'), 3)
            val_f1_score =  round(f1_score(vl_data.labels,val_preds.argmax(axis=1),average='weighted'), 3)
            test_f1_score =  round(f1_score(ts_data.labels,test_preds.argmax(axis=1),average='weighted'), 3)
            draw_confusion_matrix(ts_data.labels, test_preds.argmax(axis=1))

            iters[model.layers[start_layer].name] = {
                     "params":hist.params, 
                     "history":hist.history, 
                     "train_f1_score":train_f1_score, 
                     "val_f1_score": val_f1_score,
                     "test_f1_score": test_f1_score
                    }
            
            print("train f1_score", train_f1_score)
            print("val f1_score", val_f1_score)
            print("test f1_score", test_f1_score)

            layers_to_unfreeze -= 1
            print(f"Layers left to be released {layers_to_unfreeze}")
        start_layer -= 1

    return model, iters

def save_exp(save_dir, exp_id, model, desc, eval_dict, targets, model_name=None, training_hist=None):
    ## Save the experiments, Model, History and plots.
    p = os.path.join(save_dir,exp_id+'/')
    if not os.path.exists(p): os.mkdir(p)
        
    cm_plot = get_cm_plot(eval_dict['y_true'], eval_dict['y_pred'], eval_dict['title'], targets)
    
    temp = eval_dict.pop('y_pred', None)
    temp = eval_dict.pop('y_true', None)
    
    cm_plot.savefig(p+eval_dict['title']+".png", dpi=800, bbox_inches="tight", pad_inches=0.5)
    cm_plot.close()
    
    eval_dict['description'] = desc


    if model_name: 
        model.save(p+model_name+".h5")
        
        hist = {
            "params":training_hist.params, 
            "history":training_hist.history}

    
        with open(p+model_name+'_hist.pkl', 'wb') as handle:
            pickle.dump(hist, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(p+eval_dict['title']+".json", "w") as outfile:
        json.dump(eval_dict, outfile, indent=4)