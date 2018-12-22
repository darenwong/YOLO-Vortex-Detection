# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 07:21:18 2018

@author: Daren
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl

def conv_layer(input, size_in, size_out, fs, name="conv"):
    """Apply convolution filter, batch normalisation and activation function onto the input
    
    Parameters
    ----------     
    input : tensor
        Input is usually the output of the previous layer 
    
    size_in : int
        Depth of input, usually equals to the number of filters used in the previous layer
            
    size_out : int
        Depth of output, also equals to the number of filters used in the current layer
    
    fs : int
        Size of convolutional filter
        
    name : str
        Name to be displayed on Tensorboard for the current layer
    
    Returns
    -------
    act : Tensor
        Tensor output of the layer after applying convolution filter, batch normalisation and activation function
    
    """  
    
    with tf.name_scope(name):
        # Initialise weights
        w = tf.Variable(tf.truncated_normal([fs, fs, size_in, size_out], stddev=0.0176777), name="W")
        
        # Apply convolutional filter
        conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding = 'SAME')
        
        # Set up and apply batch normalisation
        batch_mean, batch_var = tf.nn.moments(conv,[0])
        scale = tf.Variable(tf.ones([size_out]), name = 'Scale')
        beta = tf.Variable(tf.zeros([size_out]), name = 'Beta')
        epsilon = 10e-8
        BN = tf.nn.batch_normalization(conv, batch_mean, batch_var, beta, scale, epsilon)
        
        # Apply activation function
        act = tf.nn.leaky_relu(BN)
    return act

 
def logits_layer(input, size_in, size_out, fs, name="conv"):
    """Apply convolution filter onto the input.
    
    Parameters
    ----------     
    input : tensor
        Input is usually the output of the previous layer 
    
    size_in : int
        Depth of input, usually equals to the number of filters used in the previous layer
            
    size_out : int
        Depth of output, also equals to the number of filters used in the current layer
    
    fs : int
        Size of convolutional filter
        
    name : str
        Name to be displayed on Tensorboard for the current layer
    
    Returns
    -------
    act : Tensor
        Tensor output of the layer after applying convolution filter onto input
    
    """  
    with tf.name_scope(name):
        # Initialise weights and biases
        w = tf.Variable(tf.truncated_normal([fs, fs, size_in, size_out], stddev=0.0176777), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
        
        # Apply convolutional filter
        conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding = 'SAME')
        
        # Adding biases to convoluted result
        logits = tf.add(conv, b)
    return logits


def cnn_model(x, l, label_depth, keep_prob): 
    """Build the convolution neural network structure.
    
    Parameters
    ----------     
    x : tf placeholder for input data
        Input data has size (num, l, l, 2) where:
            num : number of samples
            
    l : int
        Size of velocity vector field
    
    label_depth : int
        Number of features in the label
        
    keep_prob : float
        Probability that a node is dropped out in a layer
    
    Returns
    -------
    logits : Tensor
        Tensor output of the neural network model
    
    """  
    # Reshape input data to the form of (num, l, l, 2) where:
    #   num : number of samples
    #   l : size of velocity field
    #   2 : velocity vector is in 2 directions, u and v
    x_input = tf.reshape(x, [-1, l, l, 2])

    # Apply convolutional layer, maxpool layer and logits layer
    conv1 = conv_layer(x_input, 2, 32, 5, "conv1")
    max_pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name = 'max_pool1')
    
    conv2 = conv_layer(max_pool1, 32, 64, 5, "conv2")
    max_pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name = 'max_pool2')
    
    conv3 = conv_layer(max_pool2, 64, 128, 5, "conv3")
    max_pool3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name = 'max_pool3')
    
    conv3_drop = tf.nn.dropout(max_pool3, keep_prob, name = 'Drop_Out3')
    
    conv4 = conv_layer(conv3_drop, 128, 256, 5, "conv4")
    
    conv5 = conv_layer(conv4, 256, 512, 5, "conv5")
    conv5_drop = tf.nn.dropout(conv5, keep_prob, name = 'Drop_Out5')
    
    conv6 = conv_layer(conv5_drop, 512, 512, 5, "conv6")
    conv6_drop = tf.nn.dropout(conv6, keep_prob, name = 'Drop_Out6')
    
    conv7 = conv_layer(conv6_drop, 512, 512, 5, "conv7")
    conv7_drop = tf.nn.dropout(conv7, keep_prob, name = 'Drop_Out7')
    
    conv8 = conv_layer(conv7_drop, 512, 256, 5, "conv8")
    
    conv9 = conv_layer(conv8, 256, 64, 5, "conv9")
    conv9_drop = tf.nn.dropout(conv9, keep_prob, name = 'Drop_Out9')
    
    label_depth = int(label_depth)
    logits = logits_layer(conv9_drop, 64, size_out = label_depth, fs = 6, name = "logits")
    
    return logits

def train_predict_nn(image_input, label, mode, save_status, save_model_num, restore_model_num, epoch, batch_size, drop_out_keep_prob):
    """Run training or vortex detection and prediction.
    
    Parameters
    ----------     
    image_input : numpy array of size (num, 2, l, l) where:
        num : number of samples
        l : size of velocity vector field
    
    label : numpy array of size (num, yolo_grid_size, yolo_grid_size, 6)
        Yolo_grid_size refers to the matrix size that the image is divided into for labelling (Refer to external documentation on how YOLO labelling works)
        Each element in the matrix contains 6 parameters: cp, bx, by, bw, bh, ba
        where cp : confidence probability (1 or 0) of whether a vortex is present in the grid cell
              bx, by : location of center of vortex,
              bw, bh : width and height of vortex,
              ba : angle at which the vortex is rotated by
              
    mode : str 
        Select mode to run the neural network model at (either 'train' or 'predict)
        
    save_status : bool 
        Select whether to restore previously saved model. (1 to restore model)
        (Only matters in 'train' mode, always set to 1 in 'predict' mode)
              
    save_model_num, restore_model_num : int
        Select the model to be saved or restored
        
    epoch : int
        Set the number of Epoch for training (Only matters in 'train' mode)
        
    batch_size : int
        Set the size of mini batch
        
    drop_out_keep_prob : float (between 0 and 1)
        Set the probability that a node is dropped out in a layer
    """  
    
    # Select mode
    if mode == 'predict':
        # In 'predict' mode, save_status is always set to 1 so that a restored model is used for prediction
        save_status = 1
        
        # Load sample images and labels to be included in test batch if size of test batch is smaller than 50
        # This is because the layers in the model use batch normalization which only works for big enough test batches
        image = np.load('Artificial data/Test samples/yolo_random_vortex_image_test_sample.npy')
        label = np.load('Artificial data/Test samples/yolo_random_vortex_label_test_sample.npy')
        
        # Set test batch size to 50
        test_size = 50
        
#        # Find actual input test batch size 
#        actual_test_size = image_input.shape[0]
#            
#        # Mix actual test batch with loaded samples if actual test batch size is small
#        if actual_test_size < test_size:
#            img = np.concatenate((image_input, image[:test_size-image_input.shape[0]]), axis = 0)
#        else:
#            img = image_input

        img = image

        # Generate image and input data array            
        data = []
        image = []
        
        for i in img:
            u = i[0]
            v = i[1]    
            
            k = (u,v)
            if abs(np.amax(k)) > abs(np.amin(k)):
                u = u/abs(np.amax(k))
                v = v/abs(np.amax(k))
            else:
                u = u/abs(np.amin(k))
                v = v/abs(np.amin(k))
                
            image.append([u,v])
            
            conc_data = np.zeros((u.shape[0], u.shape[1], 2))
            for ii in range(u.shape[0]):
                for jj in range(u.shape[1]):
                    conc_data[ii,jj] = (u[ii,jj], v[ii,jj])
        
            data.append((conc_data))

        # Convert list to numpy array     
        image = np.array(image)
        data = np.array(data)
    
        # Find the size of image
        l = image.shape[2]
        
        # Reshape/Flatten data before inputting to neural network model 
        data = data.reshape((data.shape[0], l*l, 2))
    
    if mode == 'train':
        # Load image data 
        image = image_input
        data = []
        
        # Generate input data array
        for i in image:
            u = i[0]
            v = i[1]    
            
            conc_data = np.zeros((u.shape[0], u.shape[1], 2))
            for ii in range(u.shape[0]):
                for jj in range(u.shape[1]):
                    conc_data[ii,jj] = (u[ii,jj], v[ii,jj])
        
            data.append((conc_data))
            
        # Convert list to numpy array
        image = np.array(image)
        data = np.array(data)
        
        # Find the size of image
        l = image.shape[2]
        
        # Reshape/Flatten data before inputting to neural network model 
        data = data.reshape((data.shape[0], l*l, 2))
        
        # Divide input data and label to training set and evaluation set
        train_size = int(0.99*data.shape[0])
        train_data = data[:train_size]
        train_labels = label[:train_size]
        
        eval_size = int(0.01*data.shape[0])
        eval_data = data[train_size: train_size + eval_size]
        eval_labels = label[train_size: train_size + eval_size]   
    
    # Find the number of features in a label. Eg, 6 including cp, bx, by, bw, bh, ba
    label_depth = label.shape[3]
    
    # Find the YOLO grid size used in labellng. Eg, (10,10)
    label_grid_size = label.shape[2]
    
    # Find the size of image. Eg, 80 for an 80x80 image
    img_size = image.shape[2]
    
    # Determine the size of a single grid cell. For eg, if grid is 10x10 and image is 80x80, then each grid cell will be 8x8.
    grid_size = img_size/label_grid_size
    
    # Initialising placeholders for input data, label and probability of whether a note should be kept in a layer 
    x = tf.placeholder(tf.float32, [None, l*l, 2], name = 'Input')
    y = tf.placeholder(tf.float32, [None, label_grid_size, label_grid_size, 6], name = 'labels')
    keep_prob = tf.placeholder(tf.float32)
    
    # Find the number of image samples contained in the input file
    num_image = tf.shape(x, out_type = tf.int32)[0]
    
    # Compute the logits output of the neural network
    logits = cnn_model(x, l, label_depth, keep_prob)
    
    with tf.name_scope("loss"):
        # Separate logit ouput according to their features
        class_logits = tf.slice(logits, [0,0,0,0], [num_image, label_grid_size, label_grid_size, 1])
        xy_regression_logits = tf.slice(logits, [0,0,0,1], [num_image, label_grid_size, label_grid_size, 2])
        wh_regression_logits = tf.sqrt(tf.abs(tf.slice(logits, [0,0,0,3], [num_image, label_grid_size, label_grid_size, 2])))
        ang_regression_logits = tf.slice(logits, [0,0,0,5], [num_image, label_grid_size, label_grid_size, 1])
        
        # Separate label according to their features
        class_label = tf.slice(y, [0,0,0,0], [num_image, label_grid_size, label_grid_size, 1])
        xy_regression_label = tf.slice(y, [0,0,0,1], [num_image, label_grid_size, label_grid_size, 2])
        wh_regression_label = tf.sqrt(tf.abs(tf.slice(y, [0,0,0,3], [num_image, label_grid_size, label_grid_size, 2])))
        ang_regression_label = tf.slice(y, [0,0,0,5], [num_image, label_grid_size, label_grid_size, 1])
        
        # Compute loss for confidence probability (cp) using sigmoid cross entropy activation function
        loss_class_logits = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = class_logits, labels = class_label))

        # Compute loss for regression feature (bx, by, bw, bh, ba) using mean squared difference
        loss_xy_regression_logits = tf.reduce_mean(tf.squared_difference(tf.multiply(xy_regression_logits, class_label), xy_regression_label))
        loss_wh_regression_logits = tf.reduce_mean(tf.squared_difference(tf.multiply(wh_regression_logits, class_label), wh_regression_label))
        loss_ang_regression_logits = tf.reduce_mean(tf.squared_difference(tf.multiply(ang_regression_logits, class_label), ang_regression_label))

        # Sum up all regression losses
        regression_squared_diff = tf.add(loss_xy_regression_logits, loss_wh_regression_logits)
        regression_squared_diff = tf.add(loss_ang_regression_logits, regression_squared_diff)
        
        # Sum up all loss functions
        loss = tf.add(loss_class_logits, regression_squared_diff, name='loss')
        
        # Set up Tensorboard summary for every loss components
        class_cross_ent_summary = tf.summary.scalar("class_cross_ent_diff", loss_class_logits)
        regression_squared_diff_summary = tf.summary.scalar("regression_squared_diff", regression_squared_diff)
        xy_regression_squared_diff_summary = tf.summary.scalar("xy_regression_squared_diff",loss_xy_regression_logits)
        wh_regression_squared_diff_summary = tf.summary.scalar("wh_regression_squared_diff",loss_wh_regression_logits)
        ang_regression_squared_diff_summary = tf.summary.scalar("ang_regression_squared_diff",loss_ang_regression_logits)

        # Set up Tensorboard summary for loss using training set and validation set
        training_summary = tf.summary.scalar("training_loss", loss)
        validation_summary = tf.summary.scalar("validation_loss", loss)
        
    with tf.name_scope("train"):
        # Set up one training step using an Adam Optimizer
        train_step = tf.train.AdamOptimizer().minimize(loss)
    
    with tf.name_scope('prediction'):
        # Compute prediction from logits separately for classification outputs and regression outputs
        # For classification output, a sigmoid activation function is applied
        # For regression output, no activation function is applied
        class_prediction = tf.nn.sigmoid(tf.slice(logits, [0,0,0,0], [num_image, label_grid_size, label_grid_size, 1]))
        regression_prediction = tf.slice(logits, [0,0,0,1], [num_image, label_grid_size, label_grid_size, 5])
    
    # Set up mini batch
    start_batch = 0
    
    # Add ops to save and restore all the variables.
    saver = tf.train.Saver() 
    
    # Start a tensorflow session
    with tf.Session() as sess:
        # Initialise all variables (Eg, weights and biases in layers and etc)
        sess.run(tf.global_variables_initializer())
        
        # Restore prevously saved model if save_status is set to 1
        if save_status == 1:
            saver.restore(sess, '/tmp/yolo_vortex_model_random' + str(restore_model_num) + '.ckpt')
            print("Model restored.")
            
        # Set up Tensorboard summary writer to log data to Tensorboard
        writer = tf.summary.FileWriter(logdir= "../Tensorboard/yolo_vortex_random" + str(save_model_num),graph = sess.graph)
        
        # Run training mode to train and save model 
        if mode == 'train':
            
            # Divide training set into mini batches 
            # For loop to run training step for the number of Epoch selected
            for i in range(int(epoch*(train_size/batch_size))):
                # Randomly shuffle data in training set for new Epoch
                if start_batch + batch_size > train_size or i == 0:
                    start_batch = 0
                    perm = np.arange(train_size)
                    print ('Shuffling...')
                    np.random.shuffle(perm)
                    train_data = train_data[perm]
                    train_labels = train_labels[perm]
                
                # Define index to slice mini batch data from entire training set
                start = start_batch
                end = start_batch + batch_size
                start_batch = end
                
                # Extract mini batch data from training set
                batch = train_data[start:end], train_labels[start:end]
                
                # Run one training step with the extracted mini batch data
                sess.run(train_step ,feed_dict={x: batch[0], y: batch[1], keep_prob: drop_out_keep_prob})
               
                # Print the current number of Epoch and number of mini batch that has been ran
                print("Epoch: ", int((i*batch_size)/train_size), " Iteration:", i)
                
                # For every 100 mini batch:
                #   1. Save model
                #   2. Log summary to Tensorboard
                if i%100 == 0:
                    save_path = saver.save(sess, '/tmp/yolo_vortex_model_random' + str(save_model_num) + '.ckpt')
                    print("Model saved in file: %s" % save_path)
                    
                    class_cross_ent_summ = sess.run(class_cross_ent_summary, feed_dict={x: train_data[:eval_size], y: train_labels[:eval_size], keep_prob: 1.0})
                    regression_squared_diff_summ = sess.run(regression_squared_diff_summary, feed_dict={x: train_data[:eval_size], y: train_labels[:eval_size], keep_prob: 1.0})
                    xy_regression_squared_diff_summ = sess.run(xy_regression_squared_diff_summary, feed_dict={x: train_data[:eval_size], y: train_labels[:eval_size], keep_prob: 1.0})
                    wh_regression_squared_diff_summ = sess.run(wh_regression_squared_diff_summary, feed_dict={x: train_data[:eval_size], y: train_labels[:eval_size], keep_prob: 1.0})
                    ang_regression_squared_diff_summ = sess.run(ang_regression_squared_diff_summary, feed_dict={x: train_data[:eval_size], y: train_labels[:eval_size], keep_prob: 1.0})
                    
                    train_summ = sess.run(training_summary, feed_dict={x: train_data[:eval_size], y: train_labels[:eval_size], keep_prob: 1.0})
                    eval_summ = sess.run(validation_summary, feed_dict={x: eval_data, y: eval_labels, keep_prob: 1.0})
                    
                    writer.add_summary(class_cross_ent_summ, i)
                    writer.add_summary(regression_squared_diff_summ, i)
                    writer.add_summary(xy_regression_squared_diff_summ, i)
                    writer.add_summary(wh_regression_squared_diff_summ, i)
                    writer.add_summary(ang_regression_squared_diff_summ, i)
                    
                    writer.add_summary(train_summ, i)
                    writer.add_summary(eval_summ, i)
                    
        # Run predict mode to predict unseen data with previously saved model and parameters 
        if mode == 'predict':
            # Predict the confidence probability of presence of vortex in each grid cell
            score = sess.run(class_prediction, feed_dict={x: data[:test_size], keep_prob: 1.0})
            
            # Predict the bounding box parameters 
            box = sess.run(regression_prediction, feed_dict={x: data[:test_size], keep_prob: 1.0})
            
            # Round the confidence probability (cp = 0 if cp <0.5, 1 otherwise)
            prediction = sess.run(tf.round(class_prediction), feed_dict={x: data[:test_size], keep_prob: 1.0})

            # Process individual predicted sample
            for n,i in enumerate(prediction):
                # Print current number of sample that's being processed
                print("Sample: ", n)
                
                # Only process and plot results from the actual test set
                # This is because small test set (<50) are mixed with sample test set for batch normalization to work (ie, if test sample is too small, the mean and standard deviation calculated in batch norm might be off) 
                # Therefore, we don't want to plot test samples that are not in the actual test set
                if n >= 50:
                    break 
                
                # Set up empty arrays to filter out vortex parameters from grid cells that are without vortex
                tot_box_coor = []
                tot_score = []
                tot_size = []
                tot_angle = []
                
                # Find index of grid cells that contain vortices
                idx = np.flip(np.transpose(np.where(i == 1)), axis = 1)
                
                # Only extract vortex parameters if the index array is non zero (meaning there are vortices in the sample)               
                if idx.size > 0:
                    box_coor_score = []
                        
                    # Process every vortex found in the sample
                    for i in idx:
                        scor = score[n][i[2],i[1],i[0]]
                        
                        # Multiply normalised width and height output with grid size to find the actual width and height
                        width = abs(grid_size*box[n][i[2],i[1],2])
                        height = abs(grid_size*box[n][i[2],i[1],3])
                        angle = box[n][i[2],i[1],4]
                        
                        # As center coordinates of vortex are normalised to within each grid cell, 
                        # the number of grids beside the vortex need to be accounted for to find the actual center coordinate
                        cx = (grid_size*(i[1]+box[n][i[2],i[1],0]))
                        cy = (grid_size*(i[2]+box[n][i[2],i[1],1]))
                        
                        # Find the four corner of the unrotated Bounding Box coordinates
                        x1 = cx - width/2
                        y1 = cy - height/2
                        x2 = x1 + width
                        y2 = y1 + height
                        
                        box_coor_score.append([scor, cx, cy, x1, y1, x2, y2, width, height, angle])    
                    
                    box_coor_score = np.array(box_coor_score)
                    
                    # Sort the vortex parameters according to confidence probability
                    box_coor_score = box_coor_score[box_coor_score[:,0].argsort()]
                    
                    # Set up vortex parameters for non-max suppression operation
                    # Non-max suppression computes the Intersection area over Union Area between bounding boxes to determine if two bounding boxes are duplicate of one another
                    boxes = box_coor_score[:,3:7]
                    scores = box_coor_score[:,0]
                    
                    # Non-max suppression can also limit the total number of vortex detection. (In this case, max of up to 10 vortices can be detected in every sample)
                    max_boxes = 10
                    
                    # iou stands for Intersection over Union
                    iou_threshold = 0.5
                                      
                    # Find index of remaining vortex parameters after passing through non-max suppression
                    nms_indices = tf.image.non_max_suppression(boxes, scores, max_boxes, iou_threshold)
                    nms_indices = sess.run(nms_indices)
                    
                    # Use the non max suppression index to extract vortex parameters
                    for i in nms_indices:
                        tot_box_coor.append(box_coor_score[i, 1:3])
                        tot_score.append(box_coor_score[i, 0])
                        tot_size.append(abs(box_coor_score[i, 7:9]))
                        tot_angle.append(box_coor_score[i, 9])
            
            
                prediction = []
                # Define Bounding Box parameters
                for k in range(len(tot_score)):
                    Bbox_centre_x = tot_box_coor[k][0]
                    Bbox_centre_y = tot_box_coor[k][1]
                    Bbox_width = tot_size[k][0]
                    Bbox_height = tot_size[k][1]
                    Bbox_x = Bbox_centre_x - Bbox_width/2
                    Bbox_y = Bbox_centre_y - Bbox_height/2
                    Bbox_angle = tot_angle[k]
                    
                    prediction.append([Bbox_centre_x, Bbox_centre_y, Bbox_width ,Bbox_height, Bbox_angle])
                    
                # Convert list to numpy array
                prediction = np.array(prediction)
                
                # Sort the prediction array only if its not an empty array
                if prediction.shape[0] != 0:
                    prediction = prediction[prediction[:,0].argsort()]

                # Set up Quiver plot for the predicted result
                slice_interval = 2
            
                # Slicer index for smoother quiver plot
                # General note: Adjust the slice interval and scale accordingly to get the required arrow size.
                # Also, the units, and angles units are also responsible.
                skip = (slice(None, None, slice_interval), slice(None, None, slice_interval))
                
                plt.figure()
                i = image[n]
                
                # Get u and v velocity vector field
                U = i[0]
                V = i[1]
            
                # Get velocity vector field size
                l = img_size
                
                # Create a mesh grid for Quiver plot
                x = np.linspace(0, l-1, l)
                y = np.linspace(0, l-1, l)
                X, Y = np.meshgrid(x, y)
                
                # Compute Velocity scalar field from U and V    
                vels = np.hypot(U, V)
                Quiver = plt.quiver(X[skip], Y[skip],
                                    U[skip], V[skip],
                                    vels[skip],
                                    units='height',
                                    angles='uv',
                                    scale=10,
                                    pivot='mid',
                                    color='black',
                                    )
                plt.title("Vortex")
                plt.xlabel("x")
                plt.ylabel("y")
                plt.quiverkey(Quiver, 1.01, 1.01, 10, label='10m/s', labelcolor='blue', labelpos='N',
                                   coordinates='axes')
                plt.colorbar(Quiver)
                
                # Define the thickness of the outline of Bounding Box
                thickness = 2
                
                prediction = []
                for k in range(len(tot_score)):
                    # Define Bounding Box parameters
                    Bbox_centre_x = tot_box_coor[k][0]
                    Bbox_centre_y = tot_box_coor[k][1]
                    Bbox_width = tot_size[k][0]
                    Bbox_height = tot_size[k][1]
                    Bbox_x = Bbox_centre_x - Bbox_width/2
                    Bbox_y = Bbox_centre_y - Bbox_height/2
                    Bbox_angle = tot_angle[k]
                    prediction.append([Bbox_centre_x, Bbox_centre_y, Bbox_width ,Bbox_height, Bbox_angle])
                    
                    # Add Bounding Box to the plot
                    ax = plt.gca()
                    r2 = patches.Rectangle((Bbox_x, Bbox_y), Bbox_width, Bbox_height, fc='none', color = 'r', lw = thickness)
                    t2 = mpl.transforms.Affine2D().rotate_around(Bbox_centre_x, Bbox_centre_y, -Bbox_angle)
                    r2.set_transform(t2 + ax.transData)
                    ax.add_patch(r2)
                
                prediction = np.array(prediction)
                
                if prediction.shape[0] != 0:
                    prediction = prediction[prediction[:,0].argsort()]
                
                print(label[n])
                
                for n,i in enumerate(prediction):
                    print('predicted:   x: ', i[0], '  y: ', i[1], '  width: ', i[2], '  height: ', i[3], ' angle: ', i[4] )
                    
                    
                # Plot the quiver plot
                plt.xticks()
                plt.yticks()
                plt.axis([0, l, 0, l])
                plt.grid()
                plt.show()

                Quiver = plt.quiver(X[skip], Y[skip],
                                    U[skip], V[skip],
                                    vels[skip],
                                    units='height',
                                    angles='uv',
                                    scale=10,
                                    pivot='mid',
                                    color='black',
                                    )
                plt.title("Vortex")
                plt.xlabel("x")
                plt.ylabel("y")
                plt.quiverkey(Quiver, 1.01, 1.01, 10, label='10m/s', labelcolor='blue', labelpos='N',
                                   coordinates='axes')
                plt.colorbar(Quiver)
                
                # Plot the quiver plot
                plt.xticks()
                plt.yticks()
                plt.axis([0, l, 0, l])
                plt.grid()
                plt.show()
                
                # Plot U velocity field and Bounding Box
                for k in range(len(tot_score)):
                    # Define Bounding Box parameters
                    Bbox_centre_x = tot_box_coor[k][0]
                    Bbox_centre_y = tot_box_coor[k][1]
                    Bbox_width = tot_size[k][0]
                    Bbox_height = tot_size[k][1]
                    Bbox_x = Bbox_centre_x - Bbox_width/2
                    Bbox_y = Bbox_centre_y - Bbox_height/2
                    Bbox_angle = tot_angle[k]

                    # Add Bounding Box to the plot
                    ax = plt.gca()
                    r2 = patches.Rectangle((Bbox_x, Bbox_y), Bbox_width, Bbox_height, fc='none', color = 'r', lw = thickness)
                    t2 = mpl.transforms.Affine2D().rotate_around(Bbox_centre_x, Bbox_centre_y, -Bbox_angle)
                    r2.set_transform(t2 + ax.transData)
                    ax.add_patch(r2)
                    
                plt.imshow(U)
                plt.title("Horizontal velocity field")
                plt.xlabel("x")
                plt.ylabel("y")
                plt.colorbar()
                plt.axis([0, l, 0, l])
                plt.show()
 
                plt.imshow(U)
                plt.title("Horizontal velocity field")
                plt.xlabel("x")
                plt.ylabel("y")
                plt.colorbar()
                plt.axis([0, l, 0, l])
                plt.show()               
                
                # Plot V velocity field and Bounding Box
                for k in range(len(tot_score)):
                    Bbox_centre_x = tot_box_coor[k][0]
                    Bbox_centre_y = tot_box_coor[k][1]
                    Bbox_width = tot_size[k][0]
                    Bbox_height = tot_size[k][1]
                    Bbox_x = Bbox_centre_x - Bbox_width/2
                    Bbox_y = Bbox_centre_y - Bbox_height/2
                    Bbox_angle = tot_angle[k]

                    ax = plt.gca()
                    r2 = patches.Rectangle((Bbox_x, Bbox_y), Bbox_width, Bbox_height, fc='none', color = 'r', lw = thickness)
                    t2 = mpl.transforms.Affine2D().rotate_around(Bbox_centre_x, Bbox_centre_y, -Bbox_angle)
                    r2.set_transform(t2 + ax.transData)
                    ax.add_patch(r2)
                                    
                plt.imshow(V)
                plt.title("Vertical velocity field")
                plt.xlabel("x")
                plt.ylabel("y")
                plt.colorbar()
                plt.axis([0, l, 0, l])
                plt.show()
                    
                plt.imshow(V)
                plt.title("Vertical velocity field")
                plt.xlabel("x")
                plt.ylabel("y")
                plt.colorbar()
                plt.axis([0, l, 0, l])
                plt.show()
        
def main(mode, save_status, save_model_num, restore_model_num, save_file_num, epoch, batch_size, drop_out_keep_prob):
    """Run main function to train or run prediction on neural network.
    
    Parameters
    ----------                   
    mode : str 
        Select mode to run the neural network model at (either 'train' or 'predict)
        
    save_status : bool 
        Select whether to restore previously saved model. (1 to restore model)
              
    save_model_num, restore_model_num : int
        Select the model to be saved or restored
        
    save_file_num : int
        Numbered label for the save file name. For eg, "'yolo_random_vortex_label3.npy' is the saved label file name where 3 is the save_file_num
        
    epoch : int
        Set the number of Epoch for training (Only matters in 'train' mode)
        
    batch_size : int
        Set the size of mini batch 
        
    drop_out_keep_prob : float (between 0 and 1)
        Set the probability that a node is dropped out in a layer
    """  
    
    # Reset Tensorflow graph 
    tf.reset_default_graph()
    
    # Select image and label input file
    image_input = np.load('Artificial data/yolo_random_vortex_image_test' + str(save_file_num) + '.npy')
    label = np.load('Artificial data/yolo_random_vortex_label_test' + str(save_file_num) + '.npy')

    # For predict mode, process and input CFD data to the model
    if mode == 'predict':
        image_input = []
#        for i in [5,15,25,35,45]:
#            angle = 40
#            img = np.load('CFD data/CFD_Harrison_image_ang_' + str(angle) + '_i_' + str(i)  + '.npy')
#            img = img.reshape((2, 80, 80))
#            image_input.append(img)    
        image_input = np.array(image_input)    
    
    
    # Run the training/prediction function
    train_predict_nn(image_input, label, mode, save_status, save_model_num, restore_model_num, epoch, batch_size, drop_out_keep_prob) 
    
    
main(mode = 'predict', save_status = 1, save_model_num = 0, restore_model_num = 100, save_file_num = 0, epoch = 10, batch_size = 50, drop_out_keep_prob = 0.4)