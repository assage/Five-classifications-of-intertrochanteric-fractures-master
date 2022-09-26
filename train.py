import re
import pickle
import numpy as np
import os
import keras
import keras.backend as K
# import tf
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from setuptools.sandbox import save_path
from tensorflow.python.training import saver

from nets.frcnn import get_model
from nets.frcnn_training import (ProposalTargetCreator, classifier_cls_loss,
                                 classifier_smooth_l1, rpn_cls_loss,
                                 rpn_smooth_l1)
from utils.anchors import get_anchors
from utils.callbacks import LossHistory
from utils.dataloader import FRCNNDatasets
from utils.utils import get_classes
from utils.utils_bbox import BBoxUtility
from utils.utils_fit import fit_one_epoch


if __name__ == "__main__":
   
    classes_path    = 'model_data/voc_classes.txt'
  
    model_path      = 'logs/NASNet-mobile-no-top 100epoch.h5'
   
    input_shape     = [600, 600]
  
    backbone        = "vgg"
  
    anchors_size    = [64, 256, 512]

  
    Init_Epoch          = 0
    Freeze_Epoch        = 50
    Freeze_batch_size   = 8
    Freeze_lr           = 1e-4

    UnFreeze_Epoch      = 100
    Unfreeze_batch_size = 8
    Unfreeze_lr         = 1e-5
 
    Freeze_Train        = True
   
    train_annotation_path   = '2007_train.txt'
    val_annotation_path     = '2007_val.txt'

    class_names, num_classes = get_classes(classes_path)
    num_classes += 1
    anchors = get_anchors(input_shape, backbone, anchors_size)

    K.clear_session()
    model_rpn, model_all = get_model(num_classes, backbone = backbone)
    if model_path != '':
      
        print('Load weights {}.'.format(model_path))
        model_rpn.load_weights(model_path, by_name=True)
        model_all.load_weights(model_path, by_name=True)

    callback        = TensorBoard(log_dir="logs")
    callback.set_model(model_all)
    loss_history    = LossHistory("logs/")

    bbox_util       = BBoxUtility(num_classes)
    roi_helper      = ProposalTargetCreator(num_classes)
   
    with open(train_annotation_path) as f:
        train_lines = f.readlines()
    with open(val_annotation_path) as f:
        val_lines   = f.readlines()
    num_train   = len(train_lines)
    num_val     = len(val_lines)

    freeze_layers = {'vgg': 17, 'resnet50': 141}[backbone]
    if Freeze_Train:
        for i in range(freeze_layers): 
            if type(model_all.layers[i]) != keras.layers.BatchNormalization:
                model_all.layers[i].trainable = False
        print('Freeze the first {} layers of total {} layers.'.format(freeze_layers, len(model_all.layers)))

    if True:
        batch_size  = Freeze_batch_size
        lr          = Freeze_lr
        start_epoch = Init_Epoch
        end_epoch   = Freeze_Epoch

        model_rpn.compile(
            loss = {
                'classification': rpn_cls_loss(),
                'regression'    : rpn_smooth_l1()
            }, optimizer = Adam(lr=lr)
        )
        model_all.compile(
            loss = {
                'classification'                        : rpn_cls_loss(),
                'regression'                            : rpn_smooth_l1(),
                'dense_class_{}'.format(num_classes)    : classifier_cls_loss(),
                'dense_regress_{}'.format(num_classes)  : classifier_smooth_l1(num_classes - 1)
            }, optimizer = Adam(lr=lr)
        )

        gen     = FRCNNDatasets(train_lines, input_shape, anchors, batch_size, num_classes, train = True).generate()
        gen_val = FRCNNDatasets(val_lines, input_shape, anchors, batch_size, num_classes, train = False).generate()

        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError('数据集过小，无法进行训练，请扩充数据集。')

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        for epoch in range(start_epoch, end_epoch):
            fit_one_epoch(model_rpn, model_all, loss_history, callback, epoch, epoch_step, epoch_step_val, gen, gen_val, end_epoch,
                    anchors, bbox_util, roi_helper)
            lr = lr*0.96
            K.set_value(model_rpn.optimizer.lr, lr)
            K.set_value(model_all.optimizer.lr, lr)

    if Freeze_Train:
        for i in range(freeze_layers): 
            if type(model_all.layers[i]) != keras.layers.BatchNormalization:
                model_all.layers[i].trainable = True

    if True:
        batch_size  = Unfreeze_batch_size
        lr          = Unfreeze_lr
        start_epoch = Freeze_Epoch
        end_epoch   = UnFreeze_Epoch

        model_rpn.compile(
            loss = {
                'classification': rpn_cls_loss(),
                'regression'    : rpn_smooth_l1()
            }, optimizer = Adam(lr=lr)
        )
        model_all.compile(
            loss = {
                'classification'                        : rpn_cls_loss(),
                'regression'                            : rpn_smooth_l1(),
                'dense_class_{}'.format(num_classes)    : classifier_cls_loss(),
                'dense_regress_{}'.format(num_classes)  : classifier_smooth_l1(num_classes - 1)
            }, optimizer = Adam(lr=lr)
        )

        gen = FRCNNDatasets(train_lines, input_shape, anchors, batch_size, num_classes, train=True).generate()
        gen_val = FRCNNDatasets(val_lines, input_shape, anchors, batch_size, num_classes, train=False).generate()

        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError('The dataset is too small for training. Please expand the dataset.')

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        for epoch in range(start_epoch, end_epoch):
            fit_one_epoch(model_rpn, model_all, loss_history, callback, epoch, epoch_step, epoch_step_val, gen, gen_val, end_epoch,
                    anchors, bbox_util, roi_helper)
            lr = lr*0.96
            K.set_value(model_rpn.optimizer.lr, lr)
            K.set_value(model_all.optimizer.lr, lr)