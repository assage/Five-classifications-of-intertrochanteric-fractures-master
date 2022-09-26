import colorsys
import os
import time

import numpy as np
from keras.applications.imagenet_utils import preprocess_input
from PIL import ImageDraw, ImageFont

import nets.frcnn as frcnn
from utils.anchors import get_anchors
from utils.utils import cvtColor, get_classes, get_new_img_size, resize_image
from utils.utils_bbox import BBoxUtility

class FRCNN(object):
    _defaults = {
       
        "model_path"    : 'logs/ep038-loss0.330-val_loss1.532.h5',
        "classes_path"  : 'model_data/voc_classes.txt',
       
        "backbone"      : "vgg",
       
        "confidence"    : 0.5,
       
        "nms_iou"       : 0.3,
       
        'anchors_size'  : [64, 256, 512],
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

   
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
    
        self.class_names, self.num_classes  = get_classes(self.classes_path)
        self.num_classes                    = self.num_classes + 1
     
        self.bbox_util = BBoxUtility(self.num_classes, nms_iou = self.nms_iou, min_k = 150)

      
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        self.generate()

  
    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'
      
        self.model_rpn, self.model_classifier = frcnn.get_predict_model(self.num_classes, self.backbone)
        self.model_rpn.load_weights(self.model_path, by_name=True)
        self.model_classifier.load_weights(self.model_path, by_name=True)
        print('{} model, anchors, and classes loaded.'.format(model_path))
    
  
    def detect_image(self, image):
      
        image_shape = np.array(np.shape(image)[0:2])
      
        input_shape = get_new_img_size(image_shape[0], image_shape[1])
      
        image       = cvtColor(image)
       
        image_data  = resize_image(image, [input_shape[1], input_shape[0]])
       
        image_data  = np.expand_dims(preprocess_input(np.array(image_data, dtype='float32')), 0)

        rpn_pred        = self.model_rpn.predict(image_data)
       
        anchors         = get_anchors(input_shape, self.backbone, self.anchors_size)
        rpn_results     = self.bbox_util.detection_out_rpn(rpn_pred, anchors)

        results         = self.bbox_util.detection_out_classifier(classifier_pred, rpn_results, image_shape, input_shape, self.confidence)

        if len(results[0]) == 0:
            return image
            
        top_label   = np.array(results[0][:, 5], dtype = 'int32')
        top_conf    = results[0][:, 4]
        top_boxes   = results[0][:, :4]
        
        font = ImageFont.truetype(font='model_data/simhei.ttf',size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))
        thickness = max((np.shape(image)[0] + np.shape(image)[1]) // input_shape[0], 1)

        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = top_conf[i]

            top, left, bottom, right = box

            top     = max(0, np.floor(top).astype('int32'))
            left    = max(0, np.floor(left).astype('int32'))
            bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
            right   = min(image.size[0], np.floor(right).astype('int32'))

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)
            
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        return image

    def get_FPS(self, image, test_interval):
       
        image_shape = np.array(np.shape(image)[0:2])
      
        input_shape = get_new_img_size(image_shape[0], image_shape[1])
       
        image       = cvtColor(image)
       
        image_data  = resize_image(image, [input_shape[1], input_shape[0]])
       
        image_data  = np.expand_dims(preprocess_input(np.array(image_data, dtype='float32')), 0)

        anchors         = get_anchors(input_shape, self.backbone, self.anchors_size)
        rpn_results     = self.bbox_util.detection_out_rpn(rpn_pred, anchors)
      
        classifier_pred = self.model_classifier.predict([rpn_pred[2], rpn_results[:, :, [1, 0, 3, 2]]])
       
        results         = self.bbox_util.detection_out_classifier(classifier_pred, rpn_results, image_shape, input_shape, self.confidence)

        t1 = time.time()
        for _ in range(test_interval):
           
            anchors         = get_anchors(input_shape, self.backbone, self.anchors_size)
            rpn_results     = self.bbox_util.detection_out_rpn(rpn_pred, anchors)
            temp_ROIs       = rpn_results[:, :, [1, 0, 3, 2]]
           
            results         = self.bbox_util.detection_out_classifier(classifier_pred, rpn_results, image_shape, input_shape, self.confidence)

        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

    def get_map_txt(self, image_id, image, class_names, map_out_path):
        f = open(os.path.join(map_out_path, "detection-results/"+image_id+".txt"),"w") 
       
        image_shape = np.array(np.shape(image)[0:2])
       
        input_shape = get_new_img_size(image_shape[0], image_shape[1])
      
        image       = cvtColor(image)
       
        image_data  = resize_image(image, [input_shape[1], input_shape[0]])
       
        image_data  = np.expand_dims(preprocess_input(np.array(image_data, dtype='float32')), 0)
        rpn_pred        = self.model_rpn.predict(image_data)
       
        anchors         = get_anchors(input_shape, self.backbone, self.anchors_size)
        rpn_results     = self.bbox_util.detection_out_rpn(rpn_pred, anchors)
        
        classifier_pred = self.model_classifier.predict([rpn_pred[2], rpn_results[:, :, [1, 0, 3, 2]]])
       
        results         = self.bbox_util.detection_out_classifier(classifier_pred, rpn_results, image_shape, input_shape, self.confidence)

        if len(results[0])<=0:
            return 

        top_label   = np.array(results[0][:, 5], dtype = 'int32')
        top_conf    = results[0][:, 4]
        top_boxes   = results[0][:, :4]

        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = str(top_conf[i])
            
            top, left, bottom, right = box

            if predicted_class not in class_names:
                continue

            f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))

        f.close()
        return

