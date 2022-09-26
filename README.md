# Five-classifications-of-intertrochanteric-fractures-master
Five classification of trochanters using fast rcnn

-------------------------------------------------------------------Required environment------------------------------------------------------------------------
tensorflow-gpu==1.13.1  
keras==2.1.5  

-----------------------------------------------------------------------Training Steps--------------------------------------------------------------------------
###Training your own dataset
1. Data set preparation
**This article uses VOC format for training. Before training, you need to make your own data set**
Before training, put the label file in the Annotation under the VOC2007 folder under the VOCdevkit folder.
Before training, put the picture file in JPEGImages under the VOC2007 folder under the VOCdevkit folder.

2.Data set processing
After placing the dataset, we need to use voc_ Annotation.py obtained 2007 for training_ Train.txt and 2007_ val.txt.
Modify voc_ annotation. The parameter in py. The first training can only modify classes_ path，classes_ The path is used to point to the txt corresponding to the detection category.
When training your own dataset, you can create a cls_ Classes.txt, where you write the categories you need to distinguish.
Modify voc_ Classes in annotation.py_ Path to make it correspond to cls_ Classs.txt, and run voc_ annotation.py.

3. Start network training
**There are many training parameters in train.py, and the most important part is still the classes in train.py_ path.**
**classes_ Path is used to point to the txt corresponding to the detection category. This txt and voc_ annotation. The txt in py is the same! The training data set must be modified**
Finished modifying classes_ After the path, you can run train.py to start training. After training multiple epochs, the weight value will be generated in the logs folder.

4. Prediction of training results
Two files are required for training result prediction, namely frcnn.py and predict. py. Modify the model in frcnn.py_ Path and classes_ path.
**model_ The path points to the trained weight file in the logs folder. classes_ The path points to the txt corresponding to the detection category**
After modification, you can run predict. py to detect. After running, enter the image path to detect.

-----------------------------------------------------------------------Forecast Steps---------------------------------------------------------------------------
1. Follow the training steps.

2. In the frcnn.py file, modify the model_ Path and classes_ Path makes it correspond to the trained file;
**model_ Path corresponds to the weight file under the logs folder, classes_ Path is the model_ The class * * corresponding to the path.

3.Run predict.py

----------------------------------------------------------------------Evaluation Steps---------------------------------------------------------------------------
1.This paper uses VOC format for evaluation.

2.If you have run voc before training_ annotation. Py file. The code will automatically divide the data set into training set, verification set and test set. If you want to modify the scale of the test set, you can modify the voc_ annotation. Trainval under py file_ percent。 trainval_ Percent is used to specify the ratio between (training set+verification set) and test set. By default, (training set+verification set): test set=9:1. train_ Percent is used to specify the proportion of training set and verification set in (training set+verification set). By default, training set: verification set=9:1.

3.Utilize voc_ After annotation.py divides the test set, go to get_ map. Py file modifying classes_ path，classes_ Path is used to point to the txt corresponding to the detection category, which is the same as the txt during training. The evaluation data set must be modified.

4.Modify the model_ Path in frcnn.py and classes_ path.** model_ The path points to the trained weight file in the logs folder. classes_ The path points to the txt corresponding to the detection category**

5.Run get_ Map.py will get the evaluation results, which will be saved in the map_ Out folder.
