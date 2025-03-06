1.Firstly,obtained a labeled skull dataset with 51 classifications.
2.Use concat.py to crop the labeled skulls to isolate the teeth, resulting in a dataset where teeth correspond one-to-one with skulls.
3.Use split51-18.py to divide the skull and tooth datasets into 18 genus classifications.
4.Use split18-2level.py to divide the skull and tooth datasets into a hierarchical network data format.
5.Use argument18.py and argument2level.py to perform data augmentation on the 18 genus classification datasets and hierarchical network datasets respectively (for both skulls and teeth).
6.Use EB7_Train.py to train the 18 genus classification datasets for skulls and teeth separately.
7.Use HIS-EB7_Train.py to train the hierarchical network datasets for skulls and teeth separately, which focuses on polymorphism classification.
8.Use Extract_Feature.py to extract features from the previously trained models for skulls and teeth respectively.
9.Use Feature_Fusion.py to fuse the features of skulls and teeth.
10.Use HISNIFF-Test.py for the final species identification. By inputting the image addresses of skulls and teeth, the identification results can be output.