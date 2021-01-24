# Rainforest-Connection-Species-Audio-Detection

# Business Problem:
In this challenge, we have to predict based on the sounds of various species of birds and frogs which species the sound will belong to. Traditional methods of assessing the diversity and abundance of species are costly and limited in space and time. So a deep learning-based approach will be very helpful to accurately detect the species in noisy landscapes. Rainforest Connection(RFCx) created the world's first real-time monitoring system for protecting and supporting remote systems and unlike visual-based tracking systems like drones or satellites, RFCxrelies on acoustic sensors to monitor the ecosystem soundscapes at different locations all year round. The system built by RFCx also has the capacity to create convolutional neural network (CNN) models for analysis. In this problem, we have to automate the detection of birds and frog species based on sound recordings.
https://www.kaggle.com/c/rfcx-species-audio-detection/overview

# Problem Statement:
In this problem, we have to predict the species id of the birds or frog species based on the soundscape recordings of the species. This resulting real-time information could enable earlier detection of human environmental impacts, making environmental conservation more swift and effective.

# About the dataset:
This dataset contains train_tp CSV files which contain 1216 rows which contain species_ids, songtype_ids, f_min, f_max, t_min and t_max values. The CSV files contain the file ids and the corresponding .flac audio files in the train folder and the train tfrec records also. There is another CSV file train_fp which contains incorrect species ids and the corresponding flac audio files in the train folder. There are also 1992 test files in the test folder that should be used to predict the values using the model for testing.

# Performance Metric:
The metric used for evaluation is label-ranking-average precision(LRAP), this metric is linked with average precision score but based on the notion of label ranking instead of precision and recall.
