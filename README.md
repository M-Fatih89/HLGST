# HLGST: Hybrid Local-Global Spatio-Temporal Model for Travel Time Estimation Using Siamese Graph Convolutional with Triplet Networks:
A novel deep-learning framework named "Hybrid Local-Global Spatio-Temporal (HLGST) Model for Travel Time Estimation Using Siamese Graph Convolutional with Triplet Networks". It can consider both local and global spatial-temporal correlations between traffic nodes in order to improve the accuracy of TTE results.
![Fig1](https://user-images.githubusercontent.com/66161950/234546722-138f26bb-fa72-472c-b70f-20fd2699c8d7.jpg)

# Dynamic Composition Unit:
![Graphs](https://user-images.githubusercontent.com/66161950/234547534-db726e35-41b8-4632-a53f-86cdcdc7bdac.jpg)
![Fig2](https://user-images.githubusercontent.com/66161950/234547586-de211365-dad2-4260-9be3-23522894fbe3.jpg)
A dynamic composition unit that extracts local space traffic information and constructs multi-dynamic semantic graphs using Geo-hashing and Sparse-DTW methods during the data preprocessing phase. To consider large-scale domains of geo-space patterns and temporal similarity, we transfer the multiple basic graphs to dynamic semantic graphs' representations.

# Local & Global Correlation Module:
![Fig3](https://user-images.githubusercontent.com/66161950/234547920-3d316de8-4608-4fdb-8482-a1891074f75b.jpg)
A dynamic spatial-temporal correlation block, consisting of two modules; (a) the local correlation module, which combines casual TCN layers with a self-attention mechanism to capture local space correlations and extract spatial-temporal feature relations; and (b) the global correlation module, which applies parallel multi-GCN blocks integrated with Siamese nets to model global space correlations and learn complex temporal patterns from the similarities of multi-graphs.

# Datasets:
* We used two real-world large-scale traffic datasets publicly available in (https://outreach.didichuxing.com/app-vue/dataList) to demonstrate the effectiveness of our proposed method, which are the Xi'an and Chengdu trajectory datasets from August 1 to September 1, 2018.
* Outliers & Preprocessing: The original dataset contains sequences of consecutive GPS-points, which comprise longitude, latitude, and a time-stamp every three seconds. The process of filtering and cleaning outliers begins by discarding trip data that lacks pick-up or drop-off location coordinates or has erroneous GPS-locations. After that, all trips with a duration exceeding 24 hours or less than 3 minutes are excluded, along with those covering distances of less than 500 m or more than 100 Km.
* The datasets were split into train and test sets with approximately 70% to 30%, where data from days 1 to 20 were used for model training and days 21 to 31 for testing.
* For each of the Xi'an and Chengdu Taxi datasets, a sample of 100K records is provided in folders (Xian Data, Chengdu Data). Furthermore, multiple dynamic adjacency matrices have been uploaded in the datasets' folders as (Adj_distance.npy, Adj_speed.npy, Flow_speed.npy).

# Code files:
The experiment of the prediction model for each dataset is given in a separate Jupyter notebook (Xian_model.ipynb and Chengdu_model.ipynb). Moreover, a DAL_Algorithm.py file is provided.

# Hyperparameters:
Many different settings for hyperparameters were studied to obtain the most accurate results for our proposed model. In this context, the Adam optimizer and Relu activation were used to train our model with a learning rate = 0.001, batch-size = 32. For each individual model, training epochs = 30.

# Environment:
Python 3.6.13 and the following packages are required:
Pandas 1.1.5; Numpy 1.19.2; Tensorflow 2.6.2; Keras 2.6.0; Pickle5 4.0; Math, scikit-learn .0.24.2.
