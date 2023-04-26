# HLGST-Hybrid-Local-Global-Spatio-Temporal-Model-for-Travel-Time-Estimation-Using-Siamese-Graph-Conv
A novel deep-learning framework named "Hybrid Local-Global Spatio-Temporal (HLGST) Model for Travel Time Estimation Using Siamese Graph Convolutional with Triplet Networks". It can consider both local and global spatial-temporal correlations between traffic nodes in order to improve the accuracy of TTE results.
![Fig1](https://user-images.githubusercontent.com/66161950/234546722-138f26bb-fa72-472c-b70f-20fd2699c8d7.jpg)

# Dynamic Composition Unit
![Graphs](https://user-images.githubusercontent.com/66161950/234547534-db726e35-41b8-4632-a53f-86cdcdc7bdac.jpg)
![Fig2](https://user-images.githubusercontent.com/66161950/234547586-de211365-dad2-4260-9be3-23522894fbe3.jpg)
A dynamic composition unit that extracts local space traffic information and constructs multi-dynamic semantic graphs using Geo-hashing and Sparse-DTW methods during the data preprocessing phase. To consider large-scale domains of geo-space patterns and temporal similarity, we transfer the multiple basic graphs to dynamic semantic graphs' representations.

# Local & Global Correlation Module
![Fig3](https://user-images.githubusercontent.com/66161950/234547920-3d316de8-4608-4fdb-8482-a1891074f75b.jpg)
A dynamic spatial-temporal correlation block, consisting of two modules; (a) the local correlation module, which combines casual TCN layers with a self-attention mechanism to capture local space correlations and extract spatial-temporal feature relations; and (b) the global correlation module, which applies parallel multi-GCN blocks integrated with Siamese nets to model global space correlations and learn complex temporal patterns from the similarities of multi-graphs.
