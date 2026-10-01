# RANNS
This is the experiment result of the dissertation: "Reverse Approximate Nearest Neighbor Search: From Hardness to Density-aware Exploration"

The 'Ground_truth.py' file is to calclate the ground truth of the following datsets' k-NN where $k = 100$ here
- SIFT1M
- GIST1M
- MNIST
- DEEP1M
- SIFT10M

where the results are stored in Google Drive folder: https://drive.google.com/drive/folders/1QWjLzQDn9FjC4_0aJ9ybQcXLcu3fgot2?usp=sharing

The visualizations below are all generated from the codes above using these ground-truth results, illustrating the comparison of other SOTA RANNS algorithms:

<img width="2400" height="1000" alt="Figure_5" src="https://github.com/user-attachments/assets/9c5a6eda-a59b-41f6-aed3-9a4f7608cbf9" />

Rev-Recall@k over different parameter $k$ in different RANNS methods via different datasets.

<img width="1865" height="621" alt="Figure_11" src="https://github.com/user-attachments/assets/81d905b1-b8a2-4540-8b01-deafc2c14c2b" />

Rev-Recall over different parameter $k$ in different RANNS methods via different datasets.

<img width="1350" height="1071" alt="Figure_6" src="https://github.com/user-attachments/assets/d86852e5-5d4a-4169-b5e9-d9e8da0fd852" />

QPS vs. Rev-Recall over different RANNS methods via different datasets. ($k=50$)

Note that HNSW, HAMG, NN-Descent, NSG, RabitQ are all the ANNS algorithms, while Range exploration, Hop exploration and our method Density-Exploration are all the RANNS algorithms. 
ANNS methods will be used in RANNS, which means that ANNS algorithms have a inclusion relationship in RANNS. We here use RabitQ as a reference to imply that our DE method can be implenmented via non-graph ANNS method to show that our method is capable of both graph-based and non-graph based ANNS algorithms.

