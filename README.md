# Neurips 2020: Weakly-supervised-Deep-Functional-map for shape Matching
 preprint here: https://hal.archives-ouvertes.fr/hal-02872053

# Requirements:
--Tensorflow 1.x version

--Please download the tf_ops and utils folder from pointnet++ github repository and compile tf_ops according to the instructions provided there.

# Running the code
Our source code then contains 3 files:
1) train_test.py
2) model.py
3) loss.py: this contains the implementation of halimi et al. loss (pointwise_corr_layer) and also supervised loss of Donati et al.

By default, running train_test.py with suitable data runs our method and replicates all results in main paper. 

To include supervised loss of Donati et al., please replace E5 (currently set to 0) with sup_penalty_surreal. 
Similarly, to include halimi et al. unsupervised loss, please replace E5 with pointwise_corr_layer.. 
both these lines are commented out in loss.py.

# Weakly Aligned Data
Coming soon

# Partial Shape Matching Code

Coming soon

