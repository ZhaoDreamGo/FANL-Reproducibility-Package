Indices Documentation
=====================

This directory contains the exact sample indices used in all experiments
reported in the paper. These indices reproduce the 70/15/15 train, validation,
and test splits using the same random seed (42) as in the manuscript.

--------------------------------------------
1. PathMNIST 30,000-sample balanced subset
--------------------------------------------

Files:
- indices_30k_train.txt
- indices_30k_val.txt
- indices_30k_test.txt

Constructed as:
- balanced subset across 9 classes
- total = 30,000 samples
- random seed = 42
- split ratio = 70/15/15

--------------------------------------------
2. PathMNIST Full Dataset (107,180 samples)
--------------------------------------------

Files:
- indices_full_pathmnist_train.txt
- indices_full_pathmnist_val.txt
- indices_full_pathmnist_test.txt

--------------------------------------------
3. BloodMNIST Full Dataset (17,092 samples)
--------------------------------------------

Files:
- indices_full_bloodmnist_train.txt
- indices_full_bloodmnist_val.txt
- indices_full_bloodmnist_test.txt

--------------------------------------------
Format:
Each file contains one integer index per line.
These indices correspond to MedMNIST dataset sample ordering.
