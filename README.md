![vgan1-light](https://github.com/user-attachments/assets/770fe2f6-8c42-4e4d-b7bd-bf015c46f993)
============================================================================


# Introduction 
Fork of Repository for the V-GAN algorithm in paper "Adversarial Subspace Generation for Knowledge Discovery in High Dimensional Data" for subspace search.

The Framework proposed in the paper has been used for my Bachelor Thesis "Subspace Clustering via Subspace Generation".

The proposed algorithm, V-GAN from the mentioned paper, is capable of identifying a collection of subspaces relevant to a studied population $\mathbf{X}$. It does so by building on a theoretical framework that explains the _Multiple Views_ phenomenom of data. 

# Installation

To install, simply use the requirements.txt file 
`pip install -r requirements.txt`
Additionally, if you plan to train VGAN, you should also install the torch-two-sample package: 

```
git clone git@github.com:josipd/torch-two-sample.git
cd torch-two-sample
pip install .
```
