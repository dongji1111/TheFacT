# The FacT

This is the implementation for the paper titled "The FacT: Taming Latent Factor Models for Explainability with Factorization Trees". We provide all the source codes for the algorithm.

# Reference
Yiyi Tao, Yiling Jia, Nan Wang and Hongning Wang. The FacT: Taming Latent Factor Models for Explainability with Factorization Trees, SIGIR 2019.

# Quick Start
#### Data format
In ./data/yelp_train.txt
```
user_id, item_id, rating, [list of feature opinions]
```
Example:  
1, 0, 4, 1 1 2 1  
user_id = 1, item_id = 0, rating = 4, rating for feature 1 = 1, rating for feature 2 = 1.

#### How to run the algorithm
```sh
$ cd code
$ python main.py --train_file ../data/yelp_train.txt --test_file ../data/yelp_test.txt --num_dim 20 --max_depth 6
```
The results will be stored in ./results/
