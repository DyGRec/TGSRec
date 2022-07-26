# Introduction 
This is the repository of our accepted CIKM 2021 paper "Continuous-Time Sequential Recommendation with Temporal Graph Collaborative Transformer" and the proposed model is TGSRec. Paper is available on [arxiv](https://arxiv.org/abs/2108.06625). This work focuses on multi-steps continuous-time recommendation, where user and item embeddings are generated in any unseen future timestamps. Different from existing sequential recommendation methods, which are optimized for next-item prediction, this work is learned for recommendation in any timestamps.

# Update
As we just observed some bugs in existing code, we are rerunning the experiments and will update them to the paper as soon as possible.

# Citation
Please cite our paper if using this code. 
```
@inproceedings{fan2021continuous,
  title={Continuous-Time Sequential Recommendation with Temporal Graph Collaborative Transformer},
  author={Fan, Ziwei and Liu, Zhiwei and Zhang, Jiawei and Xiong, Yun and Zheng, Lei and Yu, Philip S.},
  booktitle={Proceedings of the 30th ACM International Conference on Information and Knowledge Management},
  year={2021},
  organization={ACM}
}
```

# Implementation

The code is implemented based on [TGAT](https://github.com/StatsDLMathsRecomSys/Inductive-representation-learning-on-temporal-graphs).

## Environment Setup
The code is tested under a Linux desktop (w/ GTX 1080 Ti GPU) with Pytorch and Python 3.6.
Create the requirement with the requirements.txt

## ML-100K Dataset Execution
### Sample code to run
```
python run_TGREC.py -d ml-100k --uniform --bs 600 --lr 0.001 --n_degree 30 --agg_method attn --attn_mode prod --gpu 0 --n_head 2 --n_layer 2 --prefix Video_Games_bce --node_dim 32 --time_dim 32 --drop_out 0.3 --reg 0.3 --negsampleeval 1000
```
