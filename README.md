# UR-HAT
Source code for ECAI 2023 paper: [Uncertain Relational Hypergraph Attention Networks for Document-level Event Factuality Identification](https://ebooks.iospress.nl/pdf/doi/10.3233/FAIA230508). 

Document-level event factuality identification (DocEFI) is an important task in event knowledge acquisition, which aims to detect whether an event actually occurs or not from the perspective of the document. Unlike the sentence-level task, a document can have multiple sentences with different event factualities, leading to event factuality conflicts in DocEFI. Existing studies attempt to aggregate local event factuality by exploiting document structures, but they mostly consider textual components in the document separately, degrading complicated correlations therein. 

To address the above issues, this paper proposes a novel approach, namely UR-HAT, to improve DocEFI with uncertain relational hypergraph attention networks. Particularly, we reframe a document graph as a hypergraph, and establish beneficial n-ary correlations among textual nodes with relational hyperedges, which helps to globally consider local factuality features to resolve event factuality conflicts. To better discern the importance of event factuality features, we further represent textual nodes with uncertain Gaussian distributions, and propose novel uncertain relational hypergraph attention networks to refine textual nodes with the document hypergraph. In addition, we select factuality-related keywords as nodes to enrich event factuality features. Experimental results demonstrate the effectiveness of our proposed method, and outperforms previous methods on two widely used benchmark datasets.


# Requirements

We conduct our experiments on the following environments:

```
python 3.6
CUDA: 9.0
GPU: Tesla T4
pytorch == 1.1.0
transformers == 4.9.1
```

# How to run

Put ```bert-base-uncased``` and ```bert-base-chinese``` into /plm.

For English data:
```
CUDA_VISIBLE_DEVICES=0 python main.py --data_path ./data/english.xml --saved_path ./data/english_version  --gcn_layers 2 --model_name_or_path plm/bert-base-uncased --batch_size 1 --gpu  --exp_name default --aux_loss true --kl_loss true --force_refresh true
```

For Chinese data:

```
CUDA_VISIBLE_DEVICES=0 python main.py --data_path ./data/chinese.xml  --saved_path ./data/chinese_version  --gcn_layers 2 --model_name_or_path plm/bert-base-chinese --batch_size 1 --gpu --max_sen_len 150 --max_sen_num 35 --max_tri_num 45 --exp_name zh_default --aux_loss true --kl_loss true --language zh --force_refresh true
```



# Citation

If you find this code useful, please cite our work:
```
@inproceedings{DBLP:conf/ecai/ShengCCG0WLX23,
  author       = {Jiawei Sheng and
                  Xin Cong and
                  Jiangxia Cao and
                  Shu Guo and
                  Chen Li and
                  Lihong Wang and
                  Tingwen Liu and
                  Hongbo Xu},
  title        = {Uncertain Relational Hypergraph Attention Networks for Document-Level Event Factuality Identification},
  booktitle    = {Proceedings of ECAI},
  volume       = {372},
  pages        = {2129--2137},
  year         = {2023},
  url          = {https://doi.org/10.3233/FAIA230508},
  doi          = {10.3233/FAIA230508}
}
```
