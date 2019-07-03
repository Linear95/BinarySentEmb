# BinarySentEmb
Codes for the ACL 2019 paper: Learning Compressed Sentence Representations for On-Device Text Processing.


This repository contains source code necessary to reproduce the results presented in the following paper:
* [*Learning Compressed Sentence Representations for On-Device Text Processing*](http://people.ee.duke.edu/~lcarin/Compressed_ACL2019.pdf) (ACL 2019)

This project is maintained by [Pengyu Cheng](https://linear95.github.io/). Feel free to contact pengyu.cheng@duke.edu for any relevant issues.

## Dependencies: 
This code is written in python. The dependencies are:
* Python 3.6
* Pytorch>=0.4 (0.4.1 is recommended)
* NLTK >= 3


## Download pretrained models:

First, download [GloVe](https://nlp.stanford.edu/projects/glove/) pretrained word embeddings:

```bash
mkdir dataset/GloVe
curl -Lo dataset/GloVe/glove.840B.300d.zip http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip dataset/GloVe/glove.840B.300d.zip -d dataset/GloVe/
```
Then, follow the instruction of [InferSent](https://github.com/facebookresearch/InferSent) to download pretrain universal sentence encoder:

```bash
mkdir encoder
curl -Lo encoder/infersent1.pkl https://dl.fbaipublicfiles.com/infersent/infersent1.pkl
```


## Evaluate the binary encoder on transfer tasks


-----------------------------------------------------------------------------------------------
## Citation 
Please cite our ACL paper if you found the code useful.

```latex
@article{shen2019learning,
  title={Learning Compressed Sentence Representations for On-Device Text Processing},
    author={Shen, Dinghan and Cheng, Pengyu and Sundararaman, Dhanasekar and Zhang, Xinyuan and Yang, Qian and Tang, Meng and Celikyilmaz, Asli and Carin, Lawrence},
      journal={arXiv preprint arXiv:1906.08340},
        year={2019}
}
```

