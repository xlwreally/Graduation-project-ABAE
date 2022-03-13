# About Restaurant

[Download](https://drive.google.com/open?id=1L4LRi3BWoCqJt5h45J2GIAW9eP_zjiNc)
dataset and put it in `pretrain/datasets/*`.
Then Execute following command to train and evaluate ABAE.

###Linux:
```shell script
python abae.py --neg_count 20 \
    --aspect_size 14 \
    --data_dir datasets/restaurant \
    --save_path ./model/ABAE.pt
```
###Windows:
```shell script
python abae.py --neg_count 20 --aspect_size 14  --data_dir datasets\restaurant --save_path .\model\ABAE.pt
```
The author's original code:[ruidan/Unsupervised-Aspect-Extraction
](https://github.com/ruidan/Unsupervised-Aspect-Extraction).
