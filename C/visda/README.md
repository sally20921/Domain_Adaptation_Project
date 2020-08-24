# Domain Adaptation Datasets

1. Office31 https://drive.google.com/file/d/0B4IapRTv9pJ1WGZVd1VDMmhwdlE/view
2. OfficeHome http://hemanthdv.org/OfficeHome-Dataset/
3. VisDA-C http://ai.bu.edu/visda-2017/


## How to use this code

```
# you are in folder A
mkdir datasets
conda activate A
pip3 install -r  requirements.txt
python3 download_dataset.py Office 
# python3 download_dataset.py Office-Home
# python3 download_dataset.py VisDA
python3 main.py
```

```
ssh -L 16006:127.0.0.1:6006 root@server
cd checkpoint/
tensorboard --logdir tfboard
```

# how to train & test
```
python main.py
```
