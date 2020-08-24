# Results are in checkpoint file
# in order to view tensorboard do,
```
ssh -L 16006:127.0.0.1:6006 root@server
cd checkpoint/
tensorboard --logdir tfboard 
```

# how to train & test  
```
python main.py
```
