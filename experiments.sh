nohup python -u main.py --batch_size 1 --permute-interval 5000 --optimizer SGD --lr 0.001
nohup python -u main.py --batch_size 1 --permute-interval 5000 --optimizer Adam --lr 0.0001

nohup python -u main.py --batch_size 1 --permute-interval 5000 --optimizer SGDL2Init --lr 0.001 --weight-decay 0.01
nohup python -u main.py --batch_size 1 --permute-interval 5000 --optimizer AdamL2Init --lr 0.0001 --weight-decay 0.001

nohup python -u main.py --batch_size 1 --permute-interval 5000 --optimizer SGDWC --lr 0.001 --clipping 2.0
nohup python -u main.py --batch_size 1 --permute-interval 5000 --optimizer AdamWC --lr 0.0001 --clipping 1.0
