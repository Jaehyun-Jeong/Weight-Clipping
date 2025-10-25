python main.py --batch_size 1 --optimizer SGD --lr 0.001
python main.py --batch_size 1 --optimizer Adam --lr 0.0001

python main.py --batch_size 1 --optimizer SGDL2Init --lr 0.001 --weight-decay 0.01
python main.py --batch_size 1 --optimizer AdamL2Init --lr 0.0001 --weight-decay 0.001

python main.py --batch_size 1 --optimizer SGDWC --lr 0.001 --clipping 2.0
python main.py --batch_size 1 --optimizer AdamWC --lr 0.0001 --clipping 1.0
