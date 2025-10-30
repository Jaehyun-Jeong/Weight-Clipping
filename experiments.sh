for ((i=1; i<=3; i++))
do
    nohup python -u main.py --batch_size 1 --dataset LP-MINI-IMAGENET --permute-interval 2500 --optimizer SGD --lr 0.01 --seed "$i"
    nohup python -u main.py --batch_size 1 --dataset LP-MINI-IMAGENET --permute-interval 2500 --optimizer Adam --lr 0.0001 --seed "$i"

    nohup python -u main.py --batch_size 1 --dataset LP-MINI-IMAGENET --permute-interval 2500 --optimizer SGDL2Init --lr 0.01 --weight-decay 0.01 --seed "$i"
    nohup python -u main.py --batch_size 1 --dataset LP-MINI-IMAGENET --permute-interval 2500 --optimizer AdamL2Init --lr 0.001 --weight-decay 0.01 --seed "$i"

    nohup python -u main.py --batch_size 1 --dataset LP-MINI-IMAGENET --permute-interval 2500 --optimizer SGDWC --lr 0.01 --clipping 1.0 --seed "$i"
    nohup python -u main.py --batch_size 1 --dataset LP-MINI-IMAGENET --permute-interval 2500 --optimizer AdamWC --lr 0.0001 --clipping 3.0 --seed "$i"
done
