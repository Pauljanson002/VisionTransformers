#Vit lite with 10 heads 7 layers 512 embeding dimension
#python -u train_v2.py --savename vit_lite_100_7L.pt --epoch 100 --batch_size 128 --learning_rate 0.0005 --weight_decay 3e-2 --model vit_lite_100 --dataset cifar100 --online --run_name vit_on_cifar100_14layer8head

#CCT with 4 heads and 6 layers 256 embeding dimension
python -u train_v2.py --savename vit_lite_100_cct.pt --epoch 10 --batch_size 64 --learning_rate 0.0005 --weight_decay 3e-2 --model cct7 --dataset cifar100  --run_name cct_on_cifar100
