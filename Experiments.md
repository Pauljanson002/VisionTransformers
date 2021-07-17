Experiment 1 -
vit_lite , num_layers = 14
state_dict = vit_lite_v2.pt

Experiment 2 - 
vit_lite_h, num_layers = 32
state_dict = vit_lite_h.pt

Experiment 3 -
vit_lite_h further 100 epoch training 
resume checkpoint = vit_lite_h_100.pt
save checkpoint = vit_lite_h_200.pt

Experiment 4- 
Apples and apples comparision
vit_lite_seq 200 epochs training
save checkpoint = vit_lite_seq.pt

Aim - Finding the difference between the pooling and classtoken in vision transformer

Experimnet 5- 
Layer wise classtoken check for vit_lite_h
saveimage - vit_lite_h.png

Experiment 6-
Layer wise pooling check for vit_lite_seq
logged on log_vit_lite_seq