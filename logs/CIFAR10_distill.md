# CIFAR10 Distill

## Gzoo2

```cmd
--dataset=gzoo2
--ipc=1
--syn_steps=2
--expert_epochs=2
--max_start_epoch=19
--lr_img=1000
--lr_lr=1e-02
--lr_teacher=0.0001
--pix_init=noise
--buffer_path=/data/sbcaesar/galaxy_buffers
--data_path={path_to_dataset}
--Iteration=20000
--eval_it=200
```

## CIFAR10

```cmd
--dataset=CIFAR10
--ipc=1
--syn_steps=20
--expert_epochs=2
--max_start_epoch=19
--lr_img=1000
--lr_lr=1e-05
--lr_teacher=0.01
--pix_init=noise
--buffer_path=/data/sbcaesar/buffers
--data_path={path_to_dataset}
--Iteration=40000
--eval_it=100
--zca
```

3/17/2023

```text
--lr_teacher
0.02753339894115925
0.021951986476778984
0.019854243844747543
0.020864170044660568
0.023337967693805695
0.027003154158592224
0.030240267515182495
0.03351694718003273
0.03639672324061394
0.03785112500190735
0.038243964314460754
0.038664381951093674
0.03858708217740059
--dataset=CIFAR10
--ipc=10
--syn_steps=20
--expert_epochs=1
--max_start_epoch=29
--lr_img=500
--lr_lr=0.01
--pix_init=noise
--buffer_path=/data/sbcaesar/buffers
--data_path={path_to_dataset}
--Iteration=30001
--eval_it=200
--init_epoch=15
--prev_iter=15000
--wandb_name=hg-test-cifar10-no-zca-10ipc
--load_syn_image=hg-test-cifar10-no-zca-10ipc/images_15000.pt
--pad_interval=1
--lr_img_decay_interval=200
--sigma=3
```