# CIFAR10 Distill

## Gzoo2

```cmd
--dataset=gzoo2
--ipc=1
--syn_steps=1
--expert_epochs=2
--max_start_epoch=19
--lr_img=1000
--lr_lr=1e-05
--lr_teacher=0.001
--pix_init=noise
--buffer_path=/data/sbcaesar/buffers
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