
## distill 5 ipc 

```cmd
--dataset=gzoo2
--ipc=5
--syn_steps=40
--expert_epochs=2
--max_start_epoch=28
--lr_img=1000
--lr_lr=1e-02
--lr_teacher=0.002
--pix_init=noise
--buffer_path=/data/sbcaesar/galaxy_buffers
--data_path={path_to_dataset}
--Iteration=30000
--eval_it=500
--init_epoch=6
--prev_iter=12001
```
image_syn = torch.load(os.path.join(".", "logged_files", args.dataset, 'happy-glade-259', 'images_12000.pt'))

03/08/2022
```cmd
--dataset=gzoo2
--ipc=1
--syn_steps=100
--expert_epochs=30
--max_start_epoch=28
--lr_img=1000
--lr_lr=1e-01
--lr_teacher=0.0005
--pix_init=noise
--buffer_path=/data/sbcaesar/galaxy_buffers
--data_path={path_to_dataset}
--Iteration=20000
--eval_it=100
--init_epoch=1
--prev_iter=1
--wandb_name=1ipc-head-to-tail
--load_syn_image=1ipc-300-fine-tuning/images_8300.pt
```


03/15/2022
```cmd
--lr_teacher
0.0020587416365742683
0.0012722620740532875
0.00109484710264951
0.001034212065860629
0.0010052368743345141
0.0009980791946873069
0.0009886125335469842
0.0009877606062218547
0.0009707224671728909
0.0009692400926724076
0.0009626870742067695
0.0009545870707370341
0.0009482063469476998
0.0009303857223130763
0.0009366782614961267
0.0009150573168881238
0.0008964111912064254
0.0008949931943789124
0.000874847115483135
0.0008672471158206463
0.0008558279369026423
0.0008490750915370882
0.0008370944415219128
0.0008359130006283522
0.0008243904449045658
0.0008228619117289782
0.0008177702547982335
0.0007997527718544006
0.0007968715508468449
--dataset=gzoo2
--ipc=5
--syn_steps=30
--expert_epochs=1
--max_start_epoch=29
--lr_img=10
--lr_lr=0.01
--pix_init=noise
--buffer_path=/data/sbcaesar/galaxy_buffers
--data_path={path_to_dataset}
--Iteration=30000
--eval_it=500
--init_epoch=29
--prev_iter=21001
--wandb_name=continue-5ipc-epoch-all-fine-tune
--load_syn_image=continue-5ipc-epoch-all/images_21000.pt
```