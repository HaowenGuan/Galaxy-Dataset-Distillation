
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