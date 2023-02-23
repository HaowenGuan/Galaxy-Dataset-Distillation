
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
