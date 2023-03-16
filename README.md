# Galaxy Dataset Distillation

Galaxy dataset distillation is a project on creating synthesized image representations of galaxy properties. It is an extension of the general dataset distillation method, which aims to distill a large dataset into a smaller one that can be used to train a model and possibly approximates the accuracy of a model trained on the full dataset. We apply the state-of-the-art [trajectory matching algorithm](https://georgecazenavette.github.io/mtt-distillation/) to Galaxy Zoo 2 dataset. Below are some awesome examples of our distillation result. We are currently drafting the paper for a conference workshop.

## About Galaxy Zoo 2 Dataset

[Galaxy Zoo 2](https://academic.oup.com/mnras/article/435/4/2835/1022913) is a survey based dataset. Based on original classification tree, we build a simplified version for this project.

![Classification Tree](docs/gz2_tree.png)

### Sub Dataset 100 Per Class

We sort the confidence of galaxies in descending order and pick the top 100 confident images for each class to form a sub dataset (a dataset of 1000 images).

#### Baseline Image and ACC

To study this sub dataset, we averaged the 100 images of each class and form the picture below. Training a ConvNetD3 using the **one per class AVG images**, the ACC is $19.39$%. This serves as the **baseline** of our approach.

![100 avg](docs/gzoo2-1-per-class-AVG-of-100-dataset-0-9.png)

#### Distillation and ACC

Below is our current best distilled **one per class synthetic images**. The ACC is as high as $46$%. (Full Dataset is $65$%).

![100 distill](docs/distill_100_per_class_0.46_ACC.png)


### Distillation Hyperparameter

```cmd
# Stage Distillation starting range: [0, init_epoch)
--init_epoch=1

# syntheic lr, its size should match init_epoch: list[float]
--lr_teacher
0.001
--init_epoch=1
# or
--lr_teacher
0.001
0.0005
0.0001
--init_epoch=3

# Dataset Name: str
--dataset=gzoo2

# Image Per Class: int
--ipc=1

# Synthetic Step, usually, the larger the better, uses more GPU memory and slower
--syn_steps=15

# For Stage distillation, fix this to 1
--expert_epochs=1

# Use the number of epoch in buffer minus 1
--max_start_epoch=29

# Learning rate for updating synthetic image
--lr_img=1000

# Learning rate for updating lr_teacher
--lr_lr=0.01

# Choose from {noise|real}
--pix_init=noise

# Buffer path
--buffer_path=/data/sbcaesar/galaxy_buffers

# Dataset Path
--data_path={path_to_dataset}

# Maximum number of iteration
--Iteration=10000

# Evaluation interval size
--eval_it=200

# (Optional: default 1) Count the iteration from this given value.
--prev_iter=21001

# (Optional: str) Customize your wandb job name
--wandb_name=my_job_1

# (Optional: str) Load pretrained synthetic image, format is "wandb_name/images_#.pt"
--load_syn_image=continue-5ipc-epoch-all/images_21000.pt
```