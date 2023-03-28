# WanDB Guide

Stagewise-MTT algorithm generates an independent `loss` and `synthetic lr` for each individual epoch. When the values log into WanDB server, each value will be plotted in an individual plot. In this guide, we will show you how to plot them in a same plot for better visualization. For example:

![WanDB](docs/wandb_example.png)

In general, you could follow the [official Line Plot documentation](https://docs.wandb.ai/guides/app/features/panels/line-plot) to do configuration.

Here, we will give instruction to generate the exact example showing above.

* Delete all individual plot in the format of `Grand_Loss_epoch_` and `Synthetic_LR_`. (The data is still saved in the backend of WanDB)
* Manually add some number of new panels (based on how many epochs you used). **Note that each line plot can hold at most 10 lines**. Edit each panel, select the regular expression tab in `y`, and type something like `^Grand_Loss_epoch_[0-9]$`.
