# GZoo2-Aug Dataset Evaluation

> example
> 
> example

<details>
<summary>5 Net ACC: </summary>


</details>

## 1 IPC



### No Augmentation

<details>
<summary> Detail </summary>

> args.lr_net = [mean_v] * 10

<details>
<summary>5 Net ACC: 0.5113333333333333</summary>

ssh://sbcaesar@dais10.uwb.edu:22/data/sbcaesar/xuan_venv/bin/python3 -u /data/sbcaesar/mac_galaxy/evaluate_synthetic_dataset.py --dataset=GZoo2_aug --ipc=1 --syn_steps=50 --data_path=/data/sbcaesar/gzoo2_500ipc --num_eval=5 --gpu=1
BUILDING DATASET
100%|█████████████████████████████████| 45000/45000 [00:00<00:00, 162516.17it/s]
45000it [00:00, 4497645.18it/s]
Loading test:
Load test!
Current lr schedule:
[[0, 0.00033955107210204005], [50, 0.00033955107210204005], [100, 0.00033955107210204005], [150, 0.00033955107210204005], [200, 0.00033955107210204005], [250, 0.00033955107210204005], [300, 0.00033955107210204005], [350, 0.00033955107210204005], [400, 0.00033955107210204005], [450, 0.00033955107210204005], [501, 3.395510721020401e-05]]
100%|███████████████████████████████████████| 1001/1001 [00:20<00:00, 48.19it/s]
[2023-04-27 03:30:31] Evaluate_00: epoch = 1000 train time = 20 s train loss = 0.005240, validation acc = 0.5396, test acc = 0.5222
100%|███████████████████████████████████████| 1001/1001 [00:19<00:00, 50.94it/s]
[2023-04-27 03:31:04] Evaluate_01: epoch = 1000 train time = 19 s train loss = 0.218888, validation acc = 0.4990, test acc = 0.4967
100%|███████████████████████████████████████| 1001/1001 [00:20<00:00, 48.28it/s]
[2023-04-27 03:31:40] Evaluate_02: epoch = 1000 train time = 20 s train loss = 0.124112, validation acc = 0.5164, test acc = 0.5133
100%|███████████████████████████████████████| 1001/1001 [00:19<00:00, 50.62it/s]
[2023-04-27 03:32:13] Evaluate_03: epoch = 1000 train time = 19 s train loss = 0.000369, validation acc = 0.5268, test acc = 0.5244
100%|███████████████████████████████████████| 1001/1001 [00:19<00:00, 50.67it/s]
[2023-04-27 03:32:47] Evaluate_04: epoch = 1000 train time = 19 s train loss = 0.004402, validation acc = 0.5040, test acc = 0.5000
Evaluate 5 random ConvNet, train set mean = 0.5172 std = 0.0148
Evaluate 5 random ConvNet, test set mean = 0.5113 std = 0.0113

[0.5113333333333333]
Mean test accuracy of 10 ramdom sets: 0.5113333333333333

Process finished with exit code 0


</details>

> list lr

<details>
<summary>5 Net ACC: 0.49711111111111117</summary>

ssh://sbcaesar@dais10.uwb.edu:22/data/sbcaesar/xuan_venv/bin/python3 -u /data/sbcaesar/mac_galaxy/evaluate_synthetic_dataset.py --dataset=GZoo2_aug --ipc=1 --syn_steps=50 --data_path=/data/sbcaesar/gzoo2_500ipc --num_eval=5 --gpu=0
BUILDING DATASET
100%|█████████████████████████████████| 45000/45000 [00:00<00:00, 169945.96it/s]
45000it [00:00, 4431747.16it/s]
Loading test:
Load test!
Current lr schedule:
[[0, 0.0007435352890752256], [50, 0.0005036790971644223], [100, 0.00040825619362294674], [150, 0.0003518034936860204], [200, 0.0003059091977775097], [250, 0.00027298444183543324], [300, 0.00023740495089441538], [350, 0.00020283406774979085], [400, 0.00019477325258776546], [450, 0.00017433073662687093], [501, 1.7433073662687092e-05]]
100%|███████████████████████████████████████| 1001/1001 [00:23<00:00, 42.45it/s]
[2023-04-27 03:39:23] Evaluate_00: epoch = 1000 train time = 23 s train loss = 0.184967, validation acc = 0.5406, test acc = 0.5244
100%|███████████████████████████████████████| 1001/1001 [00:20<00:00, 48.16it/s]
[2023-04-27 03:39:58] Evaluate_01: epoch = 1000 train time = 20 s train loss = 0.003867, validation acc = 0.5398, test acc = 0.5078
100%|███████████████████████████████████████| 1001/1001 [00:21<00:00, 46.60it/s]
[2023-04-27 03:40:35] Evaluate_02: epoch = 1000 train time = 21 s train loss = 0.016908, validation acc = 0.5109, test acc = 0.4744
100%|███████████████████████████████████████| 1001/1001 [00:23<00:00, 42.52it/s]
[2023-04-27 03:41:14] Evaluate_03: epoch = 1000 train time = 23 s train loss = 0.022402, validation acc = 0.4937, test acc = 0.4722
100%|███████████████████████████████████████| 1001/1001 [00:22<00:00, 45.03it/s]
[2023-04-27 03:41:53] Evaluate_04: epoch = 1000 train time = 22 s train loss = 0.034356, validation acc = 0.5135, test acc = 0.5067
Evaluate 5 random ConvNet, train set mean = 0.5197 std = 0.0181
Evaluate 5 random ConvNet, test set mean = 0.4971 std = 0.0204

[0.49711111111111117]
Mean test accuracy of 10 ramdom sets: 0.49711111111111117

Process finished with exit code 0

</details>

> list lr + 1~0.1 fine tune

<details>
<summary>5 Net ACC: 0.4993333333333333</summary>

ssh://sbcaesar@dais10.uwb.edu:22/data/sbcaesar/xuan_venv/bin/python3 -u /data/sbcaesar/mac_galaxy/evaluate_synthetic_dataset.py --dataset=GZoo2_aug --ipc=1 --syn_steps=50 --data_path=/data/sbcaesar/gzoo2_500ipc --num_eval=5 --gpu=1
BUILDING DATASET
100%|█████████████████████████████████| 45000/45000 [00:00<00:00, 156292.77it/s]
45000it [00:00, 4401466.35it/s]
Loading test:
Load test!
Current lr schedule:
[[0, 0.0007435352890752256], [50, 0.0005036790971644223], [100, 0.00040825619362294674], [150, 0.0003518034936860204], [200, 0.0003059091977775097], [250, 0.00027298444183543324], [300, 0.00023740495089441538], [350, 0.00020283406774979085], [400, 0.00019477325258776546], [450, 0.00017433073662687093], [500, 0.00013946458930149674], [550, 0.0001115716714411974], [600, 8.925733715295793e-05], [650, 7.140586972236635e-05], [700, 5.712469577789308e-05], [750, 4.569975662231447e-05], [800, 3.6559805297851576e-05], [850, 2.9247844238281263e-05], [900, 2.3398275390625013e-05], [950, 1.871862031250001e-05], [1001, 1.8718620312500012e-06]]
100%|███████████████████████████████████████| 1501/1501 [00:25<00:00, 57.99it/s]
[2023-04-27 03:39:47] Evaluate_00: epoch = 1500 train time = 25 s train loss = 0.296115, validation acc = 0.4933, test acc = 0.4811
100%|███████████████████████████████████████| 1501/1501 [00:24<00:00, 61.47it/s]
[2023-04-27 03:40:26] Evaluate_01: epoch = 1500 train time = 24 s train loss = 0.001598, validation acc = 0.5103, test acc = 0.4989
100%|███████████████████████████████████████| 1501/1501 [00:24<00:00, 61.62it/s]
[2023-04-27 03:41:05] Evaluate_02: epoch = 1500 train time = 24 s train loss = 0.004384, validation acc = 0.5116, test acc = 0.4856
100%|███████████████████████████████████████| 1501/1501 [00:25<00:00, 59.77it/s]
[2023-04-27 03:41:48] Evaluate_03: epoch = 1500 train time = 25 s train loss = 0.002061, validation acc = 0.5274, test acc = 0.5144
100%|███████████████████████████████████████| 1501/1501 [00:24<00:00, 61.40it/s]
[2023-04-27 03:42:28] Evaluate_04: epoch = 1500 train time = 24 s train loss = 0.163811, validation acc = 0.5337, test acc = 0.5167
Evaluate 5 random ConvNet, train set mean = 0.5152 std = 0.0142
Evaluate 5 random ConvNet, test set mean = 0.4993 std = 0.0145

[0.4993333333333333]
Mean test accuracy of 10 ramdom sets: 0.4993333333333333

Process finished with exit code 0

</details>

</details>

### Augmentation

<details>
<summary> Detail </summary>

> --rotate --transpose
> 
> args.lr_net = [mean_v] * 10

<details>
<summary>5 Net ACC: 0.5451111111111111</summary>

ssh://sbcaesar@dais10.uwb.edu:22/data/sbcaesar/xuan_venv/bin/python3 -u /data/sbcaesar/mac_galaxy/baseline_test.py --dataset=GZoo2_aug --ipc=1 --syn_steps=50 --data_path=/data/sbcaesar/gzoo2_500ipc --num_eval=5 --gpu=1 --rotate --transpose
BUILDING DATASET
100%|█████████████████████████████████| 45000/45000 [00:00<00:00, 169750.62it/s]
45000it [00:00, 4409486.96it/s]
Loading test:
Load test!
Transposing images for augmentation
Rotating images for augmentation
Current lr schedule:
[[0, 0.00033955107210204005], [50, 0.00033955107210204005], [100, 0.00033955107210204005], [150, 0.00033955107210204005], [200, 0.00033955107210204005], [250, 0.00033955107210204005], [300, 0.00033955107210204005], [350, 0.00033955107210204005], [400, 0.00033955107210204005], [450, 0.00033955107210204005], [501, 3.395510721020401e-05]]
100%|███████████████████████████████████████| 1001/1001 [00:55<00:00, 18.17it/s]
[2023-04-27 03:18:12] Evaluate_00: epoch = 1000 train time = 55 s train loss = 0.004096, validation acc = 0.5326, test acc = 0.5356
100%|███████████████████████████████████████| 1001/1001 [00:54<00:00, 18.44it/s]
[2023-04-27 03:19:20] Evaluate_01: epoch = 1000 train time = 54 s train loss = 0.006969, validation acc = 0.5778, test acc = 0.5600
100%|███████████████████████████████████████| 1001/1001 [00:54<00:00, 18.44it/s]
[2023-04-27 03:20:29] Evaluate_02: epoch = 1000 train time = 54 s train loss = 0.138751, validation acc = 0.5663, test acc = 0.5511
100%|███████████████████████████████████████| 1001/1001 [00:54<00:00, 18.47it/s]
[2023-04-27 03:21:38] Evaluate_03: epoch = 1000 train time = 54 s train loss = 0.002086, validation acc = 0.5575, test acc = 0.5511
100%|███████████████████████████████████████| 1001/1001 [00:54<00:00, 18.45it/s]
[2023-04-27 03:22:48] Evaluate_04: epoch = 1000 train time = 54 s train loss = 0.001665, validation acc = 0.5466, test acc = 0.5278
Evaluate 5 random ConvNet, train set mean = 0.5562 std = 0.0156
Evaluate 5 random ConvNet, test set mean = 0.5451 std = 0.0117

[0.5451111111111111]
Mean test accuracy of 10 ramdom sets: 0.5451111111111111

Process finished with exit code 0

</details>

> --rotate --transpose --flip_h --flip_v
> 
> args.lr_net = [mean_v] * 10

<details>
<summary>5 Net ACC: 0.5388888888888889</summary>

ssh://sbcaesar@dais10.uwb.edu:22/data/sbcaesar/xuan_venv/bin/python3 -u /data/sbcaesar/mac_galaxy/evaluate_synthetic_dataset.py --dataset=GZoo2_aug --ipc=1 --syn_steps=50 --data_path=/data/sbcaesar/gzoo2_500ipc --num_eval=5 --gpu=0 --rotate --transpose --flip_h --flip_v
BUILDING DATASET
100%|█████████████████████████████████| 45000/45000 [00:00<00:00, 179493.48it/s]
45000it [00:00, 4393372.59it/s]
Loading test:
Load test!
Flipping images horizontally for augmentation
Flipping images vertically for augmentation
Transposing images for augmentation
Rotating images for augmentation
Current lr schedule:
[[0, 0.00033955107210204005], [50, 0.00033955107210204005], [100, 0.00033955107210204005], [150, 0.00033955107210204005], [200, 0.00033955107210204005], [250, 0.00033955107210204005], [300, 0.00033955107210204005], [350, 0.00033955107210204005], [400, 0.00033955107210204005], [450, 0.00033955107210204005], [501, 3.395510721020401e-05]]
100%|███████████████████████████████████████| 1001/1001 [01:33<00:00, 10.66it/s]
[2023-04-27 03:25:53] Evaluate_00: epoch = 1000 train time = 93 s train loss = 0.154476, validation acc = 0.5452, test acc = 0.5267
100%|███████████████████████████████████████| 1001/1001 [01:33<00:00, 10.76it/s]
[2023-04-27 03:27:40] Evaluate_01: epoch = 1000 train time = 93 s train loss = 0.004649, validation acc = 0.5667, test acc = 0.5556
100%|███████████████████████████████████████| 1001/1001 [01:33<00:00, 10.75it/s]
[2023-04-27 03:29:28] Evaluate_02: epoch = 1000 train time = 93 s train loss = 0.056567, validation acc = 0.5264, test acc = 0.5156
100%|███████████████████████████████████████| 1001/1001 [01:35<00:00, 10.53it/s]
[2023-04-27 03:31:17] Evaluate_03: epoch = 1000 train time = 95 s train loss = 0.053179, validation acc = 0.5648, test acc = 0.5544
100%|███████████████████████████████████████| 1001/1001 [01:34<00:00, 10.56it/s]
[2023-04-27 03:33:08] Evaluate_04: epoch = 1000 train time = 94 s train loss = 0.004306, validation acc = 0.5653, test acc = 0.5422
Evaluate 5 random ConvNet, train set mean = 0.5537 std = 0.0158
Evaluate 5 random ConvNet, test set mean = 0.5389 std = 0.0157

[0.5388888888888889]
Mean test accuracy of 10 ramdom sets: 0.5388888888888889

Process finished with exit code 0

</details>


> --rotate --transpose
> 
> list lr

<details>
<summary>5 Net ACC: 0.5431111111111111</summary>

ssh://sbcaesar@dais10.uwb.edu:22/data/sbcaesar/xuan_venv/bin/python3 -u /data/sbcaesar/mac_galaxy/baseline_test.py --dataset=GZoo2_aug --ipc=1 --syn_steps=50 --data_path=/data/sbcaesar/gzoo2_500ipc --num_eval=5 --rotate --transpose
BUILDING DATASET
100%|█████████████████████████████████| 45000/45000 [00:00<00:00, 170938.70it/s]
45000it [00:00, 4402390.32it/s]
Loading test:
Load test!
Transposing images for augmentation
Rotating images for augmentation
Current lr schedule:
[[0, 0.0007435352890752256], [50, 0.0005036790971644223], [100, 0.00040825619362294674], [150, 0.0003518034936860204], [200, 0.0003059091977775097], [250, 0.00027298444183543324], [300, 0.00023740495089441538], [350, 0.00020283406774979085], [400, 0.00019477325258776546], [450, 0.00017433073662687093], [501, 1.7433073662687092e-05]]
100%|███████████████████████████████████████| 1001/1001 [00:34<00:00, 29.27it/s]
[2023-04-27 03:14:09] Evaluate_00: epoch = 1000 train time = 34 s train loss = 0.002426, validation acc = 0.5504, test acc = 0.5411
100%|███████████████████████████████████████| 1001/1001 [00:24<00:00, 41.08it/s]
[2023-04-27 03:14:48] Evaluate_01: epoch = 1000 train time = 24 s train loss = 0.217494, validation acc = 0.5708, test acc = 0.5411
100%|███████████████████████████████████████| 1001/1001 [00:23<00:00, 42.59it/s]
[2023-04-27 03:15:27] Evaluate_02: epoch = 1000 train time = 23 s train loss = 0.005356, validation acc = 0.5427, test acc = 0.5433
100%|███████████████████████████████████████| 1001/1001 [00:22<00:00, 44.89it/s]
[2023-04-27 03:16:04] Evaluate_03: epoch = 1000 train time = 22 s train loss = 0.176115, validation acc = 0.5518, test acc = 0.5433
100%|███████████████████████████████████████| 1001/1001 [00:22<00:00, 44.36it/s]
[2023-04-27 03:16:43] Evaluate_04: epoch = 1000 train time = 22 s train loss = 0.005131, validation acc = 0.5689, test acc = 0.5467
Evaluate 5 random ConvNet, train set mean = 0.5570 std = 0.0110
Evaluate 5 random ConvNet, test set mean = 0.5431 std = 0.0020

[0.5431111111111111]
Mean test accuracy of 10 ramdom sets: 0.5431111111111111

Process finished with exit code 0

</details>

> --rotate --transpose
> 
> list lr + 1~0.1 fine tune

<details>
<summary>5 Net ACC: 0.5464444444444444</summary>

ssh://sbcaesar@dais10.uwb.edu:22/data/sbcaesar/xuan_venv/bin/python3 -u /data/sbcaesar/mac_galaxy/baseline_test.py --dataset=GZoo2_aug --ipc=1 --syn_steps=50 --data_path=/data/sbcaesar/gzoo2_500ipc --num_eval=5 --rotate --transpose
BUILDING DATASET
100%|█████████████████████████████████| 45000/45000 [00:00<00:00, 173978.80it/s]
45000it [00:00, 4379305.32it/s]
Loading test:
Load test!
Transposing images for augmentation
Rotating images for augmentation
Current lr schedule:
[[0, 0.0007435352890752256], [50, 0.0005036790971644223], [100, 0.00040825619362294674], [150, 0.0003518034936860204], [200, 0.0003059091977775097], [250, 0.00027298444183543324], [300, 0.00023740495089441538], [350, 0.00020283406774979085], [400, 0.00019477325258776546], [450, 0.00017433073662687093], [500, 0.00013946458930149674], [550, 0.0001115716714411974], [600, 8.925733715295793e-05], [650, 7.140586972236635e-05], [700, 5.712469577789308e-05], [750, 4.569975662231447e-05], [800, 3.6559805297851576e-05], [850, 2.9247844238281263e-05], [900, 2.3398275390625013e-05], [950, 1.871862031250001e-05], [1001, 1.8718620312500012e-06]]
100%|███████████████████████████████████████| 1501/1501 [00:42<00:00, 35.62it/s]
[2023-04-27 03:08:03] Evaluate_00: epoch = 1500 train time = 42 s train loss = 0.004843, validation acc = 0.5627, test acc = 0.5367
100%|███████████████████████████████████████| 1501/1501 [00:28<00:00, 52.94it/s]
[2023-04-27 03:08:47] Evaluate_01: epoch = 1500 train time = 28 s train loss = 0.183206, validation acc = 0.5762, test acc = 0.5611
100%|███████████████████████████████████████| 1501/1501 [00:29<00:00, 50.83it/s]
[2023-04-27 03:09:33] Evaluate_02: epoch = 1500 train time = 29 s train loss = 0.001114, validation acc = 0.5551, test acc = 0.5311
100%|███████████████████████████████████████| 1501/1501 [00:28<00:00, 51.87it/s]
[2023-04-27 03:10:17] Evaluate_03: epoch = 1500 train time = 28 s train loss = 0.209205, validation acc = 0.5960, test acc = 0.5789
100%|███████████████████████████████████████| 1501/1501 [00:31<00:00, 47.97it/s]
[2023-04-27 03:11:05] Evaluate_04: epoch = 1500 train time = 31 s train loss = 0.002885, validation acc = 0.5425, test acc = 0.5244
Evaluate 5 random ConvNet, train set mean = 0.5665 std = 0.0184
Evaluate 5 random ConvNet, test set mean = 0.5464 std = 0.0204

[0.5464444444444444]
Mean test accuracy of 10 ramdom sets: 0.5464444444444444

Process finished with exit code 0

</details>

> --rotate --transpose
> 
> list lr + 1~0.01 fine tune

<details>
<summary>5 Net ACC: 0.5377777777777777</summary>

ssh://sbcaesar@dais10.uwb.edu:22/data/sbcaesar/xuan_venv/bin/python3 -u /data/sbcaesar/mac_galaxy/evaluate_synthetic_dataset.py --dataset=GZoo2_aug --ipc=1 --syn_steps=50 --data_path=/data/sbcaesar/gzoo2_500ipc --num_eval=5 --gpu=2 --rotate --transpose
BUILDING DATASET
100%|█████████████████████████████████| 45000/45000 [00:00<00:00, 153989.61it/s]
45000it [00:00, 3770047.94it/s]
Loading test:
Load test!
Transposing images for augmentation
Rotating images for augmentation
Current lr schedule:
[[0, 0.0007435352890752256], [50, 0.0005036790971644223], [100, 0.00040825619362294674], [150, 0.0003518034936860204], [200, 0.0003059091977775097], [250, 0.00027298444183543324], [300, 0.00023740495089441538], [350, 0.00020283406774979085], [400, 0.00019477325258776546], [450, 0.00017433073662687093], [500, 0.00013946458930149674], [550, 0.0001115716714411974], [600, 8.925733715295793e-05], [650, 7.140586972236635e-05], [700, 5.712469577789308e-05], [750, 4.569975662231447e-05], [800, 3.6559805297851576e-05], [850, 2.9247844238281263e-05], [900, 2.3398275390625013e-05], [950, 1.871862031250001e-05], [1000, 1.497489625000001e-05], [1050, 1.1979917000000009e-05], [1100, 9.583933600000008e-06], [1150, 7.667146880000007e-06], [1200, 6.133717504000006e-06], [1250, 4.906974003200005e-06], [1300, 3.925579202560005e-06], [1350, 3.140463362048004e-06], [1400, 2.5123706896384034e-06], [1450, 2.009896551710723e-06], [1501, 2.009896551710723e-07]]
100%|███████████████████████████████████████| 2001/2001 [01:37<00:00, 20.49it/s]
[2023-04-27 03:42:59] Evaluate_00: epoch = 2000 train time = 97 s train loss = 0.053268, validation acc = 0.5232, test acc = 0.5211
100%|███████████████████████████████████████| 2001/2001 [01:35<00:00, 20.92it/s]
[2023-04-27 03:44:51] Evaluate_01: epoch = 2000 train time = 95 s train loss = 0.156971, validation acc = 0.5615, test acc = 0.5367
100%|███████████████████████████████████████| 2001/2001 [01:36<00:00, 20.67it/s]
[2023-04-27 03:46:43] Evaluate_02: epoch = 2000 train time = 96 s train loss = 0.024989, validation acc = 0.5554, test acc = 0.5556
100%|███████████████████████████████████████| 2001/2001 [01:38<00:00, 20.35it/s]
[2023-04-27 03:48:40] Evaluate_03: epoch = 2000 train time = 98 s train loss = 0.006399, validation acc = 0.5429, test acc = 0.5311
100%|███████████████████████████████████████| 2001/2001 [01:38<00:00, 20.29it/s]
[2023-04-27 03:50:37] Evaluate_04: epoch = 2000 train time = 98 s train loss = 0.003890, validation acc = 0.5610, test acc = 0.5444
Evaluate 5 random ConvNet, train set mean = 0.5488 std = 0.0144
Evaluate 5 random ConvNet, test set mean = 0.5378 std = 0.0117

[0.5377777777777777]
Mean test accuracy of 10 ramdom sets: 0.5377777777777777

Process finished with exit code 0

</details>

> --rotate --transpose --flip_h --flip_v
> 
> list lr

<details>
<summary>5 Net ACC: 0.5462222222222223</summary>

ssh://sbcaesar@dais10.uwb.edu:22/data/sbcaesar/xuan_venv/bin/python3 -u /data/sbcaesar/mac_galaxy/evaluate_synthetic_dataset.py --dataset=GZoo2_aug --ipc=1 --syn_steps=50 --data_path=/data/sbcaesar/gzoo2_500ipc --num_eval=5 --gpu=0 --rotate --transpose --flip_h --flip_v
BUILDING DATASET
100%|█████████████████████████████████| 45000/45000 [00:00<00:00, 166914.59it/s]
45000it [00:00, 3168062.84it/s]
Loading test:
Load test!
Flipping images horizontally for augmentation
Flipping images vertically for augmentation
Transposing images for augmentation
Rotating images for augmentation
Current lr schedule:
[[0, 0.0007435352890752256], [50, 0.0005036790971644223], [100, 0.00040825619362294674], [150, 0.0003518034936860204], [200, 0.0003059091977775097], [250, 0.00027298444183543324], [300, 0.00023740495089441538], [350, 0.00020283406774979085], [400, 0.00019477325258776546], [450, 0.00017433073662687093], [501, 1.7433073662687092e-05]]
100%|███████████████████████████████████████| 1001/1001 [01:34<00:00, 10.58it/s]
[2023-04-27 03:48:40] Evaluate_00: epoch = 1000 train time = 94 s train loss = 0.002485, validation acc = 0.5566, test acc = 0.5600
100%|███████████████████████████████████████| 1001/1001 [01:33<00:00, 10.65it/s]
[2023-04-27 03:50:30] Evaluate_01: epoch = 1000 train time = 93 s train loss = 0.143429, validation acc = 0.5576, test acc = 0.5311
100%|███████████████████████████████████████| 1001/1001 [01:33<00:00, 10.72it/s]
[2023-04-27 03:52:19] Evaluate_02: epoch = 1000 train time = 93 s train loss = 0.170227, validation acc = 0.5646, test acc = 0.5411
100%|███████████████████████████████████████| 1001/1001 [01:34<00:00, 10.58it/s]
[2023-04-27 03:54:08] Evaluate_03: epoch = 1000 train time = 94 s train loss = 0.142622, validation acc = 0.5328, test acc = 0.5356
100%|███████████████████████████████████████| 1001/1001 [01:33<00:00, 10.69it/s]
[2023-04-27 03:55:59] Evaluate_04: epoch = 1000 train time = 93 s train loss = 0.072360, validation acc = 0.5759, test acc = 0.5633
Evaluate 5 random ConvNet, train set mean = 0.5575 std = 0.0141
Evaluate 5 random ConvNet, test set mean = 0.5462 std = 0.0130

[0.5462222222222223]
Mean test accuracy of 10 ramdom sets: 0.5462222222222223

Process finished with exit code 0

</details>

</details>