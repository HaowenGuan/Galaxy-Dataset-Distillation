

# lr 0.001
```cmd
ssh://sbcaesar@dais10.uwb.edu:22/data/sbcaesar/xuan_venv/bin/python3 -u /data/sbcaesar/mac_galaxy/buffer.py
Hyper-parameters: 
 {'dataset': 'gzoo2', 'subset': 'imagenette', 'model': 'ConvNet', 'num_experts': 10, 'lr_teacher': 0.001, 'batch_train': 256, 'batch_real': 256, 'dsa': True, 'dsa_strategy': 'color_crop_cutout_flip_scale_rotate', 'data_path': 'data', 'buffer_path': '/data/sbcaesar/galaxy_buffers', 'train_epochs': 50, 'zca': False, 'decay': False, 'mom': 0, 'l2': 0, 'save_interval': 10, 'device': 'cuda', 'dsa_param': <utils.ParamDiffAug object at 0x7faf1af48af0>}
BUILDING DATASET
100%|█████████████████████████████████████| 800/800 [00:00<00:00, 162727.60it/s]
800it [00:00, 4028143.10it/s]
class c = 0: 80 real images
class c = 1: 80 real images
class c = 2: 80 real images
class c = 3: 80 real images
class c = 4: 80 real images
class c = 5: 80 real images
class c = 6: 80 real images
class c = 7: 80 real images
class c = 8: 80 real images
class c = 9: 80 real images
real images channel 0, mean = 0.0003, std = 1.0003
real images channel 1, mean = -0.0002, std = 1.0004
real images channel 2, mean = 0.0006, std = 1.0005
Add weight to loss function tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
DC augmentation parameters: 
 {'crop': 4, 'scale': 0.2, 'rotate': 45, 'noise': 0.001, 'strategy': 'crop_scale_rotate'}
DataParallel(
  (module): ConvNet(
    (features): Sequential(
      (0): Conv2d(3, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): GroupNorm(128, 128, eps=1e-05, affine=True)
      (2): ReLU(inplace=True)
      (3): AvgPool2d(kernel_size=2, stride=2, padding=0)
      (4): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (5): GroupNorm(128, 128, eps=1e-05, affine=True)
      (6): ReLU(inplace=True)
      (7): AvgPool2d(kernel_size=2, stride=2, padding=0)
      (8): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (9): GroupNorm(128, 128, eps=1e-05, affine=True)
      (10): ReLU(inplace=True)
      (11): AvgPool2d(kernel_size=2, stride=2, padding=0)
    )
    (classifier): Linear(in_features=32768, out_features=10, bias=True)
  )
)
Itr: 0	Epoch: 0	Train Acc: 0.12375	Test Acc: 0.155	AVG Train loss: 2.2730748558044436	AVG Test loss: 2.21215765953064
Itr: 0	Epoch: 1	Train Acc: 0.20125	Test Acc: 0.285	AVG Train loss: 2.14262077331543	AVG Test loss: 2.1109929513931274
Itr: 0	Epoch: 2	Train Acc: 0.3175	Test Acc: 0.24	AVG Train loss: 1.9872863721847533	AVG Test loss: 2.087538995742798
Itr: 0	Epoch: 3	Train Acc: 0.3375	Test Acc: 0.36	AVG Train loss: 1.9618958282470702	AVG Test loss: 1.8880829524993896
Itr: 0	Epoch: 4	Train Acc: 0.3525	Test Acc: 0.38	AVG Train loss: 1.90335036277771	AVG Test loss: 1.9420484972000123
Itr: 0	Epoch: 5	Train Acc: 0.46125	Test Acc: 0.345	AVG Train loss: 1.7820195960998535	AVG Test loss: 1.7984064769744874
Itr: 0	Epoch: 6	Train Acc: 0.40125	Test Acc: 0.37	AVG Train loss: 1.8039379405975342	AVG Test loss: 1.757343430519104
Itr: 0	Epoch: 7	Train Acc: 0.50375	Test Acc: 0.375	AVG Train loss: 1.6377677392959595	AVG Test loss: 1.7087648963928224
Itr: 0	Epoch: 8	Train Acc: 0.40375	Test Acc: 0.43	AVG Train loss: 1.8141834592819215	AVG Test loss: 1.6667536449432374
Itr: 0	Epoch: 9	Train Acc: 0.45125	Test Acc: 0.485	AVG Train loss: 1.675412278175354	AVG Test loss: 1.6280900526046753
Itr: 0	Epoch: 10	Train Acc: 0.5125	Test Acc: 0.425	AVG Train loss: 1.6088180351257324	AVG Test loss: 1.5903504657745362
Itr: 0	Epoch: 11	Train Acc: 0.57625	Test Acc: 0.395	AVG Train loss: 1.4347108030319213	AVG Test loss: 1.576309757232666
Itr: 0	Epoch: 12	Train Acc: 0.525	Test Acc: 0.49	AVG Train loss: 1.4556394958496093	AVG Test loss: 1.5212853336334229
Itr: 0	Epoch: 13	Train Acc: 0.5475	Test Acc: 0.485	AVG Train loss: 1.4641204118728637	AVG Test loss: 1.4839938163757325
Itr: 0	Epoch: 14	Train Acc: 0.5125	Test Acc: 0.49	AVG Train loss: 1.5020700120925903	AVG Test loss: 1.4572901010513306
Itr: 0	Epoch: 15	Train Acc: 0.4425	Test Acc: 0.445	AVG Train loss: 1.6621863889694213	AVG Test loss: 1.5179237174987792
Itr: 0	Epoch: 16	Train Acc: 0.51625	Test Acc: 0.53	AVG Train loss: 1.4915644598007203	AVG Test loss: 1.4332083749771118
Itr: 0	Epoch: 17	Train Acc: 0.56875	Test Acc: 0.47	AVG Train loss: 1.3840842485427856	AVG Test loss: 1.4858513832092286
Itr: 0	Epoch: 18	Train Acc: 0.48625	Test Acc: 0.42	AVG Train loss: 1.523692626953125	AVG Test loss: 1.5191073608398438
Itr: 0	Epoch: 19	Train Acc: 0.5275	Test Acc: 0.525	AVG Train loss: 1.4479081153869628	AVG Test loss: 1.4076718139648436
Itr: 0	Epoch: 20	Train Acc: 0.59875	Test Acc: 0.445	AVG Train loss: 1.3053396987915038	AVG Test loss: 1.4607751369476318
Itr: 0	Epoch: 21	Train Acc: 0.5675	Test Acc: 0.535	AVG Train loss: 1.3296700048446655	AVG Test loss: 1.4320677089691163
Itr: 0	Epoch: 22	Train Acc: 0.62125	Test Acc: 0.545	AVG Train loss: 1.2735836553573607	AVG Test loss: 1.3554812717437743
Itr: 0	Epoch: 23	Train Acc: 0.57125	Test Acc: 0.535	AVG Train loss: 1.3619678497314454	AVG Test loss: 1.3358476781845092
Itr: 0	Epoch: 24	Train Acc: 0.63375	Test Acc: 0.535	AVG Train loss: 1.2111126756668091	AVG Test loss: 1.3230192565917969
Itr: 0	Epoch: 25	Train Acc: 0.64875	Test Acc: 0.605	AVG Train loss: 1.161857738494873	AVG Test loss: 1.2868883228302002
Itr: 0	Epoch: 26	Train Acc: 0.67375	Test Acc: 0.585	AVG Train loss: 1.11821280002594	AVG Test loss: 1.2630502939224244
Itr: 0	Epoch: 27	Train Acc: 0.60125	Test Acc: 0.51	AVG Train loss: 1.2188308715820313	AVG Test loss: 1.3220414161682128
Itr: 0	Epoch: 28	Train Acc: 0.66	Test Acc: 0.57	AVG Train loss: 1.1566712379455566	AVG Test loss: 1.276940836906433
Itr: 0	Epoch: 29	Train Acc: 0.51125	Test Acc: 0.62	AVG Train loss: 1.4791321277618408	AVG Test loss: 1.2430059862136842
Itr: 0	Epoch: 30	Train Acc: 0.61875	Test Acc: 0.585	AVG Train loss: 1.249719614982605	AVG Test loss: 1.2972172784805298
Itr: 0	Epoch: 31	Train Acc: 0.54625	Test Acc: 0.535	AVG Train loss: 1.34027925491333	AVG Test loss: 1.2836409139633178
Itr: 0	Epoch: 32	Train Acc: 0.655	Test Acc: 0.53	AVG Train loss: 1.0919824981689452	AVG Test loss: 1.3091031217575073
Itr: 0	Epoch: 33	Train Acc: 0.6775	Test Acc: 0.615	AVG Train loss: 1.051972336769104	AVG Test loss: 1.230714716911316
Itr: 0	Epoch: 34	Train Acc: 0.715	Test Acc: 0.53	AVG Train loss: 1.0042644071578979	AVG Test loss: 1.3565219974517821
Itr: 0	Epoch: 35	Train Acc: 0.61	Test Acc: 0.57	AVG Train loss: 1.220001049041748	AVG Test loss: 1.245973868370056
Itr: 0	Epoch: 36	Train Acc: 0.57125	Test Acc: 0.47	AVG Train loss: 1.3231479716300965	AVG Test loss: 1.3918897914886474
Itr: 0	Epoch: 37	Train Acc: 0.5175	Test Acc: 0.57	AVG Train loss: 1.3685345458984375	AVG Test loss: 1.2388555192947388
Itr: 0	Epoch: 38	Train Acc: 0.5875	Test Acc: 0.57	AVG Train loss: 1.2303138208389282	AVG Test loss: 1.226618537902832
Itr: 0	Epoch: 39	Train Acc: 0.5325	Test Acc: 0.615	AVG Train loss: 1.38200984954834	AVG Test loss: 1.1955835437774658
Itr: 0	Epoch: 40	Train Acc: 0.72375	Test Acc: 0.55	AVG Train loss: 0.9839879846572877	AVG Test loss: 1.2406506824493408
Itr: 0	Epoch: 41	Train Acc: 0.61375	Test Acc: 0.505	AVG Train loss: 1.2299713277816773	AVG Test loss: 1.3285599374771118
Itr: 0	Epoch: 42	Train Acc: 0.595	Test Acc: 0.57	AVG Train loss: 1.2254820919036866	AVG Test loss: 1.295542049407959
Itr: 0	Epoch: 43	Train Acc: 0.605	Test Acc: 0.625	AVG Train loss: 1.234113211631775	AVG Test loss: 1.151983952522278
Itr: 0	Epoch: 44	Train Acc: 0.70375	Test Acc: 0.54	AVG Train loss: 1.0115490531921387	AVG Test loss: 1.2469945669174194
Itr: 0	Epoch: 45	Train Acc: 0.48625	Test Acc: 0.625	AVG Train loss: 1.4745424222946166	AVG Test loss: 1.143201289176941
Itr: 0	Epoch: 46	Train Acc: 0.54375	Test Acc: 0.635	AVG Train loss: 1.3572304821014405	AVG Test loss: 1.1443696880340577
Itr: 0	Epoch: 47	Train Acc: 0.655	Test Acc: 0.62	AVG Train loss: 1.1126071000099182	AVG Test loss: 1.1134244918823242
Itr: 0	Epoch: 48	Train Acc: 0.73125	Test Acc: 0.67	AVG Train loss: 0.9329723644256592	AVG Test loss: 1.0971665477752686
Itr: 0	Epoch: 49	Train Acc: 0.63375	Test Acc: 0.515	AVG Train loss: 1.1193849802017213	AVG Test loss: 1.232697811126709
train set ACC of each class tensor([0.0312, 0.0063, 0.0862, 0.0825, 0.0962, 0.0875, 0.0913, 0.0712, 0.0350,
        0.0913])
[[25  0  1  5  0  6 38  5  0  0]
 [14  5  0  0  0 12 37 12  0  0]
 [ 0  0 69  6  1  1  3  0  0  0]
 [ 0  0  3 66  6  2  3  0  0  0]
 [ 0  0  0  1 77  0  0  0  0  2]
 [ 0  0  1  0  0 70  8  0  1  0]
 [ 0  0  0  0  0  5 73  1  1  0]
 [ 0  0  0  0  0 18  5 57  0  0]
 [ 1  0  0  1  0 21 24  5 28  0]
 [ 0  0  0  1  1  3  2  0  0 73]]
test set ACC of each class tensor([0.0300, 0.0050, 0.0600, 0.0700, 0.0900, 0.0750, 0.0750, 0.0200, 0.0050,
        0.0850])
[[ 6  0  0  1  0  2  9  2  0  0]
 [ 2  1  0  0  0  5 10  2  0  0]
 [ 0  0 12  5  3  0  0  0  0  0]
 [ 0  0  2 14  1  1  1  0  1  0]
 [ 0  0  1  1 18  0  0  0  0  0]
 [ 0  0  1  0  0 15  3  1  0  0]
 [ 0  0  0  0  0  5 15  0  0  0]
 [ 0  0  0  0  0 15  1  4  0  0]
 [ 0  0  1  0  1  8  9  0  1  0]
 [ 1  0  0  0  1  0  1  0  0 17]]
DataParallel(
  (module): ConvNet(
    (features): Sequential(
      (0): Conv2d(3, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): GroupNorm(128, 128, eps=1e-05, affine=True)
      (2): ReLU(inplace=True)
      (3): AvgPool2d(kernel_size=2, stride=2, padding=0)
      (4): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (5): GroupNorm(128, 128, eps=1e-05, affine=True)
      (6): ReLU(inplace=True)
      (7): AvgPool2d(kernel_size=2, stride=2, padding=0)
      (8): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (9): GroupNorm(128, 128, eps=1e-05, affine=True)
      (10): ReLU(inplace=True)
      (11): AvgPool2d(kernel_size=2, stride=2, padding=0)
    )
    (classifier): Linear(in_features=32768, out_features=10, bias=True)
  )
)
Itr: 1	Epoch: 0	Train Acc: 0.14	Test Acc: 0.16	AVG Train loss: 2.2899906730651853	AVG Test loss: 2.2252873229980468
Itr: 1	Epoch: 1	Train Acc: 0.2	Test Acc: 0.155	AVG Train loss: 2.1544591903686525	AVG Test loss: 2.2149063682556154
Itr: 1	Epoch: 2	Train Acc: 0.21125	Test Acc: 0.19	AVG Train loss: 2.1524337768554687	AVG Test loss: 2.1717884826660154
Itr: 1	Epoch: 3	Train Acc: 0.2875	Test Acc: 0.325	AVG Train loss: 2.0431860733032225	AVG Test loss: 1.9613762092590332
Itr: 1	Epoch: 4	Train Acc: 0.3675	Test Acc: 0.265	AVG Train loss: 1.9163815212249755	AVG Test loss: 1.9052960062026978
Itr: 1	Epoch: 5	Train Acc: 0.42	Test Acc: 0.345	AVG Train loss: 1.8150602912902831	AVG Test loss: 1.8398616600036621
Itr: 1	Epoch: 6	Train Acc: 0.46875	Test Acc: 0.37	AVG Train loss: 1.7485927104949952	AVG Test loss: 1.7997101593017577
Itr: 1	Epoch: 7	Train Acc: 0.47375	Test Acc: 0.435	AVG Train loss: 1.7742134428024292	AVG Test loss: 1.730880117416382
Itr: 1	Epoch: 8	Train Acc: 0.4775	Test Acc: 0.405	AVG Train loss: 1.6789113616943359	AVG Test loss: 1.6977976703643798
Itr: 1	Epoch: 9	Train Acc: 0.52	Test Acc: 0.455	AVG Train loss: 1.6352409172058104	AVG Test loss: 1.7108040142059326
Itr: 1	Epoch: 10	Train Acc: 0.535	Test Acc: 0.45	AVG Train loss: 1.5779453802108765	AVG Test loss: 1.6148549652099609
Itr: 1	Epoch: 11	Train Acc: 0.58875	Test Acc: 0.485	AVG Train loss: 1.488296127319336	AVG Test loss: 1.5898686933517456
Itr: 1	Epoch: 12	Train Acc: 0.515	Test Acc: 0.565	AVG Train loss: 1.5772628831863402	AVG Test loss: 1.5365298700332641
Itr: 1	Epoch: 13	Train Acc: 0.61	Test Acc: 0.51	AVG Train loss: 1.4289153051376342	AVG Test loss: 1.50092435836792
Itr: 1	Epoch: 14	Train Acc: 0.5825	Test Acc: 0.51	AVG Train loss: 1.4535403156280517	AVG Test loss: 1.4903758907318114
Itr: 1	Epoch: 15	Train Acc: 0.5275	Test Acc: 0.455	AVG Train loss: 1.51759126663208	AVG Test loss: 1.5480073499679565
Itr: 1	Epoch: 16	Train Acc: 0.58625	Test Acc: 0.525	AVG Train loss: 1.3592487621307372	AVG Test loss: 1.4464998197555543
Itr: 1	Epoch: 17	Train Acc: 0.62625	Test Acc: 0.43	AVG Train loss: 1.2960195112228394	AVG Test loss: 1.4951637172698975
Itr: 1	Epoch: 18	Train Acc: 0.6175	Test Acc: 0.495	AVG Train loss: 1.2949385261535644	AVG Test loss: 1.4649222660064698
Itr: 1	Epoch: 19	Train Acc: 0.59875	Test Acc: 0.555	AVG Train loss: 1.350493779182434	AVG Test loss: 1.382094259262085
Itr: 1	Epoch: 20	Train Acc: 0.62375	Test Acc: 0.495	AVG Train loss: 1.2800467777252198	AVG Test loss: 1.3987747097015382
Itr: 1	Epoch: 21	Train Acc: 0.60125	Test Acc: 0.59	AVG Train loss: 1.3147735166549683	AVG Test loss: 1.3545394802093507
Itr: 1	Epoch: 22	Train Acc: 0.665	Test Acc: 0.59	AVG Train loss: 1.2483036756515502	AVG Test loss: 1.3373303127288818
Itr: 1	Epoch: 23	Train Acc: 0.6875	Test Acc: 0.62	AVG Train loss: 1.1637018918991089	AVG Test loss: 1.289546675682068
Itr: 1	Epoch: 24	Train Acc: 0.685	Test Acc: 0.61	AVG Train loss: 1.1446172094345093	AVG Test loss: 1.2913629484176636
Itr: 1	Epoch: 25	Train Acc: 0.6575	Test Acc: 0.595	AVG Train loss: 1.2547060012817384	AVG Test loss: 1.2892288208007812
Itr: 1	Epoch: 26	Train Acc: 0.6775	Test Acc: 0.6	AVG Train loss: 1.1230110573768615	AVG Test loss: 1.2389650821685791
Itr: 1	Epoch: 27	Train Acc: 0.555	Test Acc: 0.635	AVG Train loss: 1.3762446737289429	AVG Test loss: 1.2492955446243286
Itr: 1	Epoch: 28	Train Acc: 0.70125	Test Acc: 0.615	AVG Train loss: 1.0817164087295532	AVG Test loss: 1.2383844804763795
Itr: 1	Epoch: 29	Train Acc: 0.58625	Test Acc: 0.59	AVG Train loss: 1.3133090257644653	AVG Test loss: 1.2528095483779906
Itr: 1	Epoch: 30	Train Acc: 0.67	Test Acc: 0.67	AVG Train loss: 1.1478719234466552	AVG Test loss: 1.2212050008773803
Itr: 1	Epoch: 31	Train Acc: 0.7125	Test Acc: 0.555	AVG Train loss: 1.126213071346283	AVG Test loss: 1.25143404006958
Itr: 1	Epoch: 32	Train Acc: 0.58875	Test Acc: 0.635	AVG Train loss: 1.2700254678726197	AVG Test loss: 1.247362699508667
Itr: 1	Epoch: 33	Train Acc: 0.50125	Test Acc: 0.605	AVG Train loss: 1.4796206450462341	AVG Test loss: 1.2060352993011474
Itr: 1	Epoch: 34	Train Acc: 0.66	Test Acc: 0.6	AVG Train loss: 1.1057323169708253	AVG Test loss: 1.2277215576171876
Itr: 1	Epoch: 35	Train Acc: 0.58375	Test Acc: 0.625	AVG Train loss: 1.3246320486068726	AVG Test loss: 1.2493911504745483
Itr: 1	Epoch: 36	Train Acc: 0.74375	Test Acc: 0.605	AVG Train loss: 1.0010387444496154	AVG Test loss: 1.1986258363723754
Itr: 1	Epoch: 37	Train Acc: 0.58375	Test Acc: 0.57	AVG Train loss: 1.280418906211853	AVG Test loss: 1.218112087249756
Itr: 1	Epoch: 38	Train Acc: 0.62375	Test Acc: 0.59	AVG Train loss: 1.1846627902984619	AVG Test loss: 1.177762780189514
Itr: 1	Epoch: 39	Train Acc: 0.60125	Test Acc: 0.645	AVG Train loss: 1.202919363975525	AVG Test loss: 1.1355808448791505
Itr: 1	Epoch: 40	Train Acc: 0.59625	Test Acc: 0.555	AVG Train loss: 1.229364094734192	AVG Test loss: 1.266850757598877
Itr: 1	Epoch: 41	Train Acc: 0.68875	Test Acc: 0.68	AVG Train loss: 1.0523377275466919	AVG Test loss: 1.133506588935852
Itr: 1	Epoch: 42	Train Acc: 0.7375	Test Acc: 0.655	AVG Train loss: 0.9332218003273011	AVG Test loss: 1.1303977346420289
Itr: 1	Epoch: 43	Train Acc: 0.615	Test Acc: 0.665	AVG Train loss: 1.2322908735275269	AVG Test loss: 1.1424233627319336
Itr: 1	Epoch: 44	Train Acc: 0.755	Test Acc: 0.61	AVG Train loss: 0.9064263844490051	AVG Test loss: 1.1490233659744262
Itr: 1	Epoch: 45	Train Acc: 0.73375	Test Acc: 0.585	AVG Train loss: 0.9488489961624146	AVG Test loss: 1.188851180076599
Itr: 1	Epoch: 46	Train Acc: 0.50625	Test Acc: 0.635	AVG Train loss: 1.3934201765060426	AVG Test loss: 1.171079354286194
Itr: 1	Epoch: 47	Train Acc: 0.74625	Test Acc: 0.605	AVG Train loss: 0.8958796167373657	AVG Test loss: 1.1567641305923462
Itr: 1	Epoch: 48	Train Acc: 0.625	Test Acc: 0.65	AVG Train loss: 1.1335295844078064	AVG Test loss: 1.1303832912445069
Itr: 1	Epoch: 49	Train Acc: 0.75	Test Acc: 0.615	AVG Train loss: 0.8826458072662353	AVG Test loss: 1.0995945692062379
train set ACC of each class tensor([0.0737, 0.0437, 0.0812, 0.0950, 0.0850, 0.0662, 0.0913, 0.0900, 0.0475,
        0.0925])
[[59  1  1  7  0  0  6  6  0  0]
 [28 35  0  0  0  0  2 15  0  0]
 [ 2  0 65 13  0  0  0  0  0  0]
 [ 2  0  0 76  1  0  1  0  0  0]
 [ 0  0  0 11 68  0  0  0  0  1]
 [ 2  0  2  3  0 53 11  7  2  0]
 [ 0  0  0  0  0  1 73  4  2  0]
 [ 3  0  0  0  0  1  2 72  2  0]
 [ 8  0  0  6  0  8 15  4 38  1]
 [ 2  0  0  1  0  1  1  1  0 74]]
test set ACC of each class tensor([0.0650, 0.0350, 0.0500, 0.0850, 0.0750, 0.0550, 0.0600, 0.0800, 0.0300,
        0.0800])
[[13  2  0  2  0  0  0  2  1  0]
 [ 9  7  0  0  0  0  1  3  0  0]
 [ 0  0 10  7  3  0  0  0  0  0]
 [ 2  0  0 17  0  0  0  0  1  0]
 [ 0  0  2  3 15  0  0  0  0  0]
 [ 0  0  0  1  0 11  4  4  0  0]
 [ 3  0  0  0  0  4 12  0  1  0]
 [ 0  0  0  0  0  2  1 16  1  0]
 [ 3  1  1  1  0  4  4  0  6  0]
 [ 1  1  0  2  0  0  0  0  0 16]]
DataParallel(
  (module): ConvNet(
    (features): Sequential(
      (0): Conv2d(3, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): GroupNorm(128, 128, eps=1e-05, affine=True)
      (2): ReLU(inplace=True)
      (3): AvgPool2d(kernel_size=2, stride=2, padding=0)
      (4): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (5): GroupNorm(128, 128, eps=1e-05, affine=True)
      (6): ReLU(inplace=True)
      (7): AvgPool2d(kernel_size=2, stride=2, padding=0)
      (8): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (9): GroupNorm(128, 128, eps=1e-05, affine=True)
      (10): ReLU(inplace=True)
      (11): AvgPool2d(kernel_size=2, stride=2, padding=0)
    )
    (classifier): Linear(in_features=32768, out_features=10, bias=True)
  )
)
Itr: 2	Epoch: 0	Train Acc: 0.1275	Test Acc: 0.17	AVG Train loss: 2.2739263439178465	AVG Test loss: 2.238746681213379
Itr: 2	Epoch: 1	Train Acc: 0.21	Test Acc: 0.23	AVG Train loss: 2.1381086778640745	AVG Test loss: 2.048765573501587
Itr: 2	Epoch: 2	Train Acc: 0.31875	Test Acc: 0.29	AVG Train loss: 2.001728448867798	AVG Test loss: 1.9903491926193237
Itr: 2	Epoch: 3	Train Acc: 0.38125	Test Acc: 0.2	AVG Train loss: 1.8897655820846557	AVG Test loss: 2.0670879220962526
Itr: 2	Epoch: 4	Train Acc: 0.33625	Test Acc: 0.26	AVG Train loss: 1.8683085346221924	AVG Test loss: 1.8613887119293213
Itr: 2	Epoch: 5	Train Acc: 0.4375	Test Acc: 0.39	AVG Train loss: 1.747939100265503	AVG Test loss: 1.7842545747756957
Itr: 2	Epoch: 6	Train Acc: 0.43125	Test Acc: 0.405	AVG Train loss: 1.7831233024597168	AVG Test loss: 1.7695464944839479
Itr: 2	Epoch: 7	Train Acc: 0.5025	Test Acc: 0.48	AVG Train loss: 1.6520924186706543	AVG Test loss: 1.6621052598953248
Itr: 2	Epoch: 8	Train Acc: 0.53375	Test Acc: 0.395	AVG Train loss: 1.6231935548782348	AVG Test loss: 1.6735641479492187
Itr: 2	Epoch: 9	Train Acc: 0.4575	Test Acc: 0.41	AVG Train loss: 1.7000051546096802	AVG Test loss: 1.6269296073913575
Itr: 2	Epoch: 10	Train Acc: 0.5675	Test Acc: 0.425	AVG Train loss: 1.4928314542770387	AVG Test loss: 1.568334174156189
Itr: 2	Epoch: 11	Train Acc: 0.5575	Test Acc: 0.38	AVG Train loss: 1.5191726064682007	AVG Test loss: 1.5910138511657714
Itr: 2	Epoch: 12	Train Acc: 0.51875	Test Acc: 0.525	AVG Train loss: 1.4882856845855712	AVG Test loss: 1.495439739227295
Itr: 2	Epoch: 13	Train Acc: 0.5975	Test Acc: 0.445	AVG Train loss: 1.385080132484436	AVG Test loss: 1.5174716377258302
Itr: 2	Epoch: 14	Train Acc: 0.56625	Test Acc: 0.46	AVG Train loss: 1.4138179779052735	AVG Test loss: 1.4788313865661622
Itr: 2	Epoch: 15	Train Acc: 0.595	Test Acc: 0.545	AVG Train loss: 1.34247652053833	AVG Test loss: 1.4184120845794679
Itr: 2	Epoch: 16	Train Acc: 0.65375	Test Acc: 0.465	AVG Train loss: 1.3004496431350707	AVG Test loss: 1.4290567970275878
Itr: 2	Epoch: 17	Train Acc: 0.58125	Test Acc: 0.52	AVG Train loss: 1.360975785255432	AVG Test loss: 1.4237005615234375
Itr: 2	Epoch: 18	Train Acc: 0.53	Test Acc: 0.535	AVG Train loss: 1.5274335622787476	AVG Test loss: 1.4013259887695313
Itr: 2	Epoch: 19	Train Acc: 0.64625	Test Acc: 0.42	AVG Train loss: 1.2607139205932618	AVG Test loss: 1.4888915205001831
Itr: 2	Epoch: 20	Train Acc: 0.6025	Test Acc: 0.525	AVG Train loss: 1.284244089126587	AVG Test loss: 1.4763100624084473
Itr: 2	Epoch: 21	Train Acc: 0.5025	Test Acc: 0.425	AVG Train loss: 1.4898106050491333	AVG Test loss: 1.5110834741592407
Itr: 2	Epoch: 22	Train Acc: 0.525	Test Acc: 0.575	AVG Train loss: 1.4172025918960571	AVG Test loss: 1.3299279880523682
Itr: 2	Epoch: 23	Train Acc: 0.64625	Test Acc: 0.545	AVG Train loss: 1.2418313598632813	AVG Test loss: 1.324843134880066
Itr: 2	Epoch: 24	Train Acc: 0.5775	Test Acc: 0.455	AVG Train loss: 1.3405808544158935	AVG Test loss: 1.5073629570007325
Itr: 2	Epoch: 25	Train Acc: 0.585	Test Acc: 0.565	AVG Train loss: 1.299945411682129	AVG Test loss: 1.3208946084976196
Itr: 2	Epoch: 26	Train Acc: 0.69125	Test Acc: 0.585	AVG Train loss: 1.1269514894485473	AVG Test loss: 1.2457093620300292
Itr: 2	Epoch: 27	Train Acc: 0.57	Test Acc: 0.5	AVG Train loss: 1.3248835754394532	AVG Test loss: 1.3571272230148315
Itr: 2	Epoch: 28	Train Acc: 0.605	Test Acc: 0.605	AVG Train loss: 1.220393614768982	AVG Test loss: 1.2699083948135377
Itr: 2	Epoch: 29	Train Acc: 0.60125	Test Acc: 0.58	AVG Train loss: 1.264723196029663	AVG Test loss: 1.2416142177581788
Itr: 2	Epoch: 30	Train Acc: 0.51125	Test Acc: 0.625	AVG Train loss: 1.4467035293579102	AVG Test loss: 1.2273414325714112
Itr: 2	Epoch: 31	Train Acc: 0.66	Test Acc: 0.605	AVG Train loss: 1.1537170648574828	AVG Test loss: 1.2288614559173583
Itr: 2	Epoch: 32	Train Acc: 0.6975	Test Acc: 0.535	AVG Train loss: 1.0582312059402466	AVG Test loss: 1.3157775497436524
Itr: 2	Epoch: 33	Train Acc: 0.6925	Test Acc: 0.605	AVG Train loss: 1.0688563585281372	AVG Test loss: 1.209010043144226
Itr: 2	Epoch: 34	Train Acc: 0.6775	Test Acc: 0.59	AVG Train loss: 1.0445772695541382	AVG Test loss: 1.2260477685928344
Itr: 2	Epoch: 35	Train Acc: 0.57375	Test Acc: 0.59	AVG Train loss: 1.2861105346679687	AVG Test loss: 1.228338017463684
Itr: 2	Epoch: 36	Train Acc: 0.575	Test Acc: 0.535	AVG Train loss: 1.313712956905365	AVG Test loss: 1.303295087814331
Itr: 2	Epoch: 37	Train Acc: 0.4575	Test Acc: 0.565	AVG Train loss: 1.4961669898033143	AVG Test loss: 1.2614321756362914
Itr: 2	Epoch: 38	Train Acc: 0.71375	Test Acc: 0.545	AVG Train loss: 1.01272869348526	AVG Test loss: 1.2500204992294313
Itr: 2	Epoch: 39	Train Acc: 0.52875	Test Acc: 0.6	AVG Train loss: 1.4365383052825929	AVG Test loss: 1.1791318273544311
Itr: 2	Epoch: 40	Train Acc: 0.64125	Test Acc: 0.655	AVG Train loss: 1.137209436893463	AVG Test loss: 1.134468569755554
Itr: 2	Epoch: 41	Train Acc: 0.69375	Test Acc: 0.605	AVG Train loss: 1.061123378276825	AVG Test loss: 1.146263928413391
Itr: 2	Epoch: 42	Train Acc: 0.72125	Test Acc: 0.585	AVG Train loss: 0.9429457259178161	AVG Test loss: 1.1800102424621581
Itr: 2	Epoch: 43	Train Acc: 0.6325	Test Acc: 0.59	AVG Train loss: 1.1520054268836974	AVG Test loss: 1.1522924613952636
Itr: 2	Epoch: 44	Train Acc: 0.69125	Test Acc: 0.64	AVG Train loss: 1.0214342975616455	AVG Test loss: 1.1638484573364258
Itr: 2	Epoch: 45	Train Acc: 0.645	Test Acc: 0.705	AVG Train loss: 1.1344341945648193	AVG Test loss: 1.1081962585449219
Itr: 2	Epoch: 46	Train Acc: 0.61125	Test Acc: 0.625	AVG Train loss: 1.2005587387084962	AVG Test loss: 1.1062934637069701
Itr: 2	Epoch: 47	Train Acc: 0.68625	Test Acc: 0.54	AVG Train loss: 1.001123971939087	AVG Test loss: 1.218425440788269
Itr: 2	Epoch: 48	Train Acc: 0.72	Test Acc: 0.64	AVG Train loss: 0.938728985786438	AVG Test loss: 1.1296340656280517
Itr: 2	Epoch: 49	Train Acc: 0.7575	Test Acc: 0.65	AVG Train loss: 0.8808252787590027	AVG Test loss: 1.1267661905288697
train set ACC of each class tensor([0.0475, 0.0812, 0.0887, 0.0750, 0.0962, 0.0850, 0.0613, 0.0850, 0.0538,
        0.0938])
[[38 27  1  6  0  3  1  4  0  0]
 [ 4 65  1  1  0  0  0  9  0  0]
 [ 1  2 71  6  0  0  0  0  0  0]
 [ 0  3  7 60  9  1  0  0  0  0]
 [ 0  0  0  2 77  0  0  0  0  1]
 [ 0  1  1  0  0 68  2  7  1  0]
 [ 1  5  0  0  0 20 49  3  2  0]
 [ 0  2  1  0  0  8  0 68  1  0]
 [ 1  8  2  1  0 19  1  5 43  0]
 [ 0  1  0  0  1  1  0  1  1 75]]
test set ACC of each class tensor([0.0400, 0.0800, 0.0750, 0.0750, 0.0850, 0.0800, 0.0350, 0.0800, 0.0250,
        0.0750])
[[ 8  5  0  1  0  4  0  2  0  0]
 [ 0 16  0  0  0  0  0  4  0  0]
 [ 0  0 15  2  2  1  0  0  0  0]
 [ 1  1  2 15  0  0  0  1  0  0]
 [ 0  0  2  1 17  0  0  0  0  0]
 [ 0  0  1  0  0 16  1  2  0  0]
 [ 1  2  0  0  0  8  7  1  1  0]
 [ 0  0  0  0  0  4  0 16  0  0]
 [ 1  3  1  0  0  8  0  1  5  1]
 [ 0  2  0  0  2  0  0  0  1 15]]
```

# lr 0.003
```cmd
ssh://sbcaesar@dais10.uwb.edu:22/data/sbcaesar/xuan_venv/bin/python3 -u /data/sbcaesar/mac_galaxy/buffer.py
Hyper-parameters: 
 {'dataset': 'gzoo2', 'subset': 'imagenette', 'model': 'ConvNet', 'num_experts': 10, 'lr_teacher': 0.005, 'batch_train': 256, 'batch_real': 256, 'dsa': True, 'dsa_strategy': 'color_crop_cutout_flip_scale_rotate', 'data_path': 'data', 'buffer_path': '/data/sbcaesar/galaxy_buffers', 'train_epochs': 50, 'zca': False, 'decay': False, 'mom': 0, 'l2': 0, 'save_interval': 10, 'device': 'cuda', 'dsa_param': <utils.ParamDiffAug object at 0x7f1f9dae85b0>}
BUILDING DATASET
100%|█████████████████████████████████████| 800/800 [00:00<00:00, 136861.90it/s]
800it [00:00, 3874645.73it/s]
class c = 0: 80 real images
class c = 1: 80 real images
class c = 2: 80 real images
class c = 3: 80 real images
class c = 4: 80 real images
class c = 5: 80 real images
class c = 6: 80 real images
class c = 7: 80 real images
class c = 8: 80 real images
class c = 9: 80 real images
real images channel 0, mean = 0.0003, std = 1.0003
real images channel 1, mean = -0.0002, std = 1.0004
real images channel 2, mean = 0.0006, std = 1.0005
Add weight to loss function tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
DC augmentation parameters: 
 {'crop': 4, 'scale': 0.2, 'rotate': 45, 'noise': 0.001, 'strategy': 'crop_scale_rotate'}
DataParallel(
  (module): ConvNet(
    (features): Sequential(
      (0): Conv2d(3, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): GroupNorm(128, 128, eps=1e-05, affine=True)
      (2): ReLU(inplace=True)
      (3): AvgPool2d(kernel_size=2, stride=2, padding=0)
      (4): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (5): GroupNorm(128, 128, eps=1e-05, affine=True)
      (6): ReLU(inplace=True)
      (7): AvgPool2d(kernel_size=2, stride=2, padding=0)
      (8): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (9): GroupNorm(128, 128, eps=1e-05, affine=True)
      (10): ReLU(inplace=True)
      (11): AvgPool2d(kernel_size=2, stride=2, padding=0)
    )
    (classifier): Linear(in_features=32768, out_features=10, bias=True)
  )
)
Itr: 0	Epoch: 0	Train Acc: 0.14125	Test Acc: 0.145	AVG Train loss: 2.6313953590393067	AVG Test loss: 6.61997688293457
Itr: 0	Epoch: 1	Train Acc: 0.205	Test Acc: 0.145	AVG Train loss: 5.103866634368896	AVG Test loss: 4.792423439025879
Itr: 0	Epoch: 2	Train Acc: 0.21875	Test Acc: 0.155	AVG Train loss: 4.089839887619019	AVG Test loss: 3.0807215404510497
Itr: 0	Epoch: 3	Train Acc: 0.25	Test Acc: 0.285	AVG Train loss: 2.7144878005981443	AVG Test loss: 1.9637963914871215
Itr: 0	Epoch: 4	Train Acc: 0.28125	Test Acc: 0.315	AVG Train loss: 1.9713970851898193	AVG Test loss: 2.2001331615448
Itr: 0	Epoch: 5	Train Acc: 0.29375	Test Acc: 0.26	AVG Train loss: 2.0358610057830813	AVG Test loss: 2.7501156330108643
Itr: 0	Epoch: 6	Train Acc: 0.315	Test Acc: 0.285	AVG Train loss: 2.154211416244507	AVG Test loss: 1.9224892520904542
Itr: 0	Epoch: 7	Train Acc: 0.44125	Test Acc: 0.425	AVG Train loss: 1.6902070474624633	AVG Test loss: 1.5723002862930298
Itr: 0	Epoch: 8	Train Acc: 0.505	Test Acc: 0.335	AVG Train loss: 1.5380445957183837	AVG Test loss: 1.910672698020935
Itr: 0	Epoch: 9	Train Acc: 0.47375	Test Acc: 0.425	AVG Train loss: 1.588856062889099	AVG Test loss: 1.982207441329956
Itr: 0	Epoch: 10	Train Acc: 0.4875	Test Acc: 0.28	AVG Train loss: 1.5958976316452027	AVG Test loss: 2.1746995496749877
Itr: 0	Epoch: 11	Train Acc: 0.4275	Test Acc: 0.405	AVG Train loss: 1.9473694801330566	AVG Test loss: 1.6235827445983886
Itr: 0	Epoch: 12	Train Acc: 0.46875	Test Acc: 0.525	AVG Train loss: 1.4935134840011597	AVG Test loss: 1.306244740486145
Itr: 0	Epoch: 13	Train Acc: 0.61375	Test Acc: 0.515	AVG Train loss: 1.1968937635421752	AVG Test loss: 1.310274076461792
Itr: 0	Epoch: 14	Train Acc: 0.39875	Test Acc: 0.48	AVG Train loss: 1.7500818061828614	AVG Test loss: 1.4412384223937988
Itr: 0	Epoch: 15	Train Acc: 0.55625	Test Acc: 0.44	AVG Train loss: 1.28355046749115	AVG Test loss: 1.6145812797546386
Itr: 0	Epoch: 16	Train Acc: 0.555	Test Acc: 0.43	AVG Train loss: 1.3656717586517333	AVG Test loss: 1.7494947290420533
Itr: 0	Epoch: 17	Train Acc: 0.585	Test Acc: 0.505	AVG Train loss: 1.3928412294387817	AVG Test loss: 1.3215856552124023
Itr: 0	Epoch: 18	Train Acc: 0.6225	Test Acc: 0.495	AVG Train loss: 1.0669150257110596	AVG Test loss: 1.5190074062347412
Itr: 0	Epoch: 19	Train Acc: 0.64	Test Acc: 0.33	AVG Train loss: 1.12695228099823	AVG Test loss: 2.120408630371094
Itr: 0	Epoch: 20	Train Acc: 0.48125	Test Acc: 0.43	AVG Train loss: 1.6717622089385986	AVG Test loss: 1.8042024660110474
Itr: 0	Epoch: 21	Train Acc: 0.48875	Test Acc: 0.58	AVG Train loss: 1.5156023597717285	AVG Test loss: 1.2423443174362183
Itr: 0	Epoch: 22	Train Acc: 0.63375	Test Acc: 0.475	AVG Train loss: 1.1215084767341614	AVG Test loss: 1.566048526763916
Itr: 0	Epoch: 23	Train Acc: 0.59125	Test Acc: 0.525	AVG Train loss: 1.3134966039657592	AVG Test loss: 1.373265233039856
Itr: 0	Epoch: 24	Train Acc: 0.65625	Test Acc: 0.45	AVG Train loss: 1.0163935327529907	AVG Test loss: 1.6509660577774048
Itr: 0	Epoch: 25	Train Acc: 0.58125	Test Acc: 0.54	AVG Train loss: 1.3041653943061828	AVG Test loss: 1.6302053546905517
Itr: 0	Epoch: 26	Train Acc: 0.64125	Test Acc: 0.635	AVG Train loss: 1.1509110569953918	AVG Test loss: 1.1937480354309082
Itr: 0	Epoch: 27	Train Acc: 0.6125	Test Acc: 0.475	AVG Train loss: 1.1789812898635865	AVG Test loss: 1.6077577781677246
Itr: 0	Epoch: 28	Train Acc: 0.6575	Test Acc: 0.63	AVG Train loss: 1.1039612221717834	AVG Test loss: 1.080380277633667
Itr: 0	Epoch: 29	Train Acc: 0.64125	Test Acc: 0.605	AVG Train loss: 1.0896171092987061	AVG Test loss: 1.165594367980957
Itr: 0	Epoch: 30	Train Acc: 0.705	Test Acc: 0.565	AVG Train loss: 0.9518519079685211	AVG Test loss: 1.1946162176132202
Itr: 0	Epoch: 31	Train Acc: 0.725	Test Acc: 0.54	AVG Train loss: 0.8302743482589722	AVG Test loss: 1.4799380970001221
Itr: 0	Epoch: 32	Train Acc: 0.69125	Test Acc: 0.575	AVG Train loss: 0.9201108908653259	AVG Test loss: 1.21162784576416
Itr: 0	Epoch: 33	Train Acc: 0.7325	Test Acc: 0.53	AVG Train loss: 0.7513492131233215	AVG Test loss: 1.356860122680664
Itr: 0	Epoch: 34	Train Acc: 0.64625	Test Acc: 0.515	AVG Train loss: 1.0363667154312133	AVG Test loss: 1.4690361499786377
Itr: 0	Epoch: 35	Train Acc: 0.69375	Test Acc: 0.595	AVG Train loss: 0.9593785691261292	AVG Test loss: 1.1034138107299805
Itr: 0	Epoch: 36	Train Acc: 0.64375	Test Acc: 0.6	AVG Train loss: 1.0264394569396973	AVG Test loss: 1.1233156204223633
Itr: 0	Epoch: 37	Train Acc: 0.66125	Test Acc: 0.595	AVG Train loss: 0.9910774636268616	AVG Test loss: 1.1620934343338012
Itr: 0	Epoch: 38	Train Acc: 0.645	Test Acc: 0.575	AVG Train loss: 1.0733117389678954	AVG Test loss: 1.1979745149612426
Itr: 0	Epoch: 39	Train Acc: 0.735	Test Acc: 0.605	AVG Train loss: 0.825573867559433	AVG Test loss: 1.0808425855636596
Itr: 0	Epoch: 40	Train Acc: 0.65625	Test Acc: 0.65	AVG Train loss: 1.042335331439972	AVG Test loss: 0.9866633558273316
Itr: 0	Epoch: 41	Train Acc: 0.675	Test Acc: 0.495	AVG Train loss: 0.9672158145904541	AVG Test loss: 1.7685347747802735
Itr: 0	Epoch: 42	Train Acc: 0.5	Test Acc: 0.32	AVG Train loss: 1.5307247829437256	AVG Test loss: 1.9176730585098267
Itr: 0	Epoch: 43	Train Acc: 0.49375	Test Acc: 0.595	AVG Train loss: 1.7328531169891357	AVG Test loss: 1.1089054679870605
Itr: 0	Epoch: 44	Train Acc: 0.7075	Test Acc: 0.66	AVG Train loss: 0.9206678438186645	AVG Test loss: 1.022287278175354
Itr: 0	Epoch: 45	Train Acc: 0.83375	Test Acc: 0.555	AVG Train loss: 0.5849420738220215	AVG Test loss: 1.2583406352996827
Itr: 0	Epoch: 46	Train Acc: 0.8075	Test Acc: 0.63	AVG Train loss: 0.5948824226856232	AVG Test loss: 1.091397624015808
Itr: 0	Epoch: 47	Train Acc: 0.7775	Test Acc: 0.68	AVG Train loss: 0.6449421405792236	AVG Test loss: 1.0032394647598266
Itr: 0	Epoch: 48	Train Acc: 0.68625	Test Acc: 0.62	AVG Train loss: 0.9498400211334228	AVG Test loss: 1.0328624486923217
Itr: 0	Epoch: 49	Train Acc: 0.69625	Test Acc: 0.525	AVG Train loss: 0.9169602775573731	AVG Test loss: 1.4499079513549804
train set ACC of each class tensor([0.0962, 0.0413, 0.0800, 0.1000, 0.0950, 0.0988, 0.0262, 0.0825, 0.0075,
        0.0962])
[[77  0  0  3  0  0  0  0  0  0]
 [45 33  0  0  0  1  0  1  0  0]
 [ 2  0 64 14  0  0  0  0  0  0]
 [ 0  0  0 80  0  0  0  0  0  0]
 [ 0  0  0  4 76  0  0  0  0  0]
 [ 0  0  1  0  0 79  0  0  0  0]
 [34  0  0  1  0 24 21  0  0  0]
 [ 4  0  0  1  0  9  0 66  0  0]
 [23  0  0  6  0 42  0  3  6  0]
 [ 1  0  0  0  0  2  0  0  0 77]]
test set ACC of each class tensor([0.0900, 0.0250, 0.0500, 0.0850, 0.0800, 0.0850, 0.0100, 0.0300, 0.0000,
        0.0700])
[[18  0  0  0  0  1  0  1  0  0]
 [12  5  0  0  0  2  0  1  0  0]
 [ 1  0 10  8  1  0  0  0  0  0]
 [ 2  0  0 17  0  1  0  0  0  0]
 [ 0  0  0  4 16  0  0  0  0  0]
 [ 2  0  0  1  0 17  0  0  0  0]
 [11  0  0  0  0  7  2  0  0  0]
 [ 0  0  0  0  0 14  0  6  0  0]
 [ 4  1  0  3  0 12  0  0  0  0]
 [ 2  0  0  3  0  1  0  0  0 14]]
DataParallel(
  (module): ConvNet(
    (features): Sequential(
      (0): Conv2d(3, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): GroupNorm(128, 128, eps=1e-05, affine=True)
      (2): ReLU(inplace=True)
      (3): AvgPool2d(kernel_size=2, stride=2, padding=0)
      (4): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (5): GroupNorm(128, 128, eps=1e-05, affine=True)
      (6): ReLU(inplace=True)
      (7): AvgPool2d(kernel_size=2, stride=2, padding=0)
      (8): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (9): GroupNorm(128, 128, eps=1e-05, affine=True)
      (10): ReLU(inplace=True)
      (11): AvgPool2d(kernel_size=2, stride=2, padding=0)
    )
    (classifier): Linear(in_features=32768, out_features=10, bias=True)
  )
)
Itr: 1	Epoch: 0	Train Acc: 0.15375	Test Acc: 0.21	AVG Train loss: 2.6443825435638426	AVG Test loss: 3.5951985359191894
Itr: 1	Epoch: 1	Train Acc: 0.23875	Test Acc: 0.36	AVG Train loss: 3.1748000621795653	AVG Test loss: 2.2516933059692383
Itr: 1	Epoch: 2	Train Acc: 0.3275	Test Acc: 0.28	AVG Train loss: 2.1507812976837157	AVG Test loss: 2.9505823993682863
Itr: 1	Epoch: 3	Train Acc: 0.28125	Test Acc: 0.215	AVG Train loss: 2.8081757736206057	AVG Test loss: 2.359050998687744
Itr: 1	Epoch: 4	Train Acc: 0.37875	Test Acc: 0.37	AVG Train loss: 1.928098702430725	AVG Test loss: 2.3810608768463135
Itr: 1	Epoch: 5	Train Acc: 0.44625	Test Acc: 0.355	AVG Train loss: 1.840076732635498	AVG Test loss: 1.9460735702514649
Itr: 1	Epoch: 6	Train Acc: 0.41625	Test Acc: 0.385	AVG Train loss: 1.8563692474365234	AVG Test loss: 1.8166241550445557
Itr: 1	Epoch: 7	Train Acc: 0.4925	Test Acc: 0.4	AVG Train loss: 1.5423945474624634	AVG Test loss: 2.074468812942505
Itr: 1	Epoch: 8	Train Acc: 0.37625	Test Acc: 0.455	AVG Train loss: 2.0901227951049806	AVG Test loss: 2.1696730184555055
Itr: 1	Epoch: 9	Train Acc: 0.5475	Test Acc: 0.395	AVG Train loss: 1.4980111026763916	AVG Test loss: 1.777579288482666
Itr: 1	Epoch: 10	Train Acc: 0.47375	Test Acc: 0.43	AVG Train loss: 1.725526337623596	AVG Test loss: 1.5423500919342041
Itr: 1	Epoch: 11	Train Acc: 0.575	Test Acc: 0.39	AVG Train loss: 1.2529274034500122	AVG Test loss: 1.87076904296875
Itr: 1	Epoch: 12	Train Acc: 0.4425	Test Acc: 0.465	AVG Train loss: 1.6371660327911377	AVG Test loss: 1.679443483352661
Itr: 1	Epoch: 13	Train Acc: 0.45375	Test Acc: 0.365	AVG Train loss: 1.6462672853469849	AVG Test loss: 1.9217299795150757
Itr: 1	Epoch: 14	Train Acc: 0.59875	Test Acc: 0.5	AVG Train loss: 1.261648211479187	AVG Test loss: 1.6621838665008546
Itr: 1	Epoch: 15	Train Acc: 0.63375	Test Acc: 0.565	AVG Train loss: 1.1316304683685303	AVG Test loss: 1.2124559307098388
Itr: 1	Epoch: 16	Train Acc: 0.475	Test Acc: 0.46	AVG Train loss: 1.478294506072998	AVG Test loss: 1.551826949119568
Itr: 1	Epoch: 17	Train Acc: 0.595	Test Acc: 0.435	AVG Train loss: 1.1892054080963135	AVG Test loss: 1.715443115234375
Itr: 1	Epoch: 18	Train Acc: 0.5925	Test Acc: 0.355	AVG Train loss: 1.2129432916641236	AVG Test loss: 1.6427322292327882
Itr: 1	Epoch: 19	Train Acc: 0.3825	Test Acc: 0.495	AVG Train loss: 1.939203929901123	AVG Test loss: 1.576990566253662
Itr: 1	Epoch: 20	Train Acc: 0.59	Test Acc: 0.57	AVG Train loss: 1.2392926788330079	AVG Test loss: 1.1689687538146973
Itr: 1	Epoch: 21	Train Acc: 0.67625	Test Acc: 0.43	AVG Train loss: 0.9221904373168945	AVG Test loss: 1.5277478885650635
Itr: 1	Epoch: 22	Train Acc: 0.5925	Test Acc: 0.45	AVG Train loss: 1.1792619490623475	AVG Test loss: 1.6723929643630981
Itr: 1	Epoch: 23	Train Acc: 0.57	Test Acc: 0.545	AVG Train loss: 1.3192270183563233	AVG Test loss: 1.3804325771331787
Itr: 1	Epoch: 24	Train Acc: 0.5925	Test Acc: 0.565	AVG Train loss: 1.2514840698242187	AVG Test loss: 1.1978887701034546
Itr: 1	Epoch: 25	Train Acc: 0.75	Test Acc: 0.43	AVG Train loss: 0.7771492528915406	AVG Test loss: 1.7532599449157715
Itr: 1	Epoch: 26	Train Acc: 0.59125	Test Acc: 0.44	AVG Train loss: 1.3146814441680907	AVG Test loss: 1.5100505113601685
Itr: 1	Epoch: 27	Train Acc: 0.66125	Test Acc: 0.555	AVG Train loss: 0.9409773659706115	AVG Test loss: 1.1653495168685912
Itr: 1	Epoch: 28	Train Acc: 0.76125	Test Acc: 0.505	AVG Train loss: 0.7533343386650085	AVG Test loss: 1.4129485988616943
Itr: 1	Epoch: 29	Train Acc: 0.6675	Test Acc: 0.56	AVG Train loss: 1.0910737466812135	AVG Test loss: 1.3869105577468872
Itr: 1	Epoch: 30	Train Acc: 0.525	Test Acc: 0.405	AVG Train loss: 1.3959773921966552	AVG Test loss: 1.8882520961761475
Itr: 1	Epoch: 31	Train Acc: 0.585	Test Acc: 0.535	AVG Train loss: 1.3167178630828857	AVG Test loss: 1.3292657613754273
Itr: 1	Epoch: 32	Train Acc: 0.64625	Test Acc: 0.55	AVG Train loss: 1.079548544883728	AVG Test loss: 1.2979442405700683
Itr: 1	Epoch: 33	Train Acc: 0.57375	Test Acc: 0.585	AVG Train loss: 1.2567565608024598	AVG Test loss: 1.1473075771331787
Itr: 1	Epoch: 34	Train Acc: 0.6375	Test Acc: 0.555	AVG Train loss: 1.081143114566803	AVG Test loss: 1.2582827997207642
Itr: 1	Epoch: 35	Train Acc: 0.7575	Test Acc: 0.575	AVG Train loss: 0.7207061529159546	AVG Test loss: 1.1763601875305176
Itr: 1	Epoch: 36	Train Acc: 0.685	Test Acc: 0.54	AVG Train loss: 0.9772996640205384	AVG Test loss: 1.4230539321899414
Itr: 1	Epoch: 37	Train Acc: 0.74875	Test Acc: 0.665	AVG Train loss: 0.7588109815120697	AVG Test loss: 0.9432726049423218
Itr: 1	Epoch: 38	Train Acc: 0.76625	Test Acc: 0.595	AVG Train loss: 0.7692399477958679	AVG Test loss: 1.156532220840454
Itr: 1	Epoch: 39	Train Acc: 0.72625	Test Acc: 0.67	AVG Train loss: 0.770328254699707	AVG Test loss: 1.061553382873535
Itr: 1	Epoch: 40	Train Acc: 0.8225	Test Acc: 0.595	AVG Train loss: 0.576153838634491	AVG Test loss: 1.1922815322875977
Itr: 1	Epoch: 41	Train Acc: 0.80625	Test Acc: 0.585	AVG Train loss: 0.5978289985656738	AVG Test loss: 1.3519728899002075
Itr: 1	Epoch: 42	Train Acc: 0.7325	Test Acc: 0.64	AVG Train loss: 0.8149307870864868	AVG Test loss: 1.0229345560073853
Itr: 1	Epoch: 43	Train Acc: 0.82375	Test Acc: 0.65	AVG Train loss: 0.5830126535892487	AVG Test loss: 1.0213905191421508
Itr: 1	Epoch: 44	Train Acc: 0.81	Test Acc: 0.42	AVG Train loss: 0.598373851776123	AVG Test loss: 2.171879575252533
Itr: 1	Epoch: 45	Train Acc: 0.59	Test Acc: 0.515	AVG Train loss: 1.3050516974925994	AVG Test loss: 1.481496477127075
Itr: 1	Epoch: 46	Train Acc: 0.7	Test Acc: 0.64	AVG Train loss: 0.9161373686790466	AVG Test loss: 1.0169739198684693
Itr: 1	Epoch: 47	Train Acc: 0.86	Test Acc: 0.59	AVG Train loss: 0.48010542392730715	AVG Test loss: 1.2209020042419434
Itr: 1	Epoch: 48	Train Acc: 0.8175	Test Acc: 0.675	AVG Train loss: 0.5727487409114838	AVG Test loss: 0.9127372169494629
Itr: 1	Epoch: 49	Train Acc: 0.85125	Test Acc: 0.575	AVG Train loss: 0.5343958020210267	AVG Test loss: 1.291590223312378
train set ACC of each class tensor([0.0712, 0.0988, 0.0887, 0.1000, 0.0812, 0.0463, 0.0887, 0.0988, 0.0125,
        0.0962])
[[57 19  0  2  0  0  0  2  0  0]
 [ 0 79  0  0  0  0  0  1  0  0]
 [ 0  1 71  8  0  0  0  0  0  0]
 [ 0  0  0 80  0  0  0  0  0  0]
 [ 0  0  0 15 65  0  0  0  0  0]
 [ 0  5  1  1  0 37  7 29  0  0]
 [ 0  4  0  0  0  0 71  5  0  0]
 [ 0  1  0  0  0  0  0 79  0  0]
 [ 3 20  0  9  0  0 12 25 10  1]
 [ 0  2  0  0  0  0  0  1  0 77]]
test set ACC of each class tensor([0.0450, 0.1000, 0.0600, 0.0800, 0.0750, 0.0050, 0.0450, 0.0850, 0.0000,
        0.0800])
[[ 9 10  0  0  0  0  0  1  0  0]
 [ 0 20  0  0  0  0  0  0  0  0]
 [ 0  4 12  4  0  0  0  0  0  0]
 [ 0  2  1 16  0  0  0  1  0  0]
 [ 0  0  0  5 15  0  0  0  0  0]
 [ 0  1  0  1  0  1  3 14  0  0]
 [ 2  3  0  1  0  3  9  2  0  0]
 [ 0  1  0  0  0  0  2 17  0  0]
 [ 2  4  0  4  0  1  2  7  0  0]
 [ 0  3  0  1  0  0  0  0  0 16]]
DataParallel(
  (module): ConvNet(
    (features): Sequential(
      (0): Conv2d(3, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): GroupNorm(128, 128, eps=1e-05, affine=True)
      (2): ReLU(inplace=True)
      (3): AvgPool2d(kernel_size=2, stride=2, padding=0)
      (4): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (5): GroupNorm(128, 128, eps=1e-05, affine=True)
      (6): ReLU(inplace=True)
      (7): AvgPool2d(kernel_size=2, stride=2, padding=0)
      (8): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (9): GroupNorm(128, 128, eps=1e-05, affine=True)
      (10): ReLU(inplace=True)
      (11): AvgPool2d(kernel_size=2, stride=2, padding=0)
    )
    (classifier): Linear(in_features=32768, out_features=10, bias=True)
  )
)
Itr: 2	Epoch: 0	Train Acc: 0.1625	Test Acc: 0.145	AVG Train loss: 2.8079711246490477	AVG Test loss: 4.9547334575653075
Itr: 2	Epoch: 1	Train Acc: 0.17875	Test Acc: 0.195	AVG Train loss: 4.453454313278198	AVG Test loss: 3.2022321319580076
Itr: 2	Epoch: 2	Train Acc: 0.2425	Test Acc: 0.27	AVG Train loss: 2.584178915023804	AVG Test loss: 2.8276857280731202
Itr: 2	Epoch: 3	Train Acc: 0.34875	Test Acc: 0.34	AVG Train loss: 2.3296840047836302	AVG Test loss: 2.037114853858948
Itr: 2	Epoch: 4	Train Acc: 0.39	Test Acc: 0.36	AVG Train loss: 1.8546545886993409	AVG Test loss: 1.7747511768341064
Itr: 2	Epoch: 5	Train Acc: 0.45125	Test Acc: 0.39	AVG Train loss: 1.6440216827392578	AVG Test loss: 2.097103977203369
Itr: 2	Epoch: 6	Train Acc: 0.495	Test Acc: 0.39	AVG Train loss: 1.560393533706665	AVG Test loss: 1.8684482192993164
Itr: 2	Epoch: 7	Train Acc: 0.45125	Test Acc: 0.45	AVG Train loss: 1.8055812549591064	AVG Test loss: 1.540093035697937
Itr: 2	Epoch: 8	Train Acc: 0.39	Test Acc: 0.4	AVG Train loss: 1.8068330240249635	AVG Test loss: 1.693424859046936
Itr: 2	Epoch: 9	Train Acc: 0.555	Test Acc: 0.265	AVG Train loss: 1.376451964378357	AVG Test loss: 2.5744968128204344
Itr: 2	Epoch: 10	Train Acc: 0.3925	Test Acc: 0.425	AVG Train loss: 2.3578683042526247	AVG Test loss: 1.5166057777404784
Itr: 2	Epoch: 11	Train Acc: 0.50625	Test Acc: 0.545	AVG Train loss: 1.42516170501709	AVG Test loss: 1.3940480518341065
Itr: 2	Epoch: 12	Train Acc: 0.51375	Test Acc: 0.415	AVG Train loss: 1.411408042907715	AVG Test loss: 1.653479928970337
Itr: 2	Epoch: 13	Train Acc: 0.53375	Test Acc: 0.555	AVG Train loss: 1.3838224935531616	AVG Test loss: 1.2890927934646605
Itr: 2	Epoch: 14	Train Acc: 0.64875	Test Acc: 0.425	AVG Train loss: 1.036242837905884	AVG Test loss: 1.8312272644042968
Itr: 2	Epoch: 15	Train Acc: 0.5225	Test Acc: 0.43	AVG Train loss: 1.4633785343170167	AVG Test loss: 1.548874101638794
Itr: 2	Epoch: 16	Train Acc: 0.58125	Test Acc: 0.485	AVG Train loss: 1.3774409341812133	AVG Test loss: 1.4526015663146972
Itr: 2	Epoch: 17	Train Acc: 0.5475	Test Acc: 0.545	AVG Train loss: 1.3871693181991578	AVG Test loss: 1.2012509489059449
Itr: 2	Epoch: 18	Train Acc: 0.68	Test Acc: 0.585	AVG Train loss: 0.989336006641388	AVG Test loss: 1.167855019569397
Itr: 2	Epoch: 19	Train Acc: 0.69125	Test Acc: 0.575	AVG Train loss: 0.9365837383270263	AVG Test loss: 1.3394142150878907
Itr: 2	Epoch: 20	Train Acc: 0.725	Test Acc: 0.605	AVG Train loss: 0.8547395968437195	AVG Test loss: 1.234579610824585
Itr: 2	Epoch: 21	Train Acc: 0.71625	Test Acc: 0.51	AVG Train loss: 0.8742895603179932	AVG Test loss: 1.5254779267311096
Itr: 2	Epoch: 22	Train Acc: 0.5675	Test Acc: 0.655	AVG Train loss: 1.3385869455337525	AVG Test loss: 1.1197257709503174
Itr: 2	Epoch: 23	Train Acc: 0.715	Test Acc: 0.59	AVG Train loss: 0.8837987399101257	AVG Test loss: 1.1979573965072632
Itr: 2	Epoch: 24	Train Acc: 0.72	Test Acc: 0.555	AVG Train loss: 0.8263319444656372	AVG Test loss: 1.5320885944366456
Itr: 2	Epoch: 25	Train Acc: 0.685	Test Acc: 0.57	AVG Train loss: 0.9844498920440674	AVG Test loss: 1.2010881042480468
Itr: 2	Epoch: 26	Train Acc: 0.5825	Test Acc: 0.505	AVG Train loss: 1.214962830543518	AVG Test loss: 1.3976306104660035
Itr: 2	Epoch: 27	Train Acc: 0.58875	Test Acc: 0.58	AVG Train loss: 1.159423508644104	AVG Test loss: 1.1177305269241333
Itr: 2	Epoch: 28	Train Acc: 0.7425	Test Acc: 0.635	AVG Train loss: 0.813507114648819	AVG Test loss: 1.0757607698440552
Itr: 2	Epoch: 29	Train Acc: 0.77625	Test Acc: 0.555	AVG Train loss: 0.7125509452819824	AVG Test loss: 1.3662982749938966
Itr: 2	Epoch: 30	Train Acc: 0.66	Test Acc: 0.595	AVG Train loss: 1.0813239479064942	AVG Test loss: 1.1937630081176758
Itr: 2	Epoch: 31	Train Acc: 0.725	Test Acc: 0.475	AVG Train loss: 0.8292311286926269	AVG Test loss: 1.493514094352722
Itr: 2	Epoch: 32	Train Acc: 0.54125	Test Acc: 0.575	AVG Train loss: 1.6900660037994384	AVG Test loss: 1.1692594861984253
Itr: 2	Epoch: 33	Train Acc: 0.57375	Test Acc: 0.635	AVG Train loss: 1.2013059401512145	AVG Test loss: 1.112197256088257
Itr: 2	Epoch: 34	Train Acc: 0.76875	Test Acc: 0.59	AVG Train loss: 0.6844746088981628	AVG Test loss: 1.1353414630889893
```

# lr 0.01
```cmd
ssh://sbcaesar@dais10.uwb.edu:22/data/sbcaesar/xuan_venv/bin/python3 -u /data/sbcaesar/mac_galaxy/buffer.py
Hyper-parameters: 
 {'dataset': 'gzoo2', 'subset': 'imagenette', 'model': 'ConvNet', 'num_experts': 10, 'lr_teacher': 0.01, 'batch_train': 256, 'batch_real': 256, 'dsa': True, 'dsa_strategy': 'color_crop_cutout_flip_scale_rotate', 'data_path': 'data', 'buffer_path': '/data/sbcaesar/galaxy_buffers', 'train_epochs': 50, 'zca': False, 'decay': False, 'mom': 0, 'l2': 0, 'save_interval': 10, 'device': 'cuda', 'dsa_param': <utils.ParamDiffAug object at 0x7f7b1c25dd00>}
BUILDING DATASET
100%|█████████████████████████████████████| 800/800 [00:00<00:00, 159464.08it/s]
800it [00:00, 3892625.52it/s]
class c = 0: 80 real images
class c = 1: 80 real images
class c = 2: 80 real images
class c = 3: 80 real images
class c = 4: 80 real images
class c = 5: 80 real images
class c = 6: 80 real images
class c = 7: 80 real images
class c = 8: 80 real images
class c = 9: 80 real images
real images channel 0, mean = 0.0003, std = 1.0003
real images channel 1, mean = -0.0002, std = 1.0004
real images channel 2, mean = 0.0006, std = 1.0005
Add weight to loss function tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
DC augmentation parameters: 
 {'crop': 4, 'scale': 0.2, 'rotate': 45, 'noise': 0.001, 'strategy': 'crop_scale_rotate'}
DataParallel(
  (module): ConvNet(
    (features): Sequential(
      (0): Conv2d(3, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): GroupNorm(128, 128, eps=1e-05, affine=True)
      (2): ReLU(inplace=True)
      (3): AvgPool2d(kernel_size=2, stride=2, padding=0)
      (4): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (5): GroupNorm(128, 128, eps=1e-05, affine=True)
      (6): ReLU(inplace=True)
      (7): AvgPool2d(kernel_size=2, stride=2, padding=0)
      (8): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (9): GroupNorm(128, 128, eps=1e-05, affine=True)
      (10): ReLU(inplace=True)
      (11): AvgPool2d(kernel_size=2, stride=2, padding=0)
    )
    (classifier): Linear(in_features=32768, out_features=10, bias=True)
  )
)
Itr: 0	Epoch: 0	Train Acc: 0.13875	Test Acc: 0.17	AVG Train loss: 4.76798225402832	AVG Test loss: 6.167542362213135
Itr: 0	Epoch: 1	Train Acc: 0.15625	Test Acc: 0.185	AVG Train loss: 5.115000200271607	AVG Test loss: 3.7235618305206297
Itr: 0	Epoch: 2	Train Acc: 0.22125	Test Acc: 0.215	AVG Train loss: 4.334802436828613	AVG Test loss: 4.871517143249512
Itr: 0	Epoch: 3	Train Acc: 0.24	Test Acc: 0.145	AVG Train loss: 4.646726913452149	AVG Test loss: 4.9560152053833
Itr: 0	Epoch: 4	Train Acc: 0.14125	Test Acc: 0.23	AVG Train loss: 4.46795612335205	AVG Test loss: 4.263150444030762
Itr: 0	Epoch: 5	Train Acc: 0.1825	Test Acc: 0.185	AVG Train loss: 4.3876433753967286	AVG Test loss: 5.365886631011963
Itr: 0	Epoch: 6	Train Acc: 0.20125	Test Acc: 0.34	AVG Train loss: 4.886918668746948	AVG Test loss: 2.7199996852874757
Itr: 0	Epoch: 7	Train Acc: 0.31875	Test Acc: 0.325	AVG Train loss: 2.379537010192871	AVG Test loss: 2.3034866714477538
Itr: 0	Epoch: 8	Train Acc: 0.36875	Test Acc: 0.145	AVG Train loss: 2.3478893184661866	AVG Test loss: 3.870893611907959
Itr: 0	Epoch: 9	Train Acc: 0.265	Test Acc: 0.275	AVG Train loss: 3.3951494884490967	AVG Test loss: 2.877648801803589
Itr: 0	Epoch: 10	Train Acc: 0.39375	Test Acc: 0.34	AVG Train loss: 2.3808585739135744	AVG Test loss: 2.024246459007263
Itr: 0	Epoch: 11	Train Acc: 0.285	Test Acc: 0.235	AVG Train loss: 2.2526587295532225	AVG Test loss: 3.2929081535339355
Itr: 0	Epoch: 12	Train Acc: 0.38875	Test Acc: 0.325	AVG Train loss: 2.7076805448532104	AVG Test loss: 3.080846061706543
Itr: 0	Epoch: 13	Train Acc: 0.36	Test Acc: 0.405	AVG Train loss: 2.6653996515274048	AVG Test loss: 2.2561361026763915
Itr: 0	Epoch: 14	Train Acc: 0.4725	Test Acc: 0.38	AVG Train loss: 1.875900011062622	AVG Test loss: 2.1691154718399046
Itr: 0	Epoch: 15	Train Acc: 0.41875	Test Acc: 0.395	AVG Train loss: 1.9336903524398803	AVG Test loss: 1.7658249187469481
Itr: 0	Epoch: 16	Train Acc: 0.44125	Test Acc: 0.425	AVG Train loss: 1.6426495599746704	AVG Test loss: 1.6908219623565675
Itr: 0	Epoch: 17	Train Acc: 0.38625	Test Acc: 0.45	AVG Train loss: 1.9066410875320434	AVG Test loss: 1.9919893550872803
Itr: 0	Epoch: 18	Train Acc: 0.49375	Test Acc: 0.365	AVG Train loss: 1.6759433507919312	AVG Test loss: 1.8591886520385743
Itr: 0	Epoch: 19	Train Acc: 0.42875	Test Acc: 0.51	AVG Train loss: 1.7528257036209107	AVG Test loss: 1.5903759050369262
Itr: 0	Epoch: 20	Train Acc: 0.49	Test Acc: 0.49	AVG Train loss: 1.5651722574234008	AVG Test loss: 1.4201531887054444
Itr: 0	Epoch: 21	Train Acc: 0.54625	Test Acc: 0.425	AVG Train loss: 1.2734868621826172	AVG Test loss: 1.5918267822265626
Itr: 0	Epoch: 22	Train Acc: 0.44875	Test Acc: 0.49	AVG Train loss: 1.6976061868667602	AVG Test loss: 1.5195828247070313
Itr: 0	Epoch: 23	Train Acc: 0.55625	Test Acc: 0.38	AVG Train loss: 1.2961044359207152	AVG Test loss: 1.878686285018921
Itr: 0	Epoch: 24	Train Acc: 0.54	Test Acc: 0.42	AVG Train loss: 1.4674982261657714	AVG Test loss: 1.723296227455139
Itr: 0	Epoch: 25	Train Acc: 0.56	Test Acc: 0.48	AVG Train loss: 1.3559184074401855	AVG Test loss: 1.8650385665893554
Itr: 0	Epoch: 26	Train Acc: 0.58625	Test Acc: 0.49	AVG Train loss: 1.343911085128784	AVG Test loss: 1.5950182342529298
Itr: 0	Epoch: 27	Train Acc: 0.58375	Test Acc: 0.43	AVG Train loss: 1.2842142868041992	AVG Test loss: 1.6952771377563476
Itr: 0	Epoch: 28	Train Acc: 0.51	Test Acc: 0.37	AVG Train loss: 1.5474263072013854	AVG Test loss: 2.072851548194885
Itr: 0	Epoch: 29	Train Acc: 0.46875	Test Acc: 0.54	AVG Train loss: 1.7857185292243958	AVG Test loss: 1.3229130935668945
Itr: 0	Epoch: 30	Train Acc: 0.655	Test Acc: 0.445	AVG Train loss: 1.0010245800018311	AVG Test loss: 2.2587808418273925
Itr: 0	Epoch: 31	Train Acc: 0.44875	Test Acc: 0.625	AVG Train loss: 1.8855806112289428	AVG Test loss: 1.0799832725524903
Itr: 0	Epoch: 32	Train Acc: 0.6775	Test Acc: 0.565	AVG Train loss: 1.0255211973190308	AVG Test loss: 1.311220073699951
Itr: 0	Epoch: 33	Train Acc: 0.68875	Test Acc: 0.35	AVG Train loss: 0.9616659116744996	AVG Test loss: 2.3248700714111328
Itr: 0	Epoch: 34	Train Acc: 0.645	Test Acc: 0.445	AVG Train loss: 1.3042750883102416	AVG Test loss: 1.5347071838378907
Itr: 0	Epoch: 35	Train Acc: 0.50375	Test Acc: 0.435	AVG Train loss: 1.5369546556472777	AVG Test loss: 2.1158355331420897
Itr: 0	Epoch: 36	Train Acc: 0.63125	Test Acc: 0.61	AVG Train loss: 1.3149855351448059	AVG Test loss: 1.089592423439026
Itr: 0	Epoch: 37	Train Acc: 0.66875	Test Acc: 0.485	AVG Train loss: 0.9379654002189636	AVG Test loss: 2.09335205078125
Itr: 0	Epoch: 38	Train Acc: 0.61	Test Acc: 0.52	AVG Train loss: 1.4194357204437256	AVG Test loss: 1.3290867185592652
Itr: 0	Epoch: 39	Train Acc: 0.71625	Test Acc: 0.51	AVG Train loss: 0.8495367574691772	AVG Test loss: 1.2501190876960755
Itr: 0	Epoch: 40	Train Acc: 0.62	Test Acc: 0.485	AVG Train loss: 1.1011743545532227	AVG Test loss: 1.4033613729476928
Itr: 0	Epoch: 41	Train Acc: 0.66375	Test Acc: 0.625	AVG Train loss: 1.0898595142364502	AVG Test loss: 1.0822785234451293
Itr: 0	Epoch: 42	Train Acc: 0.71375	Test Acc: 0.55	AVG Train loss: 0.7638832187652588	AVG Test loss: 1.4400528001785278
Itr: 0	Epoch: 43	Train Acc: 0.675	Test Acc: 0.53	AVG Train loss: 0.8546718835830689	AVG Test loss: 1.336280813217163
Itr: 0	Epoch: 44	Train Acc: 0.71625	Test Acc: 0.6	AVG Train loss: 0.8320592427253723	AVG Test loss: 1.2964493751525878
Itr: 0	Epoch: 45	Train Acc: 0.555	Test Acc: 0.495	AVG Train loss: 1.3015674138069153	AVG Test loss: 1.6823803138732911
Itr: 0	Epoch: 46	Train Acc: 0.75625	Test Acc: 0.615	AVG Train loss: 0.761364893913269	AVG Test loss: 1.2539031219482422
Itr: 0	Epoch: 47	Train Acc: 0.74125	Test Acc: 0.595	AVG Train loss: 0.7552393782138824	AVG Test loss: 1.2113306951522826
Itr: 0	Epoch: 48	Train Acc: 0.52125	Test Acc: 0.545	AVG Train loss: 1.6304462027549744	AVG Test loss: 1.271134958267212
Itr: 0	Epoch: 49	Train Acc: 0.70625	Test Acc: 0.46	AVG Train loss: 0.9071127319335938	AVG Test loss: 1.515779094696045
train set ACC of each class tensor([0.0525, 0.0125, 0.0913, 0.0950, 0.0975, 0.0675, 0.1000, 0.0113, 0.0538,
        0.0925])
[[42  0  0  2  0  3 32  0  1  0]
 [ 7 10  0  0  0  6 56  0  1  0]
 [ 0  0 73  5  0  0  1  0  1  0]
 [ 0  0  0 76  0  0  4  0  0  0]
 [ 0  0  0  2 78  0  0  0  0  0]
 [ 0  0  0  0  0 54 25  0  1  0]
 [ 0  0  0  0  0  0 80  0  0  0]
 [ 0  0  0  0  0 40 27  9  4  0]
 [ 0  0  0  1  0  2 34  0 43  0]
 [ 0  0  0  2  0  1  3  0  0 74]]
test set ACC of each class tensor([0.0300, 0.0050, 0.0600, 0.0750, 0.0800, 0.0300, 0.0950, 0.0000, 0.0150,
        0.0700])
[[ 6  0  0  0  0  1 12  0  1  0]
 [ 1  1  0  0  0  4 13  0  1  0]
 [ 0  0 12  6  2  0  0  0  0  0]
 [ 0  0  0 15  0  0  5  0  0  0]
 [ 0  0  0  4 16  0  0  0  0  0]
 [ 0  0  0  0  0  6 14  0  0  0]
 [ 0  0  0  0  0  1 19  0  0  0]
 [ 0  0  0  0  0 15  5  0  0  0]
 [ 0  0  0  0  0  1 16  0  3  0]
 [ 0  0  0  2  1  1  2  0  0 14]]
DataParallel(
  (module): ConvNet(
    (features): Sequential(
      (0): Conv2d(3, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): GroupNorm(128, 128, eps=1e-05, affine=True)
      (2): ReLU(inplace=True)
      (3): AvgPool2d(kernel_size=2, stride=2, padding=0)
      (4): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (5): GroupNorm(128, 128, eps=1e-05, affine=True)
      (6): ReLU(inplace=True)
      (7): AvgPool2d(kernel_size=2, stride=2, padding=0)
      (8): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (9): GroupNorm(128, 128, eps=1e-05, affine=True)
      (10): ReLU(inplace=True)
      (11): AvgPool2d(kernel_size=2, stride=2, padding=0)
    )
    (classifier): Linear(in_features=32768, out_features=10, bias=True)
  )
)
Itr: 1	Epoch: 0	Train Acc: 0.11125	Test Acc: 0.1	AVG Train loss: 5.71519027709961	AVG Test loss: 9.156143875122071
Itr: 1	Epoch: 1	Train Acc: 0.135	Test Acc: 0.145	AVG Train loss: 7.493572616577149	AVG Test loss: 6.805239868164063
Itr: 1	Epoch: 2	Train Acc: 0.19125	Test Acc: 0.235	AVG Train loss: 5.508274803161621	AVG Test loss: 5.724811220169068
Itr: 1	Epoch: 3	Train Acc: 0.22375	Test Acc: 0.25	AVG Train loss: 5.036541328430176	AVG Test loss: 2.9442679643630982
Itr: 1	Epoch: 4	Train Acc: 0.26625	Test Acc: 0.235	AVG Train loss: 2.8813237380981445	AVG Test loss: 3.8067571449279787
Itr: 1	Epoch: 5	Train Acc: 0.25625	Test Acc: 0.22	AVG Train loss: 3.3582046604156495	AVG Test loss: 3.0490235710144042
Itr: 1	Epoch: 6	Train Acc: 0.2525	Test Acc: 0.37	AVG Train loss: 3.0526961421966554	AVG Test loss: 1.9860426378250122
Itr: 1	Epoch: 7	Train Acc: 0.35	Test Acc: 0.31	AVG Train loss: 2.0151158809661864	AVG Test loss: 3.2772672271728513
Itr: 1	Epoch: 8	Train Acc: 0.33	Test Acc: 0.245	AVG Train loss: 2.7645750522613524	AVG Test loss: 2.6444906902313234
Itr: 1	Epoch: 9	Train Acc: 0.3125	Test Acc: 0.365	AVG Train loss: 2.427212986946106	AVG Test loss: 2.8243416118621827
Itr: 1	Epoch: 10	Train Acc: 0.35625	Test Acc: 0.265	AVG Train loss: 2.758901147842407	AVG Test loss: 2.930177345275879
Itr: 1	Epoch: 11	Train Acc: 0.40125	Test Acc: 0.31	AVG Train loss: 2.237958002090454	AVG Test loss: 2.427844219207764
Itr: 1	Epoch: 12	Train Acc: 0.41625	Test Acc: 0.255	AVG Train loss: 2.144882936477661	AVG Test loss: 2.43586585521698
Itr: 1	Epoch: 13	Train Acc: 0.35125	Test Acc: 0.33	AVG Train loss: 2.2509015464782713	AVG Test loss: 2.07729887008667
Itr: 1	Epoch: 14	Train Acc: 0.39	Test Acc: 0.34	AVG Train loss: 1.9226036834716798	AVG Test loss: 2.9633467197418213
Itr: 1	Epoch: 15	Train Acc: 0.41125	Test Acc: 0.395	AVG Train loss: 2.444452333450317	AVG Test loss: 1.9842488479614258
Itr: 1	Epoch: 16	Train Acc: 0.44875	Test Acc: 0.405	AVG Train loss: 1.7645822429656983	AVG Test loss: 1.8067691898345948
Itr: 1	Epoch: 17	Train Acc: 0.525	Test Acc: 0.165	AVG Train loss: 1.663073215484619	AVG Test loss: 3.1220612478256227
Itr: 1	Epoch: 18	Train Acc: 0.42125	Test Acc: 0.365	AVG Train loss: 2.4754672956466677	AVG Test loss: 2.4343114852905274
Itr: 1	Epoch: 19	Train Acc: 0.50875	Test Acc: 0.275	AVG Train loss: 1.7548866748809815	AVG Test loss: 2.562625093460083
Itr: 1	Epoch: 20	Train Acc: 0.45375	Test Acc: 0.44	AVG Train loss: 2.0188953113555907	AVG Test loss: 1.9046816110610962
Itr: 1	Epoch: 21	Train Acc: 0.55625	Test Acc: 0.44	AVG Train loss: 1.4726973485946655	AVG Test loss: 1.717947850227356
Itr: 1	Epoch: 22	Train Acc: 0.415	Test Acc: 0.46	AVG Train loss: 1.6029527807235717	AVG Test loss: 1.6363229465484619
Itr: 1	Epoch: 23	Train Acc: 0.47625	Test Acc: 0.52	AVG Train loss: 1.5905616521835326	AVG Test loss: 1.320752773284912
Itr: 1	Epoch: 24	Train Acc: 0.5075	Test Acc: 0.455	AVG Train loss: 1.3933054208755493	AVG Test loss: 1.8357195854187012
Itr: 1	Epoch: 25	Train Acc: 0.56	Test Acc: 0.415	AVG Train loss: 1.4225800800323487	AVG Test loss: 1.723500452041626
Itr: 1	Epoch: 26	Train Acc: 0.555	Test Acc: 0.495	AVG Train loss: 1.3025780534744262	AVG Test loss: 1.8676351833343505
Itr: 1	Epoch: 27	Train Acc: 0.46375	Test Acc: 0.52	AVG Train loss: 1.58253972530365	AVG Test loss: 1.3871256971359254
Itr: 1	Epoch: 28	Train Acc: 0.555	Test Acc: 0.35	AVG Train loss: 1.2805127811431884	AVG Test loss: 2.454548645019531
Itr: 1	Epoch: 29	Train Acc: 0.51625	Test Acc: 0.51	AVG Train loss: 1.804496111869812	AVG Test loss: 1.4530170345306397
Itr: 1	Epoch: 30	Train Acc: 0.6275	Test Acc: 0.465	AVG Train loss: 1.05155198097229	AVG Test loss: 1.5911358785629273
Itr: 1	Epoch: 31	Train Acc: 0.63375	Test Acc: 0.52	AVG Train loss: 1.2052961897850036	AVG Test loss: 1.324372501373291
Itr: 1	Epoch: 32	Train Acc: 0.6675	Test Acc: 0.605	AVG Train loss: 0.9864512729644775	AVG Test loss: 1.07348726272583
Itr: 1	Epoch: 33	Train Acc: 0.625	Test Acc: 0.495	AVG Train loss: 1.097245545387268	AVG Test loss: 1.7315289402008056
Itr: 1	Epoch: 34	Train Acc: 0.6725	Test Acc: 0.57	AVG Train loss: 1.0284365844726562	AVG Test loss: 1.0975663948059082
Itr: 1	Epoch: 35	Train Acc: 0.695	Test Acc: 0.53	AVG Train loss: 0.9248202562332153	AVG Test loss: 1.3878151321411132
Itr: 1	Epoch: 36	Train Acc: 0.6925	Test Acc: 0.575	AVG Train loss: 0.912808804512024	AVG Test loss: 1.2341957712173461
Itr: 1	Epoch: 37	Train Acc: 0.60125	Test Acc: 0.575	AVG Train loss: 1.2465251207351684	AVG Test loss: 1.1871489143371583
Itr: 1	Epoch: 38	Train Acc: 0.70625	Test Acc: 0.525	AVG Train loss: 0.8534765148162842	AVG Test loss: 1.270048780441284
Itr: 1	Epoch: 39	Train Acc: 0.685	Test Acc: 0.59	AVG Train loss: 0.8871678900718689	AVG Test loss: 1.0930966544151306
Itr: 1	Epoch: 40	Train Acc: 0.75125	Test Acc: 0.465	AVG Train loss: 0.6778082156181335	AVG Test loss: 1.8372099590301514
Itr: 1	Epoch: 41	Train Acc: 0.6575	Test Acc: 0.595	AVG Train loss: 1.054599142074585	AVG Test loss: 1.227991156578064
Itr: 1	Epoch: 42	Train Acc: 0.7175	Test Acc: 0.65	AVG Train loss: 0.8342770338058472	AVG Test loss: 1.0953213119506835
Itr: 1	Epoch: 43	Train Acc: 0.7725	Test Acc: 0.365	AVG Train loss: 0.6929640936851501	AVG Test loss: 2.50249942779541
Itr: 1	Epoch: 44	Train Acc: 0.575	Test Acc: 0.525	AVG Train loss: 1.5379837942123413	AVG Test loss: 1.256906147003174
Itr: 1	Epoch: 45	Train Acc: 0.7525	Test Acc: 0.545	AVG Train loss: 0.7834124302864075	AVG Test loss: 1.2488150405883789
Itr: 1	Epoch: 46	Train Acc: 0.63875	Test Acc: 0.54	AVG Train loss: 1.051391839981079	AVG Test loss: 1.3657144737243652
Itr: 1	Epoch: 47	Train Acc: 0.5925	Test Acc: 0.42	AVG Train loss: 1.2162208318710328	AVG Test loss: 1.963302083015442
Itr: 1	Epoch: 48	Train Acc: 0.57875	Test Acc: 0.645	AVG Train loss: 1.343604449033737	AVG Test loss: 1.1016614627838135
Itr: 1	Epoch: 49	Train Acc: 0.6625	Test Acc: 0.525	AVG Train loss: 0.9589612245559692	AVG Test loss: 1.4341610622406007
train set ACC of each class tensor([0.1000, 0.0688, 0.0938, 0.0475, 0.1000, 0.0750, 0.0450, 0.0850, 0.0425,
        0.0925])
[[80  0  0  0  0  0  0  0  0  0]
 [25 55  0  0  0  0  0  0  0  0]
 [ 4  1 75  0  0  0  0  0  0  0]
 [16  1  8 38 16  1  0  0  0  0]
 [ 0  0  0  0 80  0  0  0  0  0]
 [17  1  1  0  0 60  0  1  0  0]
 [40  0  0  0  0  2 36  2  0  0]
 [11  1  0  0  0  0  0 68  0  0]
 [38  1  0  0  0  5  0  2 34  0]
 [ 4  1  0  0  1  0  0  0  0 74]]
test set ACC of each class tensor([0.0950, 0.0450, 0.0700, 0.0250, 0.0850, 0.0450, 0.0050, 0.0600, 0.0200,
        0.0750])
[[19  1  0  0  0  0  0  0  0  0]
 [11  9  0  0  0  0  0  0  0  0]
 [ 4  0 14  1  1  0  0  0  0  0]
 [ 7  1  4  5  3  0  0  0  0  0]
 [ 0  0  2  1 17  0  0  0  0  0]
 [ 9  0  0  0  0  9  0  2  0  0]
 [17  0  0  0  0  2  1  0  0  0]
 [ 4  1  0  0  0  3  0 12  0  0]
 [12  0  0  0  0  3  0  1  4  0]
 [ 2  1  0  0  2  0  0  0  0 15]]
DataParallel(
  (module): ConvNet(
    (features): Sequential(
      (0): Conv2d(3, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): GroupNorm(128, 128, eps=1e-05, affine=True)
      (2): ReLU(inplace=True)
      (3): AvgPool2d(kernel_size=2, stride=2, padding=0)
      (4): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (5): GroupNorm(128, 128, eps=1e-05, affine=True)
      (6): ReLU(inplace=True)
      (7): AvgPool2d(kernel_size=2, stride=2, padding=0)
      (8): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (9): GroupNorm(128, 128, eps=1e-05, affine=True)
      (10): ReLU(inplace=True)
      (11): AvgPool2d(kernel_size=2, stride=2, padding=0)
    )
    (classifier): Linear(in_features=32768, out_features=10, bias=True)
  )
)
Itr: 2	Epoch: 0	Train Acc: 0.14375	Test Acc: 0.145	AVG Train loss: 4.368628349304199	AVG Test loss: 7.694715194702148
Itr: 2	Epoch: 1	Train Acc: 0.1675	Test Acc: 0.1	AVG Train loss: 6.513736925125122	AVG Test loss: 4.597279624938965
Itr: 2	Epoch: 2	Train Acc: 0.16125	Test Acc: 0.14	AVG Train loss: 4.592111415863037	AVG Test loss: 4.51055004119873
Itr: 2	Epoch: 3	Train Acc: 0.20875	Test Acc: 0.31	AVG Train loss: 3.633297710418701	AVG Test loss: 2.91439435005188
Itr: 2	Epoch: 4	Train Acc: 0.2725	Test Acc: 0.195	AVG Train loss: 2.9809875202178957	AVG Test loss: 4.278765134811401
Itr: 2	Epoch: 5	Train Acc: 0.19625	Test Acc: 0.245	AVG Train loss: 3.410052947998047	AVG Test loss: 3.059069457054138
Itr: 2	Epoch: 6	Train Acc: 0.28125	Test Acc: 0.295	AVG Train loss: 2.8328933715820312	AVG Test loss: 3.9435766983032225
Itr: 2	Epoch: 7	Train Acc: 0.3375	Test Acc: 0.245	AVG Train loss: 3.7102428340911864	AVG Test loss: 2.447231159210205
Itr: 2	Epoch: 8	Train Acc: 0.2975	Test Acc: 0.17	AVG Train loss: 2.413652982711792	AVG Test loss: 3.377654685974121
Itr: 2	Epoch: 9	Train Acc: 0.33375	Test Acc: 0.24	AVG Train loss: 2.6810664081573488	AVG Test loss: 3.6049216651916502
Itr: 2	Epoch: 10	Train Acc: 0.35125	Test Acc: 0.295	AVG Train loss: 2.760383434295654	AVG Test loss: 2.801701602935791
Itr: 2	Epoch: 11	Train Acc: 0.39	Test Acc: 0.37	AVG Train loss: 2.5839587211608888	AVG Test loss: 2.0842691707611083
Itr: 2	Epoch: 12	Train Acc: 0.43125	Test Acc: 0.35	AVG Train loss: 1.8010918045043944	AVG Test loss: 2.265299849510193
Itr: 2	Epoch: 13	Train Acc: 0.41625	Test Acc: 0.325	AVG Train loss: 1.8432039308547974	AVG Test loss: 2.018547296524048
Itr: 2	Epoch: 14	Train Acc: 0.4875	Test Acc: 0.45	AVG Train loss: 1.7170103645324708	AVG Test loss: 1.5684304094314576
Itr: 2	Epoch: 15	Train Acc: 0.53	Test Acc: 0.41	AVG Train loss: 1.3406614828109742	AVG Test loss: 1.7682695078849793
Itr: 2	Epoch: 16	Train Acc: 0.4175	Test Acc: 0.42	AVG Train loss: 1.8182418584823608	AVG Test loss: 1.9196968221664428
Itr: 2	Epoch: 17	Train Acc: 0.465	Test Acc: 0.405	AVG Train loss: 1.6999289083480835	AVG Test loss: 1.5915969705581665
Itr: 2	Epoch: 18	Train Acc: 0.46	Test Acc: 0.415	AVG Train loss: 1.5899662780761719	AVG Test loss: 1.62820716381073
Itr: 2	Epoch: 19	Train Acc: 0.52125	Test Acc: 0.375	AVG Train loss: 1.3954495763778687	AVG Test loss: 1.7959823989868164
Itr: 2	Epoch: 20	Train Acc: 0.48	Test Acc: 0.455	AVG Train loss: 1.559159369468689	AVG Test loss: 1.843453197479248
Itr: 2	Epoch: 21	Train Acc: 0.505	Test Acc: 0.45	AVG Train loss: 1.5300884199142457	AVG Test loss: 2.027116422653198
Itr: 2	Epoch: 22	Train Acc: 0.5825	Test Acc: 0.265	AVG Train loss: 1.297792010307312	AVG Test loss: 3.1040492057800293
Itr: 2	Epoch: 23	Train Acc: 0.43625	Test Acc: 0.405	AVG Train loss: 2.498880820274353	AVG Test loss: 2.181728496551514
Itr: 2	Epoch: 24	Train Acc: 0.5625	Test Acc: 0.465	AVG Train loss: 1.5678648519515992	AVG Test loss: 1.6100113487243652
Itr: 2	Epoch: 25	Train Acc: 0.61375	Test Acc: 0.58	AVG Train loss: 1.2182217264175415	AVG Test loss: 1.2324474143981934
Itr: 2	Epoch: 26	Train Acc: 0.675	Test Acc: 0.37	AVG Train loss: 1.0080239582061767	AVG Test loss: 2.219668159484863
Itr: 2	Epoch: 27	Train Acc: 0.44125	Test Acc: 0.5	AVG Train loss: 2.027629590034485	AVG Test loss: 1.525773801803589
Itr: 2	Epoch: 28	Train Acc: 0.63125	Test Acc: 0.505	AVG Train loss: 1.1692629051208496	AVG Test loss: 1.4951062965393067
Itr: 2	Epoch: 29	Train Acc: 0.6425	Test Acc: 0.55	AVG Train loss: 1.108938775062561	AVG Test loss: 1.325653338432312
Itr: 2	Epoch: 30	Train Acc: 0.65	Test Acc: 0.545	AVG Train loss: 1.153444731235504	AVG Test loss: 1.3442475175857544
Itr: 2	Epoch: 31	Train Acc: 0.58625	Test Acc: 0.5	AVG Train loss: 1.2638868284225464	AVG Test loss: 1.5463324403762817
Itr: 2	Epoch: 32	Train Acc: 0.59875	Test Acc: 0.445	AVG Train loss: 1.1576808023452758	AVG Test loss: 1.6679064655303955
Itr: 2	Epoch: 33	Train Acc: 0.625	Test Acc: 0.48	AVG Train loss: 1.1896233177185058	AVG Test loss: 1.6511098337173462
Itr: 2	Epoch: 34	Train Acc: 0.64125	Test Acc: 0.37	AVG Train loss: 1.1213775300979614	AVG Test loss: 1.9517458248138428
Itr: 2	Epoch: 35	Train Acc: 0.44125	Test Acc: 0.345	AVG Train loss: 1.8566929578781128	AVG Test loss: 2.1999893760681153
Itr: 2	Epoch: 36	Train Acc: 0.51	Test Acc: 0.455	AVG Train loss: 1.5582660341262817	AVG Test loss: 1.6380572128295898
Itr: 2	Epoch: 37	Train Acc: 0.5525	Test Acc: 0.435	AVG Train loss: 1.3726762914657593	AVG Test loss: 1.9267546653747558
Itr: 2	Epoch: 38	Train Acc: 0.62375	Test Acc: 0.64	AVG Train loss: 1.2637483096122741	AVG Test loss: 1.0848210120201112
Itr: 2	Epoch: 39	Train Acc: 0.67125	Test Acc: 0.6	AVG Train loss: 0.9660413312911987	AVG Test loss: 1.1578481388092041
Itr: 2	Epoch: 40	Train Acc: 0.63125	Test Acc: 0.58	AVG Train loss: 1.0590242314338685	AVG Test loss: 1.3159639596939088
Itr: 2	Epoch: 41	Train Acc: 0.73625	Test Acc: 0.545	AVG Train loss: 0.7822086119651794	AVG Test loss: 1.419256763458252
Itr: 2	Epoch: 42	Train Acc: 0.685	Test Acc: 0.56	AVG Train loss: 0.8829255199432373	AVG Test loss: 1.2061458897590638
Itr: 2	Epoch: 43	Train Acc: 0.74	Test Acc: 0.56	AVG Train loss: 0.75500981092453	AVG Test loss: 1.2938752222061156
Itr: 2	Epoch: 44	Train Acc: 0.745	Test Acc: 0.605	AVG Train loss: 0.7073636054992676	AVG Test loss: 1.1431420373916625
Itr: 2	Epoch: 45	Train Acc: 0.62	Test Acc: 0.52	AVG Train loss: 1.2592951202392577	AVG Test loss: 1.424540514945984
Itr: 2	Epoch: 46	Train Acc: 0.71375	Test Acc: 0.6	AVG Train loss: 0.8928992450237274	AVG Test loss: 1.1456988954544067
Itr: 2	Epoch: 47	Train Acc: 0.69375	Test Acc: 0.545	AVG Train loss: 0.9244849634170532	AVG Test loss: 1.411303825378418
Itr: 2	Epoch: 48	Train Acc: 0.74	Test Acc: 0.53	AVG Train loss: 0.8079372596740723	AVG Test loss: 1.4891457676887512
Itr: 2	Epoch: 49	Train Acc: 0.7225	Test Acc: 0.57	AVG Train loss: 0.802826554775238	AVG Test loss: 1.1315370655059815
train set ACC of each class tensor([0.0975, 0.0700, 0.0938, 0.0875, 0.0975, 0.0613, 0.0662, 0.0862, 0.1000,
        0.0975])
[[78  0  0  0  0  0  0  1  1  0]
 [14 56  0  0  0  0  0  3  7  0]
 [ 0  0 75  0  0  0  0  0  5  0]
 [ 1  0  1 70  1  0  0  0  7  0]
 [ 0  0  0  0 78  0  0  0  2  0]
 [ 2  0  1  0  0 49  1  1 26  0]
 [ 5  0  0  0  0  0 53  2 20  0]
 [ 2  0  0  0  0  1  0 69  8  0]
 [ 0  0  0  0  0  0  0  0 80  0]
 [ 0  0  0  0  0  0  0  1  1 78]]
test set ACC of each class tensor([0.0750, 0.0300, 0.0750, 0.0600, 0.0850, 0.0150, 0.0200, 0.0400, 0.0900,
        0.0800])
[[15  2  0  0  0  0  0  1  2  0]
 [ 8  6  0  0  0  1  0  1  4  0]
 [ 0  0 15  3  1  0  0  0  1  0]
 [ 2  0  1 12  0  0  0  0  5  0]
 [ 0  0  1  2 17  0  0  0  0  0]
 [ 0  0  0  0  0  3  0  0 17  0]
 [ 4  0  0  0  0  1  4  0 11  0]
 [ 0  1  0  0  0  3  0  8  8  0]
 [ 1  0  0  0  0  0  0  1 18  0]
 [ 0  0  0  1  1  0  0  0  2 16]]
```

# lr 0.06
```cmd
ssh://sbcaesar@dais10.uwb.edu:22/data/sbcaesar/xuan_venv/bin/python3 -u /data/sbcaesar/mac_galaxy/buffer.py
Hyper-parameters: 
 {'dataset': 'gzoo2', 'subset': 'imagenette', 'model': 'ConvNet', 'num_experts': 10, 'lr_teacher': 0.06, 'batch_train': 256, 'batch_real': 256, 'dsa': True, 'dsa_strategy': 'color_crop_cutout_flip_scale_rotate', 'data_path': 'data', 'buffer_path': '/data/sbcaesar/galaxy_buffers', 'train_epochs': 50, 'zca': False, 'decay': False, 'mom': 0, 'l2': 0, 'save_interval': 10, 'device': 'cuda', 'dsa_param': <utils.ParamDiffAug object at 0x7fdd78b60dc0>}
BUILDING DATASET
100%|█████████████████████████████████████| 800/800 [00:00<00:00, 163648.22it/s]
800it [00:00, 3817341.52it/s]
class c = 0: 80 real images
class c = 1: 80 real images
class c = 2: 80 real images
class c = 3: 80 real images
class c = 4: 80 real images
class c = 5: 80 real images
class c = 6: 80 real images
class c = 7: 80 real images
class c = 8: 80 real images
class c = 9: 80 real images
real images channel 0, mean = 0.0003, std = 1.0003
real images channel 1, mean = -0.0002, std = 1.0004
real images channel 2, mean = 0.0006, std = 1.0005
Add weight to loss function tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
DC augmentation parameters: 
 {'crop': 4, 'scale': 0.2, 'rotate': 45, 'noise': 0.001, 'strategy': 'crop_scale_rotate'}
DataParallel(
  (module): ConvNet(
    (features): Sequential(
      (0): Conv2d(3, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): GroupNorm(128, 128, eps=1e-05, affine=True)
      (2): ReLU(inplace=True)
      (3): AvgPool2d(kernel_size=2, stride=2, padding=0)
      (4): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (5): GroupNorm(128, 128, eps=1e-05, affine=True)
      (6): ReLU(inplace=True)
      (7): AvgPool2d(kernel_size=2, stride=2, padding=0)
      (8): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (9): GroupNorm(128, 128, eps=1e-05, affine=True)
      (10): ReLU(inplace=True)
      (11): AvgPool2d(kernel_size=2, stride=2, padding=0)
    )
    (classifier): Linear(in_features=32768, out_features=10, bias=True)
  )
)
Itr: 0	Epoch: 0	Train Acc: 0.10875	Test Acc: 0.145	AVG Train loss: 16.953182296752928	AVG Test loss: 68.38720367431641
Itr: 0	Epoch: 1	Train Acc: 0.16125	Test Acc: 0.155	AVG Train loss: 56.820487060546874	AVG Test loss: 15.415781021118164
Itr: 0	Epoch: 2	Train Acc: 0.15125	Test Acc: 0.25	AVG Train loss: 8.546497840881347	AVG Test loss: 2.5209626960754394
Itr: 0	Epoch: 3	Train Acc: 0.17	Test Acc: 0.215	AVG Train loss: 2.510707769393921	AVG Test loss: 2.1879886054992674
Itr: 0	Epoch: 4	Train Acc: 0.20375	Test Acc: 0.25	AVG Train loss: 2.1796947574615477	AVG Test loss: 2.0997183322906494
Itr: 0	Epoch: 5	Train Acc: 0.25	Test Acc: 0.27	AVG Train loss: 2.091883907318115	AVG Test loss: 2.0697153186798096
Itr: 0	Epoch: 6	Train Acc: 0.21	Test Acc: 0.24	AVG Train loss: 2.0954719257354735	AVG Test loss: 2.066692981719971
Itr: 0	Epoch: 7	Train Acc: 0.22625	Test Acc: 0.275	AVG Train loss: 2.0155801296234133	AVG Test loss: 2.0220601463317873
Itr: 0	Epoch: 8	Train Acc: 0.25625	Test Acc: 0.195	AVG Train loss: 2.0367108821868896	AVG Test loss: 2.0736623001098633
Itr: 0	Epoch: 9	Train Acc: 0.2575	Test Acc: 0.23	AVG Train loss: 1.998042984008789	AVG Test loss: 2.0021883630752564
Itr: 0	Epoch: 10	Train Acc: 0.2725	Test Acc: 0.315	AVG Train loss: 2.0977760934829712	AVG Test loss: 1.9375622463226319
Itr: 0	Epoch: 11	Train Acc: 0.3225	Test Acc: 0.265	AVG Train loss: 2.174144678115845	AVG Test loss: 1.9751387977600097
Itr: 0	Epoch: 12	Train Acc: 0.33	Test Acc: 0.31	AVG Train loss: 1.9540723705291747	AVG Test loss: 1.870534906387329
Itr: 0	Epoch: 13	Train Acc: 0.37875	Test Acc: 0.31	AVG Train loss: 1.7152968835830689	AVG Test loss: 1.8416397380828857
Itr: 0	Epoch: 14	Train Acc: 0.34	Test Acc: 0.315	AVG Train loss: 1.8300739097595216	AVG Test loss: 1.993616542816162
Itr: 0	Epoch: 15	Train Acc: 0.30125	Test Acc: 0.265	AVG Train loss: 1.9678187561035156	AVG Test loss: 1.8422751426696777
Itr: 0	Epoch: 16	Train Acc: 0.37875	Test Acc: 0.31	AVG Train loss: 1.7439617347717284	AVG Test loss: 1.8032426309585572
Itr: 0	Epoch: 17	Train Acc: 0.44125	Test Acc: 0.245	AVG Train loss: 1.5951306629180908	AVG Test loss: 1.9053402757644653
Itr: 0	Epoch: 18	Train Acc: 0.3075	Test Acc: 0.31	AVG Train loss: 1.9462641668319702	AVG Test loss: 1.9029268503189087
Itr: 0	Epoch: 19	Train Acc: 0.355	Test Acc: 0.465	AVG Train loss: 1.8347384452819824	AVG Test loss: 1.5547247409820557
Itr: 0	Epoch: 20	Train Acc: 0.455	Test Acc: 0.34	AVG Train loss: 1.5066209840774536	AVG Test loss: 1.7493954944610595
Itr: 0	Epoch: 21	Train Acc: 0.40375	Test Acc: 0.37	AVG Train loss: 1.6894030237197877	AVG Test loss: 1.564872760772705
Itr: 0	Epoch: 22	Train Acc: 0.435	Test Acc: 0.35	AVG Train loss: 1.500857810974121	AVG Test loss: 1.5932690715789795
Itr: 0	Epoch: 23	Train Acc: 0.46125	Test Acc: 0.345	AVG Train loss: 1.537736086845398	AVG Test loss: 1.974602928161621
Itr: 0	Epoch: 24	Train Acc: 0.395	Test Acc: 0.395	AVG Train loss: 1.7656305503845215	AVG Test loss: 1.506347794532776
Itr: 0	Epoch: 25	Train Acc: 0.495	Test Acc: 0.435	AVG Train loss: 1.495199146270752	AVG Test loss: 1.5570211553573607
Itr: 0	Epoch: 26	Train Acc: 0.42375	Test Acc: 0.425	AVG Train loss: 1.5950986289978026	AVG Test loss: 1.5093293762207032
Itr: 0	Epoch: 27	Train Acc: 0.3575	Test Acc: 0.46	AVG Train loss: 1.7124940156936646	AVG Test loss: 1.4758556509017944
Itr: 0	Epoch: 28	Train Acc: 0.4075	Test Acc: 0.49	AVG Train loss: 1.6793873023986816	AVG Test loss: 1.4167646980285644
Itr: 0	Epoch: 29	Train Acc: 0.42125	Test Acc: 0.465	AVG Train loss: 1.6682982635498047	AVG Test loss: 1.6766982173919678
Itr: 0	Epoch: 30	Train Acc: 0.53	Test Acc: 0.345	AVG Train loss: 1.3972552299499512	AVG Test loss: 1.911985936164856
Itr: 0	Epoch: 31	Train Acc: 0.42125	Test Acc: 0.415	AVG Train loss: 1.7354665517807006	AVG Test loss: 1.61539701461792
Itr: 0	Epoch: 32	Train Acc: 0.44	Test Acc: 0.375	AVG Train loss: 1.7129317903518677	AVG Test loss: 1.6433872318267821
Itr: 0	Epoch: 33	Train Acc: 0.41375	Test Acc: 0.355	AVG Train loss: 1.608062105178833	AVG Test loss: 1.623897943496704
Itr: 0	Epoch: 34	Train Acc: 0.4375	Test Acc: 0.44	AVG Train loss: 1.5278004217147827	AVG Test loss: 1.4031530618667603
Itr: 0	Epoch: 35	Train Acc: 0.54125	Test Acc: 0.385	AVG Train loss: 1.403721923828125	AVG Test loss: 1.8263426685333253
Itr: 0	Epoch: 36	Train Acc: 0.555	Test Acc: 0.505	AVG Train loss: 1.4204801368713378	AVG Test loss: 1.3822315454483032
Itr: 0	Epoch: 37	Train Acc: 0.3625	Test Acc: 0.43	AVG Train loss: 1.8118251705169677	AVG Test loss: 1.6386642456054688
Itr: 0	Epoch: 38	Train Acc: 0.425	Test Acc: 0.505	AVG Train loss: 1.8012608909606933	AVG Test loss: 1.3664966487884522
Itr: 0	Epoch: 39	Train Acc: 0.455	Test Acc: 0.48	AVG Train loss: 1.5525427150726319	AVG Test loss: 1.3155631160736083
Itr: 0	Epoch: 40	Train Acc: 0.59375	Test Acc: 0.535	AVG Train loss: 1.2362730503082275	AVG Test loss: 1.313019814491272
Itr: 0	Epoch: 41	Train Acc: 0.62125	Test Acc: 0.505	AVG Train loss: 1.139338607788086	AVG Test loss: 1.2975600481033325
Itr: 0	Epoch: 42	Train Acc: 0.62375	Test Acc: 0.39	AVG Train loss: 1.1156415271759033	AVG Test loss: 1.8554465770721436
Itr: 0	Epoch: 43	Train Acc: 0.535	Test Acc: 0.485	AVG Train loss: 1.4933098459243774	AVG Test loss: 1.3866602087020874
Itr: 0	Epoch: 44	Train Acc: 0.53625	Test Acc: 0.375	AVG Train loss: 1.3996584701538086	AVG Test loss: 1.804101629257202
Itr: 0	Epoch: 45	Train Acc: 0.44	Test Acc: 0.5	AVG Train loss: 1.741145944595337	AVG Test loss: 1.2597090101242066
Itr: 0	Epoch: 46	Train Acc: 0.64	Test Acc: 0.465	AVG Train loss: 1.1369489097595216	AVG Test loss: 1.389370698928833
Itr: 0	Epoch: 47	Train Acc: 0.57125	Test Acc: 0.53	AVG Train loss: 1.348936038017273	AVG Test loss: 1.4124354171752929
Itr: 0	Epoch: 48	Train Acc: 0.4875	Test Acc: 0.335	AVG Train loss: 1.4543073654174805	AVG Test loss: 2.049860620498657
Itr: 0	Epoch: 49	Train Acc: 0.54875	Test Acc: 0.425	AVG Train loss: 1.5327147436141968	AVG Test loss: 1.5101373863220215
train set ACC of each class tensor([0.0000, 0.1000, 0.0862, 0.0825, 0.0850, 0.0550, 0.0275, 0.0088, 0.0237,
        0.0913])
[[ 0 75  0  3  0  1  1  0  0  0]
 [ 0 80  0  0  0  0  0  0  0  0]
 [ 0  8 69  3  0  0  0  0  0  0]
 [ 0  8  2 66  3  1  0  0  0  0]
 [ 0  0  3  8 68  0  0  0  0  1]
 [ 0 34  0  0  0 44  0  0  2  0]
 [ 0 34  0  0  0 21 22  1  2  0]
 [ 0 52  1  0  0 20  0  7  0  0]
 [ 0 35  0  4  0 20  2  0 19  0]
 [ 0  3  1  1  2  0  0  0  0 73]]
test set ACC of each class tensor([0.0000, 0.0950, 0.0800, 0.0550, 0.0600, 0.0450, 0.0050, 0.0000, 0.0100,
        0.0750])
[[ 0 19  0  0  0  1  0  0  0  0]
 [ 0 19  0  0  0  1  0  0  0  0]
 [ 0  2 16  2  0  0  0  0  0  0]
 [ 0  3  4 11  2  0  0  0  0  0]
 [ 0  0  3  5 12  0  0  0  0  0]
 [ 0 10  0  0  0  9  0  0  1  0]
 [ 0 11  0  0  0  6  1  0  2  0]
 [ 0  9  0  0  0 10  0  0  1  0]
 [ 0  8  0  1  0  8  1  0  2  0]
 [ 0  2  0  1  0  1  1  0  0 15]]
DataParallel(
  (module): ConvNet(
    (features): Sequential(
      (0): Conv2d(3, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): GroupNorm(128, 128, eps=1e-05, affine=True)
      (2): ReLU(inplace=True)
      (3): AvgPool2d(kernel_size=2, stride=2, padding=0)
      (4): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (5): GroupNorm(128, 128, eps=1e-05, affine=True)
      (6): ReLU(inplace=True)
      (7): AvgPool2d(kernel_size=2, stride=2, padding=0)
      (8): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (9): GroupNorm(128, 128, eps=1e-05, affine=True)
      (10): ReLU(inplace=True)
      (11): AvgPool2d(kernel_size=2, stride=2, padding=0)
    )
    (classifier): Linear(in_features=32768, out_features=10, bias=True)
  )
)
Itr: 1	Epoch: 0	Train Acc: 0.16125	Test Acc: 0.145	AVG Train loss: 13.273557205200195	AVG Test loss: 75.56193115234375
Itr: 1	Epoch: 1	Train Acc: 0.15	Test Acc: 0.185	AVG Train loss: 66.52030822753906	AVG Test loss: 36.48496957778931
Itr: 1	Epoch: 2	Train Acc: 0.1975	Test Acc: 0.19	AVG Train loss: 19.324784212112426	AVG Test loss: 3.5122555923461913
Itr: 1	Epoch: 3	Train Acc: 0.1925	Test Acc: 0.16	AVG Train loss: 2.824844856262207	AVG Test loss: 2.2139148044586183
Itr: 1	Epoch: 4	Train Acc: 0.18625	Test Acc: 0.255	AVG Train loss: 2.187813148498535	AVG Test loss: 2.11358678817749
Itr: 1	Epoch: 5	Train Acc: 0.23625	Test Acc: 0.25	AVG Train loss: 2.127857389450073	AVG Test loss: 2.0594852066040037
Itr: 1	Epoch: 6	Train Acc: 0.23875	Test Acc: 0.235	AVG Train loss: 2.229639835357666	AVG Test loss: 2.0528406381607054
Itr: 1	Epoch: 7	Train Acc: 0.2775	Test Acc: 0.28	AVG Train loss: 2.093685283660889	AVG Test loss: 1.9813235425949096
Itr: 1	Epoch: 8	Train Acc: 0.23625	Test Acc: 0.36	AVG Train loss: 2.062790298461914	AVG Test loss: 1.9397670602798462
Itr: 1	Epoch: 9	Train Acc: 0.365	Test Acc: 0.355	AVG Train loss: 2.0204410266876223	AVG Test loss: 1.9035308790206908
Itr: 1	Epoch: 10	Train Acc: 0.425	Test Acc: 0.385	AVG Train loss: 1.8516300678253175	AVG Test loss: 1.800016498565674
Itr: 1	Epoch: 11	Train Acc: 0.43125	Test Acc: 0.205	AVG Train loss: 1.7440817165374756	AVG Test loss: 2.2401461791992188
Itr: 1	Epoch: 12	Train Acc: 0.325	Test Acc: 0.345	AVG Train loss: 1.9536020946502686	AVG Test loss: 1.8388594722747802
Itr: 1	Epoch: 13	Train Acc: 0.35875	Test Acc: 0.44	AVG Train loss: 1.8117924404144288	AVG Test loss: 1.810480890274048
Itr: 1	Epoch: 14	Train Acc: 0.38	Test Acc: 0.37	AVG Train loss: 1.957040777206421	AVG Test loss: 1.7621695899963379
Itr: 1	Epoch: 15	Train Acc: 0.42125	Test Acc: 0.285	AVG Train loss: 1.6395295429229737	AVG Test loss: 1.8330319213867188
Itr: 1	Epoch: 16	Train Acc: 0.3725	Test Acc: 0.44	AVG Train loss: 1.6810458707809448	AVG Test loss: 1.7343980598449706
Itr: 1	Epoch: 17	Train Acc: 0.41625	Test Acc: 0.39	AVG Train loss: 1.5412938451766969	AVG Test loss: 1.6060016870498657
Itr: 1	Epoch: 18	Train Acc: 0.4275	Test Acc: 0.415	AVG Train loss: 1.494709300994873	AVG Test loss: 1.5676443004608154
Itr: 1	Epoch: 19	Train Acc: 0.3725	Test Acc: 0.245	AVG Train loss: 1.7538547325134277	AVG Test loss: 2.0580187892913817
Itr: 1	Epoch: 20	Train Acc: 0.37125	Test Acc: 0.47	AVG Train loss: 1.7937257862091065	AVG Test loss: 1.5445224571228027
Itr: 1	Epoch: 21	Train Acc: 0.47375	Test Acc: 0.415	AVG Train loss: 1.567425618171692	AVG Test loss: 1.6000782537460327
Itr: 1	Epoch: 22	Train Acc: 0.47125	Test Acc: 0.405	AVG Train loss: 1.4579328632354736	AVG Test loss: 1.6158632373809814
Itr: 1	Epoch: 23	Train Acc: 0.41125	Test Acc: 0.43	AVG Train loss: 1.6649592542648315	AVG Test loss: 1.5875913333892822
Itr: 1	Epoch: 24	Train Acc: 0.51	Test Acc: 0.405	AVG Train loss: 1.3927868366241456	AVG Test loss: 1.5702363443374634
Itr: 1	Epoch: 25	Train Acc: 0.395	Test Acc: 0.435	AVG Train loss: 1.5806929206848144	AVG Test loss: 1.5287713766098023
Itr: 1	Epoch: 26	Train Acc: 0.4975	Test Acc: 0.495	AVG Train loss: 1.4724539279937745	AVG Test loss: 1.4495608377456666
Itr: 1	Epoch: 27	Train Acc: 0.51875	Test Acc: 0.44	AVG Train loss: 1.3883458995819091	AVG Test loss: 1.647820372581482
Itr: 1	Epoch: 28	Train Acc: 0.53375	Test Acc: 0.465	AVG Train loss: 1.3162067890167237	AVG Test loss: 1.4286158609390258
Itr: 1	Epoch: 29	Train Acc: 0.435	Test Acc: 0.45	AVG Train loss: 1.618716983795166	AVG Test loss: 1.4399827194213868
Itr: 1	Epoch: 30	Train Acc: 0.42625	Test Acc: 0.41	AVG Train loss: 1.5593482589721679	AVG Test loss: 1.6769049501419067
Itr: 1	Epoch: 31	Train Acc: 0.4725	Test Acc: 0.57	AVG Train loss: 1.5295863676071166	AVG Test loss: 1.3168281745910644
Itr: 1	Epoch: 32	Train Acc: 0.42	Test Acc: 0.435	AVG Train loss: 1.6936521530151367	AVG Test loss: 1.5114138984680177
Itr: 1	Epoch: 33	Train Acc: 0.45625	Test Acc: 0.505	AVG Train loss: 1.509842085838318	AVG Test loss: 1.3543874835968017
Itr: 1	Epoch: 34	Train Acc: 0.59875	Test Acc: 0.57	AVG Train loss: 1.1932597160339355	AVG Test loss: 1.2394731903076173
Itr: 1	Epoch: 35	Train Acc: 0.57625	Test Acc: 0.51	AVG Train loss: 1.2213706636428834	AVG Test loss: 1.2548534536361695
Itr: 1	Epoch: 36	Train Acc: 0.53875	Test Acc: 0.495	AVG Train loss: 1.3668129110336305	AVG Test loss: 1.3602856969833375
Itr: 1	Epoch: 37	Train Acc: 0.56125	Test Acc: 0.525	AVG Train loss: 1.2956736755371094	AVG Test loss: 1.219122896194458
Itr: 1	Epoch: 38	Train Acc: 0.5425	Test Acc: 0.355	AVG Train loss: 1.3632668018341065	AVG Test loss: 1.581733741760254
Itr: 1	Epoch: 39	Train Acc: 0.465	Test Acc: 0.45	AVG Train loss: 1.5040855121612549	AVG Test loss: 1.4536698293685912
Itr: 1	Epoch: 40	Train Acc: 0.515	Test Acc: 0.43	AVG Train loss: 1.4361433172225953	AVG Test loss: 1.4838034200668335
Itr: 1	Epoch: 41	Train Acc: 0.46625	Test Acc: 0.545	AVG Train loss: 1.5381661701202392	AVG Test loss: 1.2249825096130371
Itr: 1	Epoch: 42	Train Acc: 0.5475	Test Acc: 0.465	AVG Train loss: 1.2695654225349426	AVG Test loss: 1.4427210807800293
Itr: 1	Epoch: 43	Train Acc: 0.58875	Test Acc: 0.48	AVG Train loss: 1.291089506149292	AVG Test loss: 1.3923929262161254
Itr: 1	Epoch: 44	Train Acc: 0.57125	Test Acc: 0.485	AVG Train loss: 1.2859897136688232	AVG Test loss: 1.3989156246185304
Itr: 1	Epoch: 45	Train Acc: 0.5925	Test Acc: 0.575	AVG Train loss: 1.2253652906417847	AVG Test loss: 1.1839513826370238
Itr: 1	Epoch: 46	Train Acc: 0.64	Test Acc: 0.48	AVG Train loss: 1.0399880909919739	AVG Test loss: 1.3818897819519043
Itr: 1	Epoch: 47	Train Acc: 0.53	Test Acc: 0.54	AVG Train loss: 1.3630541515350343	AVG Test loss: 1.2062569856643677
Itr: 1	Epoch: 48	Train Acc: 0.61875	Test Acc: 0.49	AVG Train loss: 1.1286623191833496	AVG Test loss: 1.4802835893630981
Itr: 1	Epoch: 49	Train Acc: 0.5825	Test Acc: 0.49	AVG Train loss: 1.2247581815719604	AVG Test loss: 1.3637751197814942
train set ACC of each class tensor([0.0463, 0.0950, 0.0812, 0.0512, 0.0938, 0.0288, 0.0300, 0.0400, 0.0200,
        0.0950])
[[37 40  0  3  0  0  0  0  0  0]
 [ 3 76  0  0  0  0  1  0  0  0]
 [ 0 15 65  0  0  0  0  0  0  0]
 [ 0  9 17 41 12  0  1  0  0  0]
 [ 0  0  2  2 75  0  0  0  0  1]
 [ 0 54  1  0  0 23  1  1  0  0]
 [ 5 49  0  0  0  0 24  2  0  0]
 [ 1 40  1  0  0  6  0 32  0  0]
 [ 3 50  3  0  0  5  2  1 16  0]
 [ 0  3  1  0  0  0  0  0  0 76]]
test set ACC of each class tensor([0.0550, 0.1000, 0.0750, 0.0450, 0.0800, 0.0200, 0.0100, 0.0200, 0.0100,
        0.0750])
[[11  9  0  0  0  0  0  0  0  0]
 [ 0 20  0  0  0  0  0  0  0  0]
 [ 0  4 15  1  0  0  0  0  0  0]
 [ 0  5  4  9  2  0  0  0  0  0]
 [ 0  0  2  2 16  0  0  0  0  0]
 [ 0 15  0  0  0  4  0  0  1  0]
 [ 3 15  0  0  0  0  2  0  0  0]
 [ 0 13  0  0  0  3  0  4  0  0]
 [ 0 12  0  1  0  1  3  0  2  1]
 [ 0  2  0  1  2  0  0  0  0 15]]
DataParallel(
  (module): ConvNet(
    (features): Sequential(
      (0): Conv2d(3, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): GroupNorm(128, 128, eps=1e-05, affine=True)
      (2): ReLU(inplace=True)
      (3): AvgPool2d(kernel_size=2, stride=2, padding=0)
      (4): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (5): GroupNorm(128, 128, eps=1e-05, affine=True)
      (6): ReLU(inplace=True)
      (7): AvgPool2d(kernel_size=2, stride=2, padding=0)
      (8): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (9): GroupNorm(128, 128, eps=1e-05, affine=True)
      (10): ReLU(inplace=True)
      (11): AvgPool2d(kernel_size=2, stride=2, padding=0)
    )
    (classifier): Linear(in_features=32768, out_features=10, bias=True)
  )
)
Itr: 2	Epoch: 0	Train Acc: 0.10625	Test Acc: 0.1	AVG Train loss: 14.931098175048827	AVG Test loss: 43.46874603271485
Itr: 2	Epoch: 1	Train Acc: 0.1275	Test Acc: 0.2	AVG Train loss: 39.14881462097168	AVG Test loss: 18.78092514038086
Itr: 2	Epoch: 2	Train Acc: 0.18125	Test Acc: 0.175	AVG Train loss: 12.196818923950195	AVG Test loss: 3.6044853496551514
Itr: 2	Epoch: 3	Train Acc: 0.215	Test Acc: 0.17	AVG Train loss: 3.012586555480957	AVG Test loss: 2.1742889308929443
Itr: 2	Epoch: 4	Train Acc: 0.21125	Test Acc: 0.21	AVG Train loss: 2.108015594482422	AVG Test loss: 2.0044136238098145
Itr: 2	Epoch: 5	Train Acc: 0.26875	Test Acc: 0.32	AVG Train loss: 2.143045449256897	AVG Test loss: 1.9668945932388306
Itr: 2	Epoch: 6	Train Acc: 0.31125	Test Acc: 0.225	AVG Train loss: 2.0636188316345216	AVG Test loss: 2.1353430461883547
Itr: 2	Epoch: 7	Train Acc: 0.30625	Test Acc: 0.25	AVG Train loss: 2.159897208213806	AVG Test loss: 2.2705676126480103
Itr: 2	Epoch: 8	Train Acc: 0.26375	Test Acc: 0.215	AVG Train loss: 2.0839982986450196	AVG Test loss: 2.1369046211242675
Itr: 2	Epoch: 9	Train Acc: 0.27875	Test Acc: 0.255	AVG Train loss: 2.068892660140991	AVG Test loss: 1.989776554107666
Itr: 2	Epoch: 10	Train Acc: 0.2675	Test Acc: 0.29	AVG Train loss: 2.0138737773895263	AVG Test loss: 2.0778014564514162
Itr: 2	Epoch: 11	Train Acc: 0.30875	Test Acc: 0.285	AVG Train loss: 2.002443380355835	AVG Test loss: 1.8521271467208862
Itr: 2	Epoch: 12	Train Acc: 0.3575	Test Acc: 0.28	AVG Train loss: 1.8743639945983888	AVG Test loss: 1.9458656311035156
Itr: 2	Epoch: 13	Train Acc: 0.3125	Test Acc: 0.325	AVG Train loss: 1.989824471473694	AVG Test loss: 1.872975034713745
Itr: 2	Epoch: 14	Train Acc: 0.3525	Test Acc: 0.325	AVG Train loss: 1.8169976711273192	AVG Test loss: 1.7599969291687012
Itr: 2	Epoch: 15	Train Acc: 0.37375	Test Acc: 0.245	AVG Train loss: 1.7859968662261962	AVG Test loss: 1.9883139085769654
Itr: 2	Epoch: 16	Train Acc: 0.3475	Test Acc: 0.4	AVG Train loss: 1.9105315685272217	AVG Test loss: 1.7092950201034547
Itr: 2	Epoch: 17	Train Acc: 0.38125	Test Acc: 0.265	AVG Train loss: 1.7327391529083251	AVG Test loss: 1.8321000003814698
Itr: 2	Epoch: 18	Train Acc: 0.37125	Test Acc: 0.31	AVG Train loss: 1.7889988803863526	AVG Test loss: 1.8789927291870117
Itr: 2	Epoch: 19	Train Acc: 0.3625	Test Acc: 0.365	AVG Train loss: 1.8035300970077515	AVG Test loss: 1.7790240573883056
Itr: 2	Epoch: 20	Train Acc: 0.35875	Test Acc: 0.37	AVG Train loss: 1.7424649572372437	AVG Test loss: 1.6299411153793335
```