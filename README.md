# BCIComp_IVa : LinkPred 

## Paper
- Systematic Enchancement of Functional Connectivity In Brain-Computer Interfacing Using Common Spatial Patterns and Tangent Space Mapping

## Dataset

- IVa 
  - The epochs are pre-processed at [8, 30] Hz using a 4th order butterworth filter and segmented from the trigger onset to 3.5 seconds. The constants file contains channel related information and sampling rate. To load the numpy files, use the following command: np.load(filename, allow_pickle=True)

  - To retrieve individual information, use:

    - Example: get the epochs from the numpy file for subject aa: test = np.load(path + '/preprocessed_epochs_aa.npy', allow_pickle=True) epochs = test.item().get('epochs')

- Data Download:
 -  Link：https://pan.baidu.com/s/1DXca4jFZBLcpm-kq5mPSWA   Extract Number：u54w

## Data file structure：
-   data/
      - epochs/constants.npy, preprocessed_epochs_aa.npy, preprocessed_epochs_al.npy, preprocessed_epochs_av.npy, preprocessed_epochs_aw.npy, preprocessed_epochs_ay.np
      - adj_dict/aa...al/0.npy, 1.npy ... 
      - dataset/aa...al/ rt_foot_train_data.npy, rt_foot_train_label.npy, rt_foot_test_data.npy, rt_foot_test_label.npy, 
                     rt_hand_train_data.npy, rt_hand_train_label.npy, rt_hand_test_data.npy, rt_hand_test_label.npy
      - result/aa...al/ rt_hand_train_data_generator.pkl, rt_foot_train_data_generator.pkl

## Refered Papers：

- Paper Map: http://naotu.baidu.com/file/52a7a9fbcf4d5845a7266da5f3937b3d?token=a0e8ae77a1b3bb78

## Comp2015 & IVa Dataset Thinking

- https://shimo.im/docs/gpHTHChVVjtCPJDK/ 「Comp2015 & IVa Dataset Thinkings」(Comment allowed)

## Working Log
- https://shimowendang.com/docs/38d8PTDxhTKcrxXW/ 「IVa-Comp: LinkPred Working Log」，可复制链接后用石墨文档 App 或小程序打开 (Before Oct.)

- 【腾讯文档】IVaComp Working Log 
https://docs.qq.com/doc/DU1RmTXV3b1ZCb01V

## Code Ref：
- GCN-GAN: https://github.com/jiangqn/GCN-GAN-pytorch
- SubGNN：https://github.com/mims-harvard/SubGNN
