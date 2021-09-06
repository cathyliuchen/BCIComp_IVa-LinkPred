# BCIComp_IVa : LinkPred 

## Paper
- Systematic Enchancement of Functional Connectivity In Brain-Computer Interfacing Using Common Spatial Patterns and Tangent Space Mapping

## Dataset

- BCI Competition 2015 NER
- IVa 
- LinkPred_Graphs
  - The epochs are pre-processed at [8, 30] Hz using a 4th order butterworth filter and segmented from the trigger onset to 3.5 seconds. The constants file contains channel related information and sampling rate. To load the numpy files, use the following command: np.load(filename, allow_pickle=True)

  - To retrieve individual information, use:

    - Example: get the epochs from the numpy file for subject aa: test = np.load(path + '/preprocessed_epochs_aa.npy', allow_pickle=True) epochs = test.item().get('epochs')

- Data Download:
 -  链接：https://pan.baidu.com/s/17Kruh22HYh0-dFhVdY8pOA  提取码：85qt

## Refered Papers：

- Paper Map: http://naotu.baidu.com/file/52a7a9fbcf4d5845a7266da5f3937b3d?token=a0e8ae77a1b3bb78

## Comp2015 & IVa Dataset Thinking

- https://shimo.im/docs/gpHTHChVVjtCPJDK/ 「Comp2015 & IVa Dataset Thinkings」(Comment allowed)

## Working Log

- https://shimo.im/docs/38d8PTDxhTKcrxXW/ 「Project_Saugat2019 Working Log」

## Code Ref：
- GCN-GAN: https://github.com/jiangqn/GCN-GAN-pytorch
- SubGNN：https://github.com/mims-harvard/SubGNN
