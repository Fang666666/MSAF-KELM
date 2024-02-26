# PyTorch Demo of the Hyperspectral Image Classification method - MSAF-KELM.

Using the code should cite the following paper:

L. Sun, Y. Fang, Y. Chen, W. Huang, Z. Wu and B. Jeon, "Multi-Structure KELM With Attention Fusion Strategy for Hyperspectral Image Classification," in IEEE Transactions on Geoscience and Remote Sensing, vol. 60, pp. 1-17, 2022, Art no. 5539217, doi: 10.1109/TGRS.2022.3208165.

@ARTICLE{9895428,
  author={Sun, Le and Fang, Yu and Chen, Yuwen and Huang, Wei and Wu, Zebin and Jeon, Byeungwoo},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Multi-Structure KELM With Attention Fusion Strategy for Hyperspectral Image Classification}, 
  year={2022},
  volume={60},
  number={},
  pages={1-17},
  keywords={Feature extraction;Training;Deep learning;Kernel;Hyperspectral imaging;Electronic mail;Extreme learning machines;Attention mechanisms;hyperspectral image (HSI) classification;kernel extreme learning machine (KELM);multifeature;multiscale (MS)},
  doi={10.1109/TGRS.2022.3208165}}


# Description.
Hyperspectral image (HSI) classification refers to accurately corresponding each pixel in an HSI to a land-cover label. Recently, the successful application of multi-scale and multi-feature methods has greatly improved the performance of HSI classification due to their enhanced utilization of the available spectral-spatial information. However, as the number of scales and the number of features increases, it becomes more difficult to achieve an optimal degree of fusion for multiple classifiers (e.g., Kernel Extreme Learning Machine, KELM). On the other hand, a limited sample size of the HSI may cause overfitting problems, which seriously affects the classification accuracy. Therefore, in this paper, a novel Multi-Structure KELM with Attention Fusion Strategy (MSAF-KELM) is proposed to achieve accurate fusion of multiple classifiers for effective HSI classification with ultra-small sample rates. First, a multi-structure network is built that combines multiple scales and multiple features to extract abundant spectral-spatial information. Second, a fast and efficient KELM is employed to enable rapid classification. Finally, a Weighted Self Attention Fusion Strategy (WSAFS) is introduced, which combines the output weights of each KELM sub-branch and the self-attention mechanism to achieve an efficient fusion result on multi-structure networks. We conducted experiments on four types of HSI datasets with different evaluation methods and compared them with several classical and state-of-the-art methods, which demonstrate the excellent performance of our method on ultra-small sample rates.
