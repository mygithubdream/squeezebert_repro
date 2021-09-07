# squeezebert_repro
SqueezeBERT runs 4.3x faster than BERT-base on the Pixel 3 while achieving competitive accuracy on the GLUE test set. It replaces several operations in self-attention layers with grouped convolutions.

测试代码见pt_reproduce.py
pytorch代码复现后结果如下：
output: tensor([[[-0.2027, -0.2009,  0.1490,  ...,  0.1702,  0.0511, -0.0263],
         [-0.0647,  0.0196, -0.0424,  ..., -0.0356, -0.0179,  0.1650],
         [-0.1433, -0.0392,  0.1251,  ..., -0.0568, -0.0579,  0.0355],
         ...,
         [-0.0220, -0.0297,  0.0214,  ...,  0.0092, -0.0790, -0.0063],
         [-0.1417, -0.0813,  0.1718,  ...,  0.3336, -0.1188,  0.1433],
         [ 0.0354,  0.0468, -0.0306,  ...,  0.2047,  0.1707,  0.1287]]],
       grad_fn=<PermuteBackward>)
