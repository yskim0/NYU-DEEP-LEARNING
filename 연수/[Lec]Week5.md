# NYU Deep Learning - Week 5




```py
In [1]: import torch

In [2]: from torch import nn

In [3]: conv = nn.Conv2d?

In [4]: conv = nn.Conv2d

In [5]: conv = nn.Conv2d(2, 16, 3)

In [6]: conv
Out[6]: Conv2d(2, 16, kernel_size=(3, 3), stride=(1, 1))

In [7]: conv = nn.Conv1d(2, 16, 3)

In [8]: conv
Out[8]: Conv1d(2, 16, kernel_size=(3,), stride=(1,))

In [9]: conv.weight.size()
Out[9]: torch.Size([16, 2, 3])

In [10]: x = torch.randn(1, 2, 64)

In [11]: conv.bias.size()
Out[11]: torch.Size([16])

In [12]: conv(x).size()
Out[12]: torch.Size([1, 16, 62])

In [13]: conv = nn.Conv1d(2,16,5)

In [14]: conv(x).size()
Out[14]: torch.Size([1, 16, 60])

In [15]: # conv = nn.Conv2d(20, 16,

In [16]: x = torch.rand(1, 20, 64, 128)

In [17]: x.size()
Out[17]: torch.Size([1, 20, 64, 128])

In [18]: conv = nn.Conv2d(20, 16, (3,5))

In [19]: conv.weight.size()
Out[19]: torch.Size([16, 20, 3, 5])

In [20]: conv(x).size()
Out[20]: torch.Size([1, 16, 62, 124])

In [21]: conv = nn.Conv2d(20, 16, (3,5), 1, (1, 2))

In [22]: conv(x).size()
Out[22]: torch.Size([1, 16, 64, 128])
```
