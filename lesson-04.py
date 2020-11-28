import torch

# ====================================== type check ==============================================
flag = False
if flag:
    a = torch.randn(2, 3)
    # .shape 为成员属性， .size()为成员方法
    print(a, a.shape, a.size(), a.type(), isinstance(a, torch.FloatTensor))

    data = torch.tensor([1.])
    print(isinstance(data, torch.cuda.DoubleTensor))
    data = data.cuda()  # x.cuda()会返回一个GPU上的引用
    print(isinstance(data, torch.cuda.FloatTensor))

# ====================================== 标量 ==============================================
# loss最后是一个标量
flag = False
if flag:
    print(torch.tensor(1.3))  # (1.3)是0维
    print(torch.tensor([1.3]))  # [1.3]是一维，长度为1的Tensor

    print(torch.full((2, 3), 7.0))
    print(torch.full([2, 3], 7.0))

# ====================================== 等分，等差 ==============================================
flag = False
if flag:
    print(torch.arange(0, 10, 4))
    print(torch.linspace(0, 10, 11))
    print(torch.logspace(0, -1, 10))

# ====================================== 索引 ==============================================
flag = False
if flag:
    a = torch.rand(4, 3, 28, 28)
    # print(a[0].shape, a[0, 0].shape, a[0, 0, 2, 4])  # 最后为一个标量
    # print(a[:2].shape, a[:2, :1, :, :].shape, a[:2, 1:, :, :].shape, a[:2, -1:, :, :].shape)  # 左开右闭
    # #  隔行采样
    # print(a[:, :, 0:28:2, 0:28:2].shape, a[:, :, ::2, ::2].shape)

    # 第二个参数必须为tensor, 不能直接给list
    print(a.index_select(0, torch.tensor([0, 2])).shape)
    print(a.index_select(1, torch.tensor([1, 2])).shape)
    print(a.index_select(2, torch.arange(8)).shape)

    print(a[...].shape, a[0, ...].shape, a[:, 1, ...].shape, a[..., :2].shape)

    x = torch.randn(3, 4)
    mask = x.ge(0.5)  # 打平x
    print(mask, torch.masked_select(x, mask))

    src = torch.tensor([[4, 3, 5], [6, 7, 8]])
    print(torch.take(src, torch.tensor([0, 2, 4])))

# ====================================== squeeze ==============================================
flag = True
if flag:
    a = torch.rand(4, 1, 28, 28)
    # 并不改变数据本身 插入范围[-5,5),最好不用负数插入，超出范围进行插入会报错
    print(a.shape, a.unsqueeze(0).shape, a.unsqueeze(-1).shape)  # 增加了一个概念，在对应索引位置插入

    # 例子
    print(a)