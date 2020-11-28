import torch

# ======================================= example 1 =======================================
# 用cat进行拼接
flag = False
if flag:
    t = torch.ones((2, 3))

    t_0 = torch.cat([t, t], dim=0)  # 在第0维进行拼接
    t_1 = torch.cat([t, t], dim=1)  # 在第1维进行拼接

    print("t_0:{} shape:{}\n t_1:{} shape:{}".format(t_0, t_0.shape, t_1, t_1.shape))

# ======================================= example 2 ====================================
# #  用stack进行拼接===
flag = False
if flag:
    a = torch.tensor([[1, 2, 3], [4, 5, 6]])
    b = torch.tensor([[10, 20, 30], [40, 50, 60]])
    c = torch.tensor([[100, 200, 300], [400, 500, 600]])
    print(torch.stack([a, b, c], dim=0))  # 也就是最终生成的结果位于第i维
    print(torch.stack([a, b, c], dim=1))  # 将每个tensor的第i行按行连接组成一个新的2维tensor，再将这些新tensor按照dim=0的方式连接。
    print(torch.stack([a, b, c], dim=2))  # 将每个tensor的第i行转置后按列连接组成一个新的2维tensor，再将这些新tesnor按照dim=0的方式连接

# # ======================================= example 3 =======================================
# #  用chunk进行切片
# flag = False
# if flag:
#     a = torch.ones((2, 7))
#     list_of_tensor = torch.chunk(a, chunks=3, dim=1)
#     #  enumerate(sequence)sequence:一个序列、迭代器或其他支持迭代对象。函数接受一个集合（例如元组），并将其作为枚举对象返回。
#     for idx, t in enumerate(list_of_tensor):
#         print("第{}个张量：{},shape is {}".format(idx+1, t, t.shape)

# ======================================= example 4 =======================================
#  用split进行切片
flag = False
if flag:
    t = torch.ones((2, 5))
    list_of_tensor = torch.split(t, [2, 1, 2], dim=1)  # 2+1+2=5,必须等于5不然会报错
    for idx, t in enumerate(list_of_tensor):
        print("第{}个张量：{},shape is {}".format(idx + 1, t, t.shape))

# ======================================= example 5 =======================================
#  用index_select进行索引
flag = False
if flag:
    t = torch.randint(0, 9, size=(3, 3))  # 张量形状为3*3的均匀分布
    idx = torch.tensor([0, 2], dtype=torch.long)  # idx的索引类型为long类型
    t_select = torch.index_select(t, dim=0, index=idx)
    print("t:\n{}\nt_select:\n{}".format(t, t_select))

# ======================================= example 6 =======================================
#  用maskded_select进行索引
flag = False
if flag:
    t = torch.randint(0, 9, size=(3, 3))
    mask = t.ge(5)  # ge means greater than or equal
    t_select = torch.masked_select(t, mask)
    print("t:\n{}\nmask:\n{}\nt_select:\n{}".format(t, mask, t_select))

# ======================================= example 7 =======================================
#  用reshape进行张量变换,共享内存
flag = False
if flag:
    t = torch.randperm(8)  # 随机产生1-8的数
    t_reshape = torch.reshape(t, (-1, 2, 2))  # -1的意思是不用这维，由其他维度决定：8/2/2=2
    print("t:{}\nt_reshape:{}\n".format(t, t_reshape))

    t[0] = 1024
    print("t:{}\nt_reshape:{}\n".format(t, t_reshape))
    print("t.data 内存地址:{}".format(id(t.data)))
    print("t_reshape.data 内存地址{}".format(id(t_reshape.data)))

# ======================================= example 8 =======================================
#  用transpose进行变换(交换)
flag = False
if flag:
    t = torch.rand((2, 3, 4))  # 均匀分布
    print(t)
    t_transpose = torch.transpose(t, dim0=1, dim1=2)
    print("t shape:{}\nt_transpose shape:{}\n".format(t.shape, t_transpose.shape))

# ======================================= example 9 =======================================
#  使用squeeze进行压缩
flag = False
if flag:
    t = torch.rand((1, 2, 3, 1))
    t_sq = torch.squeeze(t)
    t_0 = torch.squeeze(t, dim=0)
    t_1 = torch.squeeze(t, dim=1)
    print(t.shape)
    print(t_sq.shape)  # 长度为1的会被自然压缩
    print(t_0.shape)   # 指定的长度为1的会被压缩
    print(t_1.shape)   # 指定的长度不为1的不会被压缩

# ======================================= example 10 =======================================
# 张量计算
flag = True
if flag:
    t_0 = torch.randn((3, 3))  # 标准正态分布
    t_1 = torch.ones_like(t_0)
    t_add = torch.add(t_0, 10, t_1)
    print("t_0:\n{}\nt_1:\n{}\nt_add:\n{}\n".format(t_0, t_1, t_add))
