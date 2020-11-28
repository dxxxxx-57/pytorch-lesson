import torch

# ====================================== retain_graph ==============================================
flag = False
if flag:
    w = torch.tensor([1.], requires_grad=True)
    x = torch.tensor([2.], requires_grad=True)

    a = torch.add(w, x)
    b = torch.add(x, 1)
    y = torch.mul(a, b)

    y.backward(retain_graph=True)  # 若想要二次反向传播，需要保存计算图
    print(w.grad)
    y.backward()

# ====================================== grad_tensors ==============================================
flag = False
if flag:
    w = torch.tensor([1.], requires_grad=True)
    x = torch.tensor([2.], requires_grad=True)

    a = torch.add(w, x)
    b = torch.add(w, 1)

    y0 = torch.mul(a, b)  # y0 = (x+w) * (w+1)
    y1 = torch.add(a, b)  # y1 = (x+w) + (w+1)

    loss = torch.cat([y0, y1], dim=0)
    grad_tensors = torch.tensor([1., 2.])

    loss.backward(gradient=grad_tensors)  # 多梯度权重
    print(w.grad)

# ====================================== autograd.gard ==============================================
flag = False
if flag:
    x = torch.tensor([3.], requires_grad=True)
    y = torch.pow(x, 2)  # y = x**2

    grad_1 = torch.autograd.grad(y, x, create_graph=True)  # creat_graph创建倒数计算图，用于高阶求导
    print(grad_1)

    grad_2 = torch.autograd.grad(grad_1[0], x)
    print(grad_2)

# ====================================== 梯度不自动清零 ==============================================
flag = False
if flag:
    w = torch.tensor([1.], requires_grad=True)
    x = torch.tensor([2.], requires_grad=True)

    for i in range(4):
        a = torch.add(w, x)
        b = torch.add(w, 1)
        y = torch.mul(a, b)

        y.backward()
        print(w.grad)

        w.grad.zero_()  # 梯度清零

# ====================================== 依赖于叶子节点的点，requires_grad默认为True ==============================================
flag = False
if flag:
    w = torch.tensor([1.], requires_grad=True)
    x = torch.tensor([2.], requires_grad=True)

    a = torch.add(w, x)
    b = torch.add(w, 1)
    y = torch.mul(a, b)

    print(a.requires_grad, b.requires_grad, y.requires_grad)

# ====================================== 叶子节点不可执行in place操作(即原位操作)==============================================
flag = False
if flag:
    a = torch.ones((1,))
    print(id(a), a)

    a = a + torch.ones((1,))  # 改变了a存储的地址
    print(id(a), a)

    a += torch.ones((1,))  # 只改变了a中存储的值
    print(id(a), a)

flag = True
if flag:
    w = torch.tensor([1.], requires_grad=True)
    x = torch.tensor([2.], requires_grad=True)

    a = torch.add(w, x)
    b = torch.add(w, 1)
    y = torch.mul(a, b)

    w.add_(1)  # 带下划线的意思就是原位操作
    # 在反向更新之前对W进行原位操作会报错
    y.backward()
