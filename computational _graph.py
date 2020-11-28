import torch

w = torch.tensor([1., ], requires_grad=True)
x = torch.tensor([2., ], requires_grad=True)

a = torch.add(w, x)
a.retain_grad()  # retained_grad()可以保存梯度
b = torch.add(w, 1)
y = torch.mul(a, b) 

y.backward()
print(w.grad)

# 查是否是叶子节点
print(w.is_leaf, x.is_leaf, a.is_leaf, b.is_leaf, y.is_leaf)
# 查看梯度
print(w.grad, x.grad, a.grad, b.grad, y.grad)
# 查看 graf_fn
print((w.grad_fn, x.grad_fn, a.grad_fn, b.grad_fn, y.grad_fn))