import torch
import torch.nn as nn

x = torch.Tensor([[1,2,3],[2,3,4],[3,4,5]]).requires_grad_()
m = nn.Linear(3, 10)
out = m(x)
print(x.requires_grad, out.requires_grad)

# flat_x = x.reshape(-1)
# flat_p = out.reshape(-1)
# print(flat_x.requires_grad, flat_p.requires_grad)

grad = torch.autograd.grad(out, x, grad_outputs=torch.ones_like(out), create_graph=True)[0]
print(grad)

print(torch.mean(grad))