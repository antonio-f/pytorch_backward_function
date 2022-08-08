import torch

print("\n---First example---")
x = torch.ones(2, 2, requires_grad=True)
y = x + 2
z = y * y * 3
out = z.mean()

out.backward()

print("x.grad:", x.grad)

# # ----- ----- ----- -----
# # alternative: comment previous backward() and x.grad references
# print("x.grad alternative:", torch.autograd.grad(outputs=out, inputs=x))
# # ----- ----- ----- -----




# ----- ----- ----- -----
# Neural network example 
# ----- ----- ----- -----
print("\n---Neural network example---")

x = torch.ones(8)  # input tensor
y = torch.zeros(10)  # expected output
W = torch.randn(8, 10, requires_grad=True) # weights
b = torch.randn(10, requires_grad=True) # bias vector
z = torch.matmul(x, W)+b # output
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

loss.backward()
# print(W.grad) #OK
print("b.grad:", b.grad) #OK
print("x.grad:",x.grad)
print("y.grad:",y.grad)
# print(z.grad) # WARNING
# print(loss.grad) # WARNING




# ----- ----- ----- -----
# Vector-Jacobian example #1 
# ----- ----- ----- -----
print("\n---Vector-Jacobian example #1---")

x = torch.rand(3, requires_grad=True)
y = x + 2
# y.backward() <---
# RuntimeError: grad can be implicitly 
# created only for scalar outputs
# try ---> y.backward(v) where v is any tensor of length 3
# v = torch.rand(3)
v = torch.tensor([1.,2,3])
y.backward(v)
print("x.grad:", x.grad)

# # ----- ----- ----- -----
# # alternative: comment previous backward() and x.grad references
# print("x.grad alternative:",torch.autograd.grad(outputs=y, inputs=x, grad_outputs=v))
# # ----- ----- ----- -----




# ----- ----- ----- -----
# Vector-Jacobian example #2 
# ----- ----- ----- -----
print("\n---Vector-Jacobian example #2---")

x = torch.tensor([1., 2], requires_grad=True)
print('x:', x)

y = torch.empty(3)
y[0] = x[0]**2
y[1] = x[0]**2 + 5*x[1]**2
y[2] = 3*x[1]
print('y:', y)

v = torch.tensor([1., 1, 1,])
y.backward(v) 
print('x.grad:', x.grad)




# ----- ----- ----- -----
# Vector-Jacobian example #2 
# ----- ----- ----- -----
print("\n---General case example---")

x = torch.tensor([[1.,2,3],[4,5,6]], requires_grad=True)
y = torch.log(x)
# y is a 2x2 tensor obtained by taking logarithm entry-wise
v = torch.tensor([[3.,2,0],[4,0,1]], requires_grad=True)
# v is not a 1D tensor!
y.backward(v)
print("x.grad:", x.grad) # returns dl/dx, as evaluated by "matrix-Jacobian" product v * dy/dx
# therefore we can interpret v as a matrix dl/dy
# for which the chain rule expression dl/dx = dl/dy * dy/dx holds.