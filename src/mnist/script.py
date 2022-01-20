import torch

f1 = []

for n in range(10):
    for i in range(2000):
        f1.append({'n': f'{n}', '$fa': f'F{i}', 'a': f'ID{i}'})

formulas = [f1]
torch.save(formulas, 'mnist_ground_train_simple.pt')
print(len(f1))
