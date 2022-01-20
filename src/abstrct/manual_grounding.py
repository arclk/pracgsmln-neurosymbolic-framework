import torch

with open('neoplasm25_train.db', 'r') as file:
    data = file.read()

f1 = []
f2 = []
f3 = []
f4 = []

for line in data.split('\n'):
    if 'Link' in line:
        temp = line.split('(')[1].split(',')
        d1 = temp[0][2:]
        d2 = temp[1].strip()[2:-1]
        f2.append({'id1': f'ID{d1}', '$f1': f'F{d1}', 'id2': f'ID{d2}', '$f2': f'F{d2}'})
        f3.append({'id1': f'ID{d1}', 'id2': f'ID{d2}'})
        f4.append({'id1': f'ID{d1}', 'id2': f'ID{d2}'})
    elif 'Type' in line:
        temp = line.split('(')[1].split(',')
        d1 = temp[0][2:]
        d2 = temp[1].strip()[2:-1]
        f1.append({'id1': f'ID{d1}', '$f1': f'F{d1}', 'type': 'Claim'})
        f1.append({'id1': f'ID{d1}', '$f1': f'F{d1}', 'type': 'Premise'})

formulas = [f1, f2, f3, f4]

# torch.save(formulas, 'neoplasm25_ground_train_new.pt')
print(len(f1))
print(len(f2))
print(len(f3))
print(len(f4))
