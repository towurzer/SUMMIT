import torch
import pandas as pd



torch.mean(torch.zeros(4))
print(torch.zeros(4))
print(torch.mean(torch.zeros(4)))
print(type(pd.DataFrame()))

print(f'has cuda? {torch.cuda.is_available()}')