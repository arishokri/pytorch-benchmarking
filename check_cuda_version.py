import torch
print(f'Is CDUA available? {torch.cuda.is_available()}')
print(f'CUDA version available to torch: {torch.version.cuda}')
print(f'First GPU module: {torch.cuda.get_device_name(0)}')
print(f'Total GPU count: {torch.cuda.device_count()}')

x = torch.rand(3, 3)
x = x.to('cuda')
print(f'Operation ran successfully without errors and returned: {x}')
