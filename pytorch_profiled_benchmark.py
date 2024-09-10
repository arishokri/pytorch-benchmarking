import torch
import torch.profiler

# Simple test function
def run_test():
    for _ in range(100):
        x = torch.rand(10000, 1000).to('cuda')
        y = torch.matmul(x, x.t())
        y = torch.relu(y)

# Profiling the function
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, 
                                        torch.profiler.ProfilerActivity.CUDA],
                            record_shapes=True) as prof:
    run_test()

# Display profiling results
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
