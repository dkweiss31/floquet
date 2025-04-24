from tqdm import tqdm
import pathos.pools
from typing import Iterable

def parallel_map(num_cpus: int, func: callable, parameters: Iterable) -> map:
    # Convert parameters to list to get length for progress bar
    param_list = list(parameters)
    
    if num_cpus == 1:
        return list(tqdm(map(func, param_list), total=len(param_list), desc=f"Processing {func.__name__}"))

    with pathos.pools.ProcessPool(nodes=num_cpus) as pool:
        return list(tqdm(pool.imap(func, param_list), total=len(param_list), desc=f"Processing {func.__name__}"))