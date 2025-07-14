from collections.abc import Iterable
from tqdm import tqdm
import pathos


def parallel_map(num_cpus: int, func: callable, parameters: Iterable) -> map:
    param_list = list(parameters)

    if num_cpus == 1:
        return map(func, parameters)

    with pathos.pools.ProcessPool(nodes=num_cpus) as pool:
        return list(tqdm(pool.imap(func, param_list), total=len(param_list), desc=f"Processing {func.__name__}"))