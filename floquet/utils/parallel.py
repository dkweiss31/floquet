from collections.abc import Iterable

import pathos


def parallel_map(num_cpus: int, func: callable, parameters: Iterable) -> map:
    if num_cpus == 1:
        return map(func, parameters)

    with pathos.pools.ProcessPool(nodes=num_cpus) as pool:
        return pool.map(func, parameters)
