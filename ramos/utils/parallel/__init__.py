from itertools import chain
from multiprocessing import Process, Queue
from operator import itemgetter
import os


__all__ = ['parmap']


def split(lst, n):
    k, m = divmod(len(lst), n)
    return [lst[i*k+min(i,m):(i+1)*k+min(i+1,m)] for i in range(n)]


def delegate(pid, q, target, chunk, args, reduction):
    result = [target(*c, *args) for c in chunk]

    if reduction:
        q.put((pid, reduction(result)))
    else:
        q.put((pid, result))


def parmap(target, varying, constant=(), reduction=None, ncpus=None, unwrap=True):
    if not ncpus:
        ncpus = os.cpu_count()
    if not unwrap:
        varying = [(v,) for v in varying]
    chunks = split(varying, ncpus)
    q = Queue()

    processes = []
    for pid, chunk in enumerate(chunks):
        p = Process(
            target=delegate,
            args=(pid, q, target, chunk, constant, reduction),
        )
        p.start()
        processes.append((pid, p))

    result = [q.get() for _ in range(ncpus)]

    for pid, p in processes:
        p.join()

    result = [v for _, v in sorted(result, key=itemgetter(0))]

    if reduction:
        return reduction(result)
    return list(chain.from_iterable(result))
