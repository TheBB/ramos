from itertools import chain
from multiprocessing import Process, Queue
from operator import itemgetter
import os


__all__ = ['parmap']


def split(lst, n):
    """Split a list `lst` into n blocks, as equal as possible in size."""
    k, m = divmod(len(lst), n)
    return [lst[i*k+min(i,m):(i+1)*k+min(i+1,m)] for i in range(n)]


def delegate(pid, q, target, chunk, args, reduction):
    """Worker function for use in parmap.

    - `pid`: The ID of this worker process
    - `q`: Queue to which result will be sent
    - `target`: The worker function that will actually do the work
    - `chunk`: List of argument tuples
    - `args`: Tuple of constant arguments
    - `reduction`: Function to reduce the result, or None

    Will put the tuple (pid, <result>) to the queue `q`.
    """
    result = []
    for c in chunk:
        cur_args = list(c) + list(args)
        result.append(target(*cur_args))

    if reduction:
        q.put((pid, reduction(result)))
    else:
        q.put((pid, result))


def parmap(target, varying, constant=(), reduction=None, ncpus=None, unwrap=True):
    """Parallel map

    - `target`: a function to be called on all inputs
    - `varying`: a list of tuples of arguments to pass to the target function
    - `constant`: a tuple of arguments to pass to the target function
    - `reduction`: optional function to reduce the output
    - `ncpus`: optionally, number of parallel workers to use
    - `unwrap`: if false, treat `varying` as a list of single arguments, rather
      than as a list of argument tuples

    The target function must be written so that the variable arguments come
    before the constant ones.
    """
    if not ncpus:
        ncpus = os.cpu_count()
    if not unwrap:
        varying = [(v,) for v in varying]

    chunks = split(varying, ncpus)
    q = Queue()                 # Return values will be read from the queue

    # Start a process for each worker, and run the delegate function
    processes = []
    for pid, chunk in enumerate(chunks):
        p = Process(
            target=delegate,
            args=(pid, q, target, chunk, constant, reduction),
        )
        p.start()
        processes.append((pid, p))

    # Grab the results from each worker and shut them down
    result = [q.get() for _ in range(ncpus)]

    for pid, p in processes:
        p.join()

    # Sort results by pid
    result = [v for _, v in sorted(result, key=itemgetter(0))]

    # Return either reduction or flattened list
    if reduction:
        return reduction(result)
    return list(chain.from_iterable(result))
