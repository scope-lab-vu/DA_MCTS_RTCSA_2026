"""Seeded CPU load injector for the contention study.

Usage:
    python tools/cpu_load.py CORES MODE [SEED]

    CORES  comma-separated core ids, e.g. 2,3
    MODE   steady  -> burn the cores continuously until killed
           bursty  -> random ON/OFF periods (0.5-3.0 s each), reproducible from SEED
    SEED   integer; the same seed gives the same ON/OFF schedule

Every ON/OFF change is printed with a Unix timestamp so it can be matched to the
per-step wall_time column in results/timing/*.csv. Stop it with SIGTERM/Ctrl-C;
the worker processes are daemons and exit with the parent.
"""
import multiprocessing as mp
import os
import random
import signal
import sys
import time


def burn(core, stop_evt):
    signal.signal(signal.SIGTERM, signal.SIG_DFL)   # workers just die on SIGTERM
    try:
        os.sched_setaffinity(0, {core})   # pin this worker to one core
    except OSError:
        pass
    while not stop_evt.is_set():
        x = 0
        for _ in range(100000):           # pure busy work
            x += 1


def start_workers(cores, stop_evt):
    procs = [mp.Process(target=burn, args=(c, stop_evt), daemon=True) for c in cores]
    for p in procs:
        p.start()
    return procs


ACTIVE = []   # worker processes currently running


def _shutdown(*_):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)   # ignore repeated signals while cleaning up
    for p in ACTIVE:
        if p.is_alive():
            p.terminate()
    for p in ACTIVE:
        p.join(timeout=2)
    os._exit(0)


def main():
    cores = [int(c) for c in sys.argv[1].split(",")]
    mode = sys.argv[2] if len(sys.argv) > 2 else "steady"
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)
    rng = random.Random(seed)

    if mode == "steady":
        stop = mp.Event()
        ACTIVE[:] = start_workers(cores, stop)
        print(f"{time.time():.3f} ON steady cores={cores}", flush=True)
        while True:
            time.sleep(1)
    elif mode == "bursty":
        while True:
            stop = mp.Event()
            procs = start_workers(cores, stop)
            ACTIVE[:] = procs
            on = rng.uniform(0.5, 3.0)
            print(f"{time.time():.3f} ON  {on:.2f}s cores={cores}", flush=True)
            time.sleep(on)
            stop.set()
            for p in procs:
                p.join()
            off = rng.uniform(0.5, 3.0)
            print(f"{time.time():.3f} OFF {off:.2f}s", flush=True)
            time.sleep(off)
    else:
        raise SystemExit("MODE must be steady or bursty")


if __name__ == "__main__":
    main()
