"""One updating progress line, shared by the long-running measurement scripts.

On a terminal it rewrites the line in place; when stdout is redirected to a log (nohup/tee) it prints
spaced discrete lines instead, because carriage returns make a log unreadable. It is throttled so it
floods neither one: at most once per `mininterval` (0.5s on a tty, 15s to a log) plus always on the
final update. Drive it from the PARENT loop -- worker processes must not contend for stdout.

    from progress import bar
    t0 = time.time()
    for i, _ in enumerate(as_completed(futs), 1):
        ...
        bar(i, len(futs), t0, fails=failed, label="validate ")
"""
import sys
import time

_last = [0.0]


def hms(sec):
    sec = int(max(0, sec))
    return f"{sec // 3600:d}:{(sec % 3600) // 60:02d}:{sec % 60:02d}"


def bar(n, total, t0, fails=0, label="", width=28, tty=None, mininterval=None):
    """Render n/total with rate, elapsed and ETA. `t0` is the start time (time.time()). Emits at most
    once per `mininterval` and always when n reaches total; the throttle is shared, so run one bar at
    a time. `label` prefixes the line (e.g. the current shape)."""
    now = time.time()
    if tty is None:
        tty = sys.stdout.isatty()
    if mininterval is None:
        mininterval = 0.5 if tty else 15.0
    if n < total and now - _last[0] < mininterval:
        return
    _last[0] = now
    frac = n / total if total else 1.0
    el = now - t0
    rate = n / el if el > 0 else 0.0
    filled = int(width * frac)
    msg = (f"{label}[{'#' * filled}{'-' * (width - filled)}] {n}/{total} ({100 * frac:5.1f}%)  "
           f"{rate * 60:6.1f}/min  elapsed {hms(el)}  eta {hms((total - n) / rate if rate else 0)}"
           + (f"  ({fails} failed)" if fails else ""))
    if tty:
        print("\r" + msg + "  ", end="\n" if n >= total else "", flush=True)
    else:
        print(msg, flush=True)
