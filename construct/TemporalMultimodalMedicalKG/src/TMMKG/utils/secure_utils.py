import time
import base36
from nanoid import generate

ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def short_id():
    ts = base36.dumps(time.time_ns())[-6:]
    rand = generate(ALPHABET, size=8)
    return (ts + rand).upper()
