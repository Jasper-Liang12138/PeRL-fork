import sys
try:
    import lzma
except ModuleNotFoundError:
    from backports import lzma
    sys.modules["lzma"] = lzma

from perl.train import train
from perl.utils.utils import parse_args_to_config

if __name__ == "__main__":
    config = parse_args_to_config()
    train(config)