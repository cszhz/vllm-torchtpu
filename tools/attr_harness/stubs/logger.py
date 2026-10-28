"""Stub of vllm_torchtpu.logger (the real one pulls in vllm)."""

import logging


class _Log:
    def __init__(self, name):
        self._l = logging.getLogger(name)

    def warning_once(self, msg, *args):
        self._l.warning(msg, *args)

    def __getattr__(self, item):
        return getattr(self._l, item)


def init_logger(name):
    return _Log(name)
