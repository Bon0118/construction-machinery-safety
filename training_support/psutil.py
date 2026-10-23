"""Small Windows-only psutil compatibility used by the local YOLOv5 trainer.

The referenced YOLOv5 checkout only needs ``virtual_memory()`` while building
its image cache.  Keeping this shim in the training subprocess avoids changing
the platform runtime or copying binary packages between Conda environments.
"""

from __future__ import annotations

import ctypes
from collections import namedtuple


VirtualMemory = namedtuple("VirtualMemory", "total available percent used free")


class _MemoryStatus(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def virtual_memory() -> VirtualMemory:
    status = _MemoryStatus()
    status.dwLength = ctypes.sizeof(_MemoryStatus)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise ctypes.WinError()
    total = int(status.ullTotalPhys)
    available = int(status.ullAvailPhys)
    used = max(0, total - available)
    percent = round(used * 100.0 / total, 1) if total else 0.0
    return VirtualMemory(total, available, percent, used, available)
