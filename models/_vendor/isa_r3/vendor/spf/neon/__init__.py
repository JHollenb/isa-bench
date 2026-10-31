"""NEON-bound ctypes paths (placeholder).

NEON kernels are exercised through ``spf.scalar.<type>`` today — the C
dispatcher selects the NEON variant at libspf32 load time based on the
runtime CPU features. This package is reserved for arch-aware Python
entry points that need to bypass the dispatcher (e.g. force-NEON
variants for parity testing or benchmarking).

Currently empty; populate as NEON-only Python wrappers land.
"""
