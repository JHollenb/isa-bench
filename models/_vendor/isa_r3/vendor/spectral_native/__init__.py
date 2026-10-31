"""Spectral-native research models; FieldLM selects the current research base."""

__all__ = ["FieldLM"]


def __getattr__(name):
    if name == "FieldLM":
        from .base_fieldlm import FieldLM
        return FieldLM
    raise AttributeError(name)
