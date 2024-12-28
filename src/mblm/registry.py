__copyright__ = """MIT License

Copyright (c) 2024 - IBM Research

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE."""

from typing import Any, Callable, Generic, TypeVar

_T = TypeVar("_T")


class DictRegistry(Generic[_T]):
    __registry: dict[str, type[_T]]

    def __init__(self, name: str):
        self.__name = name
        self.__registry = {}

    def register(self, id: str, klass: type[_T]) -> None:
        if id in self.__registry:
            raise ValueError(f"{self.__name} already has {id} ({klass})")
        self.__registry.setdefault(id, klass)

    def __contains__(self, item: Any) -> bool:
        return item in self.__registry

    def get(self, id: str) -> type[_T]:
        if (klass := self.__registry.get(id, None)) is None:
            raise ValueError(f"{id} not found in {self.__name}")
        return klass

    def __str__(self) -> str:
        return f"{self.__registry}"


class SetRegistry(Generic[_T]):
    __registry: set[type[_T]]

    def __init__(self, name: str):
        self.__name = name
        self.__registry = set()

    def register(self, klass: type[_T]) -> None:
        if klass in self.__registry:
            raise ValueError(f"{self.__name} already has {klass}")
        self.__registry.add(klass)

    def __contains__(self, item: Any) -> bool:
        return item in self.__registry

    def try_parse(self, data: Any, parse_func: Callable[[type[_T], Any], _T]):
        for klass in self.__registry:
            try:
                return parse_func(klass, data)
            except Exception:
                pass

        raise ValueError(f"Coult not parse data to any of {self.__registry}")

    def __str__(self) -> str:
        return f"{self.__registry}"
