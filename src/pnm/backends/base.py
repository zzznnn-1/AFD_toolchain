from abc import ABC, abstractmethod
from pnm.protocol import Command, Completion


class Backend(ABC):
    @abstractmethod
    def execute(self, command: Command, payload: dict) -> Completion: ...

    @abstractmethod
    def close(self): ...
