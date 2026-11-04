from abc import ABC, abstractmethod
from pnm.protocol import Command, Completion


class Backend(ABC):
    def describe(self):
        return {'backend':type(self).__name__,'protocol_version':1}

    @abstractmethod
    def execute(self, command: Command, payload: dict) -> Completion: ...

    @abstractmethod
    def close(self): ...
