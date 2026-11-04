"""Deterministic fault injection around a real backend, for failure-path validation."""
from dataclasses import replace
from .base import Backend
from pnm.protocol import Op, Completion


class FaultBackend(Backend):
    def __init__(self, delegate):
        self.delegate=delegate
        self.fault=None
        self.last_completion=None

    def arm(self, fault):
        if fault not in ('timeout_before','timeout_after','duplicate_completion','transfer_error','zero_output'):
            raise ValueError(fault)
        self.fault=fault

    def execute(self,command,payload):
        fault=self.fault if command.op==Op.APPEND else None
        if fault: self.fault=None
        if fault=='timeout_before': raise TimeoutError('injected: no completion before dispatch')
        if fault=='transfer_error':
            return Completion(command.command_id,'FAIL',error_code='TRANSFER_ERROR',error_message='injected link failure')
        result=self.delegate.execute(command,payload)
        if fault=='zero_output' and result.output is not None:
            result.output=result.output*0
        if fault=='timeout_after': raise TimeoutError('injected: device committed but completion was lost')
        if fault=='duplicate_completion':
            if self.last_completion is None: raise RuntimeError('need a preceding completion')
            return replace(self.last_completion)
        self.last_completion=result
        return result

    def close(self): self.delegate.close()

    def describe(self): return self.delegate.describe()
