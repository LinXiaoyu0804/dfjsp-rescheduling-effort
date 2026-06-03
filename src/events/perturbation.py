from __future__ import annotations

from copy import deepcopy

from src.data.schema import ProblemInstance
from src.events.schema import DynamicEvent


def apply_processing_time_perturbation(instance: ProblemInstance, event: DynamicEvent) -> ProblemInstance:
    if event.event_type != "processing_time_perturbation":
        return instance

    new_instance = deepcopy(instance)
    op_id = int(event.payload["op_global_id"])
    machine_id = int(event.payload["machine_id"])
    new_pt = float(event.payload["new_processing_time"])

    op = new_instance.get_operation(op_id)
    for option in op.options:
        if option.machine_id == machine_id:
            option.processing_time = new_pt
            return new_instance

    raise KeyError(f"Operation {op_id} has no option on machine {machine_id}")
