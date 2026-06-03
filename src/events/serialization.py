from __future__ import annotations

from typing import Any

from src.data.schema import Job, Operation, OperationOption
from src.events.schema import DynamicEvent


def serialize_operation_option(option: OperationOption) -> dict[str, Any]:
    return {
        "machine_id": int(option.machine_id),
        "processing_time": float(option.processing_time),
    }


def deserialize_operation_option(data: dict[str, Any]) -> OperationOption:
    return OperationOption(
        machine_id=int(data["machine_id"]),
        processing_time=float(data["processing_time"]),
    )


def serialize_operation(operation: Operation) -> dict[str, Any]:
    return {
        "op_global_id": int(operation.op_global_id),
        "job_id": int(operation.job_id),
        "op_index": int(operation.op_index),
        "release_time": float(operation.release_time),
        "due_date": None if operation.due_date is None else float(operation.due_date),
        "options": [serialize_operation_option(option) for option in operation.options],
    }


def deserialize_operation(data: dict[str, Any]) -> Operation:
    return Operation(
        op_global_id=int(data["op_global_id"]),
        job_id=int(data["job_id"]),
        op_index=int(data["op_index"]),
        release_time=float(data.get("release_time", 0.0)),
        due_date=None if data.get("due_date") is None else float(data["due_date"]),
        options=[deserialize_operation_option(option) for option in data.get("options", [])],
    )


def serialize_job(job: Job) -> dict[str, Any]:
    return {
        "job_id": int(job.job_id),
        "release_time": float(job.release_time),
        "due_date": None if job.due_date is None else float(job.due_date),
        "name": job.name,
        "operations": [serialize_operation(operation) for operation in job.operations],
    }


def deserialize_job(data: dict[str, Any]) -> Job:
    return Job(
        job_id=int(data["job_id"]),
        release_time=float(data.get("release_time", 0.0)),
        due_date=None if data.get("due_date") is None else float(data["due_date"]),
        name=data.get("name"),
        operations=[deserialize_operation(operation) for operation in data.get("operations", [])],
    )


def serialize_event_payload(event: DynamicEvent) -> dict[str, Any]:
    payload = dict(event.payload)
    if event.event_type == "compound":
        serialized_subevents = []
        for index, subevent in enumerate(payload.get("subevents", [])):
            if isinstance(subevent, DynamicEvent):
                serialized_subevents.append(serialize_dynamic_event(subevent))
            elif isinstance(subevent, dict):
                serialized_subevents.append(dict(subevent))
            else:
                raise TypeError(f"Unsupported compound subevent at index {index}: {type(subevent)!r}")
        return {
            "subevents": serialized_subevents,
            "profile": payload.get("profile"),
            "epoch_index": payload.get("epoch_index"),
        }
    if event.event_type == "job_arrival":
        job_object = payload.get("job_object")
        return {
            "new_job_id": int(payload["new_job_id"]),
            "template_job_id": None if payload.get("template_job_id") is None else int(payload["template_job_id"]),
            "release_time": float(payload.get("release_time", event.time)),
            "new_job": None if job_object is None else serialize_job(job_object),
        }
    if event.event_type == "machine_breakdown":
        start_time = float(payload.get("start_time", event.time))
        end_time = float(payload.get("end_time", start_time))
        return {
            "machine_id": int(payload["machine_id"]),
            "down_start": start_time,
            "down_end": end_time,
        }
    return payload


def deserialize_event_payload(event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
    if event_type == "compound":
        return {
            "subevents": [
                deserialize_dynamic_event(subevent, fallback_event_id=index)
                for index, subevent in enumerate(payload.get("subevents", []))
            ],
            "profile": payload.get("profile"),
            "epoch_index": payload.get("epoch_index"),
        }
    if event_type == "job_arrival":
        job_data = payload.get("new_job")
        return {
            "new_job_id": int(payload["new_job_id"]),
            "template_job_id": None if payload.get("template_job_id") is None else int(payload["template_job_id"]),
            "release_time": float(payload.get("release_time", 0.0)),
            "job_object": None if job_data is None else deserialize_job(job_data),
        }
    if event_type == "machine_breakdown":
        return {
            "machine_id": int(payload["machine_id"]),
            "start_time": float(payload.get("down_start", payload.get("start_time", 0.0))),
            "end_time": float(payload.get("down_end", payload.get("end_time", 0.0))),
        }
    return dict(payload)


def serialize_dynamic_event(event: DynamicEvent, event_id: str | int | None = None) -> dict[str, Any]:
    return {
        "event_id": event.event_id if event_id is None else event_id,
        "type": event.event_type,
        "time": float(event.time),
        "payload": serialize_event_payload(event),
    }


def deserialize_dynamic_event(data: dict[str, Any], fallback_event_id: int = 0) -> DynamicEvent:
    event_type = data.get("type", data.get("event_type"))
    if event_type is None:
        raise KeyError("Serialized event is missing 'type'.")
    raw_event_id = data.get("event_id", fallback_event_id)
    numeric_event_id = int(raw_event_id) if isinstance(raw_event_id, int) else int(fallback_event_id)
    return DynamicEvent(
        event_id=numeric_event_id,
        time=float(data["time"]),
        event_type=event_type,
        payload=deserialize_event_payload(event_type, data.get("payload", {})),
    )
