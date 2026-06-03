from __future__ import annotations

from src.data.schema import Job, Machine, Operation, OperationOption, ProblemInstance
from src.events.generator import generate_dynamic_events
from src.events.schema import DynamicEvent
from src.events.serialization import deserialize_dynamic_event, serialize_dynamic_event
from src.scheduling.incumbent import IncumbentSchedule, MachineCalendar, OperationSchedule
from src.scheduling.rho import compute_rho_descriptors
from src.scheduling.state_builder import StateSnapshot


def _toy_instance() -> ProblemInstance:
    return ProblemInstance(
        family="fjsp",
        machines=[Machine(0), Machine(1)],
        jobs=[
            Job(
                job_id=0,
                operations=[
                    Operation(
                        op_global_id=0,
                        job_id=0,
                        op_index=0,
                        options=[OperationOption(0, 3.0), OperationOption(1, 5.0)],
                    ),
                    Operation(
                        op_global_id=1,
                        job_id=0,
                        op_index=1,
                        options=[OperationOption(0, 4.0), OperationOption(1, 7.0)],
                    ),
                ],
            )
        ],
    )


def test_compound_event_serialization_roundtrip() -> None:
    arrival = DynamicEvent(
        event_id=1,
        time=10.0,
        event_type="job_arrival",
        payload={"new_job_id": 2, "template_job_id": None, "release_time": 10.0, "job_object": _toy_instance().jobs[0]},
    )
    breakdown = DynamicEvent(
        event_id=2,
        time=10.0,
        event_type="machine_breakdown",
        payload={"machine_id": 1, "start_time": 10.0, "end_time": 15.0},
    )
    compound = DynamicEvent(
        event_id=0,
        time=10.0,
        event_type="compound",
        payload={"subevents": [arrival, breakdown], "profile": "R5", "epoch_index": 0},
    )

    serialized = serialize_dynamic_event(compound, event_id="cmp_0")
    restored = deserialize_dynamic_event(serialized)

    assert restored.event_type == "compound"
    assert len(restored.payload["subevents"]) == 2
    assert restored.payload["subevents"][0].event_type == "job_arrival"
    assert restored.payload["subevents"][1].payload["machine_id"] == 1


def test_rho_descriptors_use_pending_window_assigned_processing_mass() -> None:
    instance = _toy_instance()
    incumbent = IncumbentSchedule(
        operations={
            0: OperationSchedule(0, 0, 0, 1, 0.0, 5.0, status="active"),
            1: OperationSchedule(1, 0, 1, 1, 6.0, 13.0, status="unstarted"),
        },
        machine_calendars={0: MachineCalendar(0), 1: MachineCalendar(1)},
        current_time=2.0,
    )
    snapshot = StateSnapshot(
        current_time=2.0,
        completed_op_ids=[],
        active_op_ids=[0],
        unfinished_op_ids=[0, 1],
        window_op_ids=[0, 1],
        directly_impacted_op_ids=[0, 1],
        affected_machine_ids=[1],
    )

    rho = compute_rho_descriptors(
        instance=instance,
        incumbent=incumbent,
        snapshot=snapshot,
        makespan_before=14.0,
    )

    assert rho["rho_window_work_mass"] == 7.0
    assert rho["rho_footprint_work_mass"] == 7.0
    assert rho["rho_t"] == 0.5
    assert rho["rho_pending_window_ops"] == 1


def test_rho_profile_generation_is_seed_stable_and_compound() -> None:
    instance = _toy_instance()
    cfg = {"rho_intensity_profile": {"enabled": True, "label": "R5"}}

    first = generate_dynamic_events(instance, cfg, seed=3)
    second = generate_dynamic_events(instance, cfg, seed=3)

    assert [(event.event_type, event.time) for event in first] == [(event.event_type, event.time) for event in second]
    assert first
    assert all(event.event_type == "compound" for event in first)
    assert all(len(event.payload["subevents"]) == 8 for event in first)
