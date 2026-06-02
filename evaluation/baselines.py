from __future__ import annotations


def full_frame_indices(total_frames: int) -> list[int]:
    return list(range(total_frames))


def fixed_skip_indices(total_frames: int, step: int) -> list[int]:
    if step < 1:
        raise ValueError("step must be >= 1")
    return list(range(0, total_frames, step))


def event_window_indices(
    event_frames: list[int],
    total_frames: int,
    pre_event_frames: int = 10,
    post_event_frames: int = 10,
) -> list[int]:
    selected: set[int] = set()
    for frame_number in event_frames:
        start = max(0, frame_number - pre_event_frames)
        end = min(total_frames - 1, frame_number + post_event_frames)
        selected.update(range(start, end + 1))
    return sorted(selected)


def representative_event_indices(
    scored_frames: dict[int, float],
    event_frames: list[int],
    total_frames: int,
    pre_event_frames: int = 10,
    post_event_frames: int = 10,
) -> list[int]:
    representatives = []
    for event_frame in event_frames:
        window = event_window_indices(
            [event_frame], total_frames, pre_event_frames, post_event_frames
        )
        if not window:
            continue
        representatives.append(
            max(window, key=lambda frame_number: scored_frames.get(frame_number, 0.0))
        )
    return sorted(set(representatives))
