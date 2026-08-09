"""Deterministic episode-target selection for the deleafing benchmark."""

from __future__ import annotations

import dataclasses
import random


@dataclasses.dataclass(frozen=True)
class EpisodeTarget:
    """One physical vine/petiole target selected for an episode."""

    vine_name: str
    organ_label: str
    candidate_index: int
    candidate_count: int
    seed: int

    @property
    def key(self) -> str:
        return f"{self.vine_name}/{self.organ_label}"


def candidate_targets(runtimes) -> tuple[tuple[str, str], ...]:
    """Return stable, physically authored cut targets from runtime objects."""

    candidates = []
    for runtime in sorted(runtimes, key=lambda item: item.name):
        labels = set(runtime.rig.cut_joints).intersection(runtime.rig.junctions)
        candidates.extend(
            (runtime.name, label)
            for label in sorted(labels)
            if label.startswith("SubStem_")
        )
    return tuple(candidates)


def resolve_target(
    runtimes,
    *,
    vine_name: str,
    organ_label: str,
    seed: int,
) -> EpisodeTarget:
    """Resolve exact or ``auto`` selectors without depending on traversal order."""

    candidates = candidate_targets(runtimes)
    if vine_name != "auto":
        candidates = tuple(item for item in candidates if item[0] == vine_name)
    if organ_label != "auto":
        candidates = tuple(item for item in candidates if item[1] == organ_label)
    if not candidates:
        available = ", ".join(f"{vine}/{organ}" for vine, organ in candidate_targets(runtimes))
        raise ValueError(
            f"no physical target matches vine={vine_name!r}, organ={organ_label!r}; "
            f"available targets: {available or 'none'}"
        )

    index = random.Random(int(seed)).randrange(len(candidates)) if "auto" in (vine_name, organ_label) else 0
    selected_vine, selected_organ = candidates[index]
    return EpisodeTarget(
        vine_name=selected_vine,
        organ_label=selected_organ,
        candidate_index=index,
        candidate_count=len(candidates),
        seed=int(seed),
    )
