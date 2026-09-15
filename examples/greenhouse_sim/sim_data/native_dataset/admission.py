"""Versioned, in-memory candidate accounting, NEVER a training release gate.

This module does no I/O and imports no capture/review/geometry code. Callers
must supply verified original provenance, the complete global context inventory
and equivalence edges, and all known views (including failed reviews and heldout
views). Batch-local inventories cannot establish global novelty. Evidence IDs
are references to external validation, not measurements performed here.

Four entry points: Policy, decoded_rgb_digest, finalize_context_groups, and
select_candidates. Automatic annotation review is allowed. Actual native clear
contract, calibration evidence, and release validation remain external gates.
"""
from collections import Counter, defaultdict, deque
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math


SCHEMA = "greenhouse.native_admission_candidates.v1"
GROUP_SCHEMA = "greenhouse.global_augmentation_contexts.v1"
POLICY_SCHEMA = "greenhouse.native_admission_policy.v1"
SPLITS = ("train", "validation", "test")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _id(value):
    _require(isinstance(value, str) and bool(value.strip()), "Nonempty identity required")
    return value


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _sha(value):
    _require(isinstance(value, str) and len(value) == 64
             and all(c in "0123456789abcdef" for c in value), "SHA256 identity required")
    return value


@dataclass(frozen=True)
class Policy:
    """Explicit goals and calibrated gates; no invented threshold defaults.

    counts must specify train >= 20000, validation > 0, test > 0, separately.
    calibration must contain geometry and near_image mappings, each with metric,
    threshold (finite, nonnegative), evidence_id, and validated=True. These are
    externally calibrated settings used to compute the supplied pair edges.
    Changing any setting changes the policy digest and invalidates old groups.
    """

    version: str
    counts: dict
    calibration: dict
    max_views_per_context: int = 12

    def _snapshot(self):
        _id(self.version)
        counts, calibration = deepcopy(self.counts), deepcopy(self.calibration)
        _require(set(counts) == set(SPLITS), "Explicit train, validation and test counts required")
        _require(all(type(n) is int and n > 0 for n in counts.values())
                 and counts["train"] >= 20000, "Minimum 20000 TRAIN plus additional heldout required")
        _require(type(self.max_views_per_context) is int and self.max_views_per_context > 0,
                 "Positive integer context cap required")
        _require(set(calibration) == {"geometry", "near_image"}, "Both calibration gates required")
        for gate in calibration.values():
            _require(isinstance(gate, dict) and gate.get("validated") is True,
                     "Validated calibration evidence required")
            _id(gate.get("metric"))
            _id(gate.get("evidence_id"))
            threshold = gate.get("threshold")
            _require(type(threshold) in (int, float) and math.isfinite(threshold) and threshold >= 0,
                     "Calibrated finite nonnegative threshold required")
        return dict(schema=POLICY_SCHEMA, version=self.version, counts=counts,
                    calibration=calibration, max_views_per_context=self.max_views_per_context)


def decoded_rgb_digest(pixels, *, width, height):
    """Hash decoded row-major uint8 RGB bytes AND dimensions, never PNG bytes.

    For a verified bundle use reader.image('inputs/rgb.png'), check RGB uint8
    shape, then pass its tobytes() and dimensions. Decoding stays outside this
    pure-stdlib module; encoded file hashes are not interchangeable with this.
    """
    _require(type(width) is int and type(height) is int and width > 0 and height > 0,
             "Positive RGB dimensions required")
    _require(isinstance(pixels, (bytes, bytearray, memoryview)), "Decoded RGB bytes required")
    pixels = bytes(pixels)
    _require(len(pixels) == width * height * 3, "Decoded RGB byte count mismatch")
    return hashlib.sha256(f"decoded-rgb-u8:{width}:{height}:".encode() + pixels).hexdigest()


class _UnionFind:
    def __init__(self, keys):
        self.parent = {key: key for key in keys}
        self.size = dict.fromkeys(self.parent, 1)

    def find(self, key):
        _require(key in self.parent, "Equivalence endpoint outside global inventory")
        while key != self.parent[key]:
            self.parent[key] = self.parent[self.parent[key]]
            key = self.parent[key]
        return key

    def union(self, left, right):
        left, right = self.find(left), self.find(right)
        if left != right:
            if self.size[left] < self.size[right]:
                left, right = right, left
            self.parent[right] = left
            self.size[left] += self.size[right]

    def edges(self, edges):
        for edge in edges:
            _require(isinstance(edge, (list, tuple)) and len(edge) == 2,
                     "Pair equivalence edge required")
            self.union(_id(edge[0]), _id(edge[1]))

    def groups(self):
        members = defaultdict(list)
        for key in sorted(self.parent):
            members[self.find(key)].append(key)
        # Identity depends on all members, not insertion order or an early leader.
        result = {}
        for group in members.values():
            identity = _digest(group)
            result.update(dict.fromkeys(group, identity))
        return result


def finalize_context_groups(contexts, equivalence_edges, *, policy, evidence_id):
    """Close ALL pair edges transitively, including bridges discovered later.

    contexts maps context_id -> original source_family, source_target,
    source_context_id, qualified_geometry (bool), and geometry_evidence_id when
    qualified. Originals point to themselves and are not augmentations. Each
    source target has exactly one original context. Generated contexts point
    directly to that original and MUST preserve its source family and target.
    Unqualified rigid/scale/rename/color/seed variants are merged with it even
    without an edge. Qualified variants still share any global duplicate group.

    evidence_id attests that geometry comparisons cover this global inventory
    under this policy (including rigid/scale/name-invariant context comparison).
    No online admission: rebuild the snapshot after adding contexts or edges.
    """
    declaration = policy._snapshot()
    _id(evidence_id)
    records = deepcopy(contexts)
    _require(isinstance(records, dict), "Context provenance mapping required")
    union = _UnionFind(records)
    originals, owners = {}, {}
    for key, record in records.items():
        _id(key)
        for field in ("source_family", "source_target", "source_context_id"):
            _id(record.get(field))
        _require(type(record.get("qualified_geometry")) is bool, "Geometry qualification required")
        family, target = record["source_family"], record["source_target"]
        _require(owners.setdefault(target, family) == family, "Source target changed donor family")
        source = record["source_context_id"]
        _require(source in records, "Original source context missing")
        original = records[source]
        _require(original.get("source_context_id") == source
                 and original.get("source_family") == family
                 and original.get("source_target") == target, "Generated source identity changed")
        if key == source:
            _require(not record["qualified_geometry"], "Original cannot claim augmentation novelty")
            _require(originals.setdefault(target, key) == key, "Multiple original contexts for source target")
        if record["qualified_geometry"]:
            _id(record.get("geometry_evidence_id"))
        else:
            union.union(key, source)
    union.edges(equivalence_edges)
    result = dict(schema=GROUP_SCHEMA, finalized=True, policy_sha256=_digest(declaration),
                  evidence_id=evidence_id, contexts=records, groups=union.groups())
    result["sha256"] = _digest(result)
    return result


def _round_robin(iterables):
    active = deque(iter(it) for it in iterables)
    while active:
        iterator = active.popleft()
        try:
            value = next(iterator)
        except StopIteration:
            continue
        active.append(iterator)
        yield value


def select_candidates(records, frozen_splits, context_groups, *, policy,
                      near_image_edges, image_inventory_evidence_id):
    """Select a balanced candidate manifest without mutating any input.

    records: sample_id, target_id (may be generated), source_family,
    source_target (original!), context_id, decoded_rgb_sha256, scene_sha256,
    camera_sha256, annotation_review={passed: bool, method: automatic|manual,
    evidence_id: str}. Optional split must equal the ORIGINAL donor's frozen
    split. Scene/camera identities must be canonical physical scene/pose/optics
    identities, unaffected by filenames, render seeds, or RGB noise.

    Supply the entire known view inventory and finalized near-image pair edges
    (explicitly empty if none), not only eligible or train views. The evidence ID
    attests global near-image comparison under the declared calibrated policy.
    All duplicates spanning splits are quarantined, including ineligible views.
    Same-split view duplicates keep at most one representative; geometry/context
    duplicates share one GLOBAL cap, not one cap per generated ID or donor.

    Hierarchical round robin: original family -> original target -> global
    context -> sample_id, lexicographic and independent of input order. This is
    best-effort balance among nonexhausted eligible buckets, not upsampling.
    """
    declaration = policy._snapshot()
    _id(image_inventory_evidence_id)
    frozen_splits = dict(frozen_splits)
    _require(all(isinstance(k, str) and k and v in SPLITS for k, v in frozen_splits.items()),
             "Original donor-family split mapping required")
    inventory = deepcopy(context_groups)
    receipt = inventory.pop("sha256", None)
    _require(inventory.get("schema") == GROUP_SCHEMA and inventory.get("finalized") is True
             and receipt == _digest(inventory), "Finalized global context snapshot required")
    _require(inventory.get("policy_sha256") == _digest(declaration), "Context policy binding changed")
    contexts, groups = inventory["contexts"], inventory["groups"]
    _require(set(contexts) == set(groups), "Incomplete global context groups")
    context_splits = defaultdict(set)
    for key, provenance in contexts.items():
        _require(provenance["source_family"] in frozen_splits, "Original donor missing frozen split")
        context_splits[groups[key]].add(frozen_splits[provenance["source_family"]])

    views, reasons, splits = {}, {}, {}
    for incoming in records:
        row = deepcopy(incoming)
        key = _id(row.get("sample_id"))
        _require(key not in views, "Unique sample_id required")
        for field in ("target_id", "source_family", "source_target", "context_id"):
            _id(row.get(field))
        _require(row["context_id"] in contexts, "View context outside global inventory")
        for field in ("decoded_rgb_sha256", "scene_sha256", "camera_sha256"):
            _sha(row.get(field))
        provenance = contexts[row["context_id"]]
        split = frozen_splits[provenance["source_family"]]
        splits[key], views[key], reasons[key] = split, row, []
        if any(row[field] != provenance[field] for field in ("source_family", "source_target")):
            reasons[key].append("source_identity_mismatch")
        if "split" in row and row["split"] != split:
            reasons[key].append("frozen_split_mismatch")
        review = row.get("annotation_review", {})
        if not (isinstance(review, dict) and review.get("passed") is True
                and review.get("method") in ("automatic", "manual")
                and isinstance(review.get("evidence_id"), str) and review["evidence_id"].strip()):
            reasons[key].append("annotation_review_required")
        if len(context_splits[groups[row["context_id"]]]) > 1:
            reasons[key].append("cross_split_context_duplicate")

    duplicates = _UnionFind(views)
    rgb_seen, pose_seen = {}, {}
    for key, row in views.items():
        rgb = row["decoded_rgb_sha256"]
        pose = (row["scene_sha256"], row["camera_sha256"])
        duplicates.union(key, rgb_seen.setdefault(rgb, key))
        duplicates.union(key, pose_seen.setdefault(pose, key))
    duplicates.edges(near_image_edges)
    view_groups = duplicates.groups()
    duplicate_splits = defaultdict(set)
    for key, group in view_groups.items():
        duplicate_splits[group].add(splits[key])
    for key, group in view_groups.items():
        if len(duplicate_splits[group]) > 1:
            reasons[key].append("cross_split_view_duplicate")

    tree = {split: {} for split in SPLITS}
    for key in sorted(views):
        if reasons[key]:
            continue
        row = views[key]
        leaf = tree[splits[key]].setdefault(row["source_family"], {}).setdefault(
            row["source_target"], {}).setdefault(groups[row["context_id"]], [])
        leaf.append(key)
    context_counts, used_duplicates = Counter(), set()

    def admit(keys):
        for key in keys:
            context = groups[views[key]["context_id"]]
            if view_groups[key] in used_duplicates:
                reasons[key].append("duplicate_view")
            elif context_counts[context] >= declaration["max_views_per_context"]:
                reasons[key].append("context_cap")
            else:
                context_counts[context] += 1
                used_duplicates.add(view_groups[key])
                yield key

    def schedule(node):
        if isinstance(node, list):
            return admit(node)
        return _round_robin(schedule(node[key]) for key in sorted(node))

    selected, counts = [], dict.fromkeys(SPLITS, 0)
    for split in SPLITS:
        for key in schedule(tree[split]):
            row = views[key]
            selected.append(dict(sample_id=key, target_id=row["target_id"], split=split,
                                 source_family=row["source_family"], source_target=row["source_target"],
                                 augmentation_context_id=groups[row["context_id"]],
                                 training_approved=False))
            counts[split] += 1
            if counts[split] == declaration["counts"][split]:
                break
    chosen = {row["sample_id"] for row in selected}
    excluded = {key: reasons[key] or ["split_quota_reached"]
                for key in sorted(views) if key not in chosen}
    shortfall = {split: declaration["counts"][split] - counts[split] for split in SPLITS}
    return dict(schema=SCHEMA, state="candidate_selection_only", policy=declaration,
                policy_sha256=_digest(declaration), frozen_splits_sha256=_digest(frozen_splits),
                context_inventory_sha256=receipt, image_inventory_evidence_id=image_inventory_evidence_id,
                view_inventory_sha256=_digest([views[key] for key in sorted(views)]),
                view_duplicate_groups=view_groups, selected=selected, excluded=excluded,
                requested_counts=deepcopy(declaration["counts"]), counts=counts, shortfall=shortfall,
                candidate_counts_met=not any(shortfall.values()), context_counts=dict(context_counts),
                family_counts=dict(Counter(row["source_family"] for row in selected)),
                training_approved=False, release_validation_required=True,
                calibration_status="externally_attested_not_verified_here",
                new_biological_families=False, original_reviews_modified=False,
                captures_modified=False, frozen_splits_modified=False)
