"""

Reading and playback of Vector animation files.

Vector uses the same FlatBuffers envelope as Cozmo (``AnimClips``), but its
schema adds sprite boxes, audio event groups, scanline opacity, and a narrower
backpack-light layout.  This module reads the schema directly instead of
depending on generated FlatBuffers code.  It therefore works with all Python
versions supported by PyCozmo and does not need ``flatc`` at build time.

The public data model is deliberately small and easy to inspect: load a file
with :meth:`VectorAnimations.from_fb_file`, iterate over ``.clips`` and use
:class:`VectorAnimator` to drive a Cozmo-like client or custom callbacks.

"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import os
import random
import struct
import threading
import time
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union


__all__ = [
    "VectorAnimationError",
    "VectorKeyframe",
    "VectorLiftHeight",
    "VectorProceduralFace",
    "VectorHeadAngle",
    "VectorAudioEventGroup",
    "VectorAudioState",
    "VectorAudioSwitch",
    "VectorAudioParameter",
    "VectorRobotAudio",
    "VectorBackpackLights",
    "VectorFaceAnimation",
    "VectorSpriteBox",
    "VectorEvent",
    "VectorBodyMotion",
    "VectorRecordHeading",
    "VectorTurnToRecordedHeading",
    "VectorClip",
    "VectorAnimations",
    "VectorAnimClips",
    "VectorFrame",
    "VectorAnimator",
    "load_vector_animations",
    "is_vector_animation",
    "cozmo_clips_to_vector",
]


class VectorAnimationError(ValueError):
    """ Raised when a Vector animation file is malformed. """


# ---------------------------------------------------------------------------
# Minimal FlatBuffers reader
# ---------------------------------------------------------------------------

class _FlatBuffer:
    def __init__(self, data: Union[bytes, bytearray, memoryview]) -> None:
        self.data = memoryview(bytes(data))
        self.size = len(self.data)

    def _check(self, pos: int, length: int) -> None:
        if pos < 0 or length < 0 or pos + length > self.size:
            raise VectorAnimationError("Truncated FlatBuffers data at byte {}.".format(pos))

    def u8(self, pos: int) -> int:
        self._check(pos, 1)
        return int(self.data[pos])

    def i8(self, pos: int) -> int:
        self._check(pos, 1)
        return struct.unpack_from("<b", self.data, pos)[0]

    def u16(self, pos: int) -> int:
        self._check(pos, 2)
        return struct.unpack_from("<H", self.data, pos)[0]

    def i16(self, pos: int) -> int:
        self._check(pos, 2)
        return struct.unpack_from("<h", self.data, pos)[0]

    def u32(self, pos: int) -> int:
        self._check(pos, 4)
        return struct.unpack_from("<I", self.data, pos)[0]

    def i32(self, pos: int) -> int:
        self._check(pos, 4)
        return struct.unpack_from("<i", self.data, pos)[0]

    def f32(self, pos: int) -> float:
        self._check(pos, 4)
        return struct.unpack_from("<f", self.data, pos)[0]

    def string_at(self, pos: int) -> str:
        length = self.u32(pos)
        self._check(pos + 4, length)
        return bytes(self.data[pos + 4:pos + 4 + length]).decode("utf-8", errors="replace")

    def table(self, pos: int) -> "_Table":
        return _Table(self, pos)


class _Table:
    """ One FlatBuffers table. """

    __slots__ = ("reader", "pos", "vtable", "vtable_size", "table_size")

    def __init__(self, reader: _FlatBuffer, pos: int) -> None:
        self.reader = reader
        self.pos = int(pos)
        reader._check(self.pos, 4)
        soffset = reader.i32(self.pos)
        self.vtable = self.pos - soffset
        reader._check(self.vtable, 4)
        self.vtable_size = reader.u16(self.vtable)
        self.table_size = reader.u16(self.vtable + 2)
        if self.vtable_size < 4 or self.table_size < 4:
            raise VectorAnimationError("Invalid FlatBuffers table at byte {}.".format(self.pos))

    def field(self, slot: int) -> Optional[int]:
        offset_pos = self.vtable + 4 + int(slot) * 2
        if offset_pos + 2 > self.vtable + self.vtable_size:
            return None
        offset = self.reader.u16(offset_pos)
        if not offset:
            return None
        field_pos = self.pos + offset
        self.reader._check(field_pos, 1)
        return field_pos

    def u8(self, slot: int, default: int = 0) -> int:
        pos = self.field(slot)
        return default if pos is None else self.reader.u8(pos)

    def i8(self, slot: int, default: int = 0) -> int:
        pos = self.field(slot)
        return default if pos is None else self.reader.i8(pos)

    def bool(self, slot: int, default: bool = False) -> bool:
        return bool(self.u8(slot, 1 if default else 0))

    def u16(self, slot: int, default: int = 0) -> int:
        pos = self.field(slot)
        return default if pos is None else self.reader.u16(pos)

    def i16(self, slot: int, default: int = 0) -> int:
        pos = self.field(slot)
        return default if pos is None else self.reader.i16(pos)

    def u32(self, slot: int, default: int = 0) -> int:
        pos = self.field(slot)
        return default if pos is None else self.reader.u32(pos)

    def i32(self, slot: int, default: int = 0) -> int:
        pos = self.field(slot)
        return default if pos is None else self.reader.i32(pos)

    def f32(self, slot: int, default: float = 0.0) -> float:
        pos = self.field(slot)
        return default if pos is None else self.reader.f32(pos)

    def string(self, slot: int, default: str = "") -> str:
        pos = self.field(slot)
        if pos is None:
            return default
        target = pos + self.reader.u32(pos)
        self.reader._check(target, 4)
        return self.reader.string_at(target)

    def _target(self, slot: int) -> Optional[int]:
        pos = self.field(slot)
        if pos is None:
            return None
        target = pos + self.reader.u32(pos)
        self.reader._check(target, 4)
        return target

    def table(self, slot: int) -> Optional["_Table"]:
        target = self._target(slot)
        return None if target is None else _Table(self.reader, target)

    def vector_length(self, slot: int) -> int:
        target = self._target(slot)
        if target is None:
            return 0
        length = self.reader.u32(target)
        # Vectors of offsets take four bytes per element; this is the minimum.
        if length > (self.reader.size - target - 4) // 4:
            raise VectorAnimationError("Invalid vector length at byte {}.".format(target))
        return int(length)

    def vector_scalars(self, slot: int, fmt: str) -> List[Any]:
        target = self._target(slot)
        if target is None:
            return []
        length = self.reader.u32(target)
        size = struct.calcsize(fmt)
        start = target + 4
        if length > (self.reader.size - start) // size:
            raise VectorAnimationError("Truncated scalar vector at byte {}.".format(target))
        if not length:
            return []
        try:
            return list(struct.unpack_from("<{}{}".format(length, fmt), self.reader.data, start))
        except struct.error as exc:
            raise VectorAnimationError("Invalid scalar vector.") from exc

    def vector_tables(self, slot: int) -> List["_Table"]:
        target = self._target(slot)
        if target is None:
            return []
        length = self.reader.u32(target)
        start = target + 4
        if length > (self.reader.size - start) // 4:
            raise VectorAnimationError("Truncated table vector at byte {}.".format(target))
        result: List[_Table] = []
        for index in range(length):
            element = start + index * 4
            obj = element + self.reader.u32(element)
            result.append(_Table(self.reader, obj))
        return result


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _integer(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _lookup(data: Dict[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in data:
            return data[name]
    return default


def _color_vector(value: Any) -> Tuple[float, float, float, float]:
    values = list(value or ())
    values = [_float(item) for item in values[:4]]
    while len(values) < 4:
        values.append(0.0)
    return values[0], values[1], values[2], values[3]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class VectorKeyframe:
    """ Base class for a Vector animation keyframe. """

    trigger_time_ms: int = 0

    def __post_init__(self) -> None:
        self.trigger_time_ms = max(0, _integer(self.trigger_time_ms))

    @property
    def time_ms(self) -> int:
        """ Alias used by timeline code. """
        return self.trigger_time_ms

    def to_dict(self) -> Dict[str, Any]:
        return {"triggerTime_ms": self.trigger_time_ms}


@dataclass
class VectorLiftHeight(VectorKeyframe):
    duration_time_ms: int = 0
    height_mm: int = 0
    height_variability_mm: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        self.duration_time_ms = max(0, _integer(self.duration_time_ms))
        self.height_mm = max(0, _integer(self.height_mm))
        self.height_variability_mm = max(0, _integer(self.height_variability_mm))

    @property
    def duration_ms(self) -> int:
        return self.duration_time_ms

    @classmethod
    def from_table(cls, table: _Table) -> "VectorLiftHeight":
        return cls(trigger_time_ms=table.u32(0),
                   duration_time_ms=table.u32(1),
                   height_mm=table.u8(2),
                   height_variability_mm=table.u8(3))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VectorLiftHeight":
        return cls(
            trigger_time_ms=_lookup(data, "triggerTime_ms", "trigger_time_ms", default=0),
            duration_time_ms=_lookup(data, "durationTime_ms", "duration_time_ms", "duration_ms", default=0),
            height_mm=_lookup(data, "height_mm", default=0),
            height_variability_mm=_lookup(data, "heightVariability_mm", "height_variability_mm", default=0),
        )

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "durationTime_ms": self.duration_time_ms,
            "height_mm": self.height_mm,
            "heightVariability_mm": self.height_variability_mm,
        })
        return data


@dataclass
class VectorProceduralFace(VectorKeyframe):
    face_angle: float = 0.0
    face_center_x: float = 0.0
    face_center_y: float = 0.0
    face_scale_x: float = 1.0
    face_scale_y: float = 1.0
    left_eye: List[float] = field(default_factory=list)
    right_eye: List[float] = field(default_factory=list)
    scanline_opacity: float = 1.0

    def __post_init__(self) -> None:
        super().__post_init__()
        self.face_angle = _float(self.face_angle)
        self.face_center_x = _float(self.face_center_x)
        self.face_center_y = _float(self.face_center_y)
        self.face_scale_x = _float(self.face_scale_x, 1.0)
        self.face_scale_y = _float(self.face_scale_y, 1.0)
        self.left_eye = [_float(value) for value in self.left_eye]
        self.right_eye = [_float(value) for value in self.right_eye]
        self.scanline_opacity = _float(self.scanline_opacity, 1.0)

    @property
    def face_params(self) -> List[float]:
        """ Parameters accepted by :class:`pycozmo.procedural_face.ProceduralFace`. """
        return [self.face_center_x, self.face_center_y,
                self.face_scale_x, self.face_scale_y, self.face_angle,
                *self.left_eye, *self.right_eye]

    @classmethod
    def from_table(cls, table: _Table) -> "VectorProceduralFace":
        return cls(trigger_time_ms=table.u32(0),
                   face_angle=table.f32(1),
                   face_center_x=table.f32(2),
                   face_center_y=table.f32(3),
                   face_scale_x=table.f32(4, 1.0),
                   face_scale_y=table.f32(5, 1.0),
                   left_eye=table.vector_scalars(6, "f"),
                   right_eye=table.vector_scalars(7, "f"),
                   scanline_opacity=table.f32(8, 1.0))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VectorProceduralFace":
        return cls(
            trigger_time_ms=_lookup(data, "triggerTime_ms", "trigger_time_ms", default=0),
            face_angle=_lookup(data, "faceAngle", "face_angle", default=0.0),
            face_center_x=_lookup(data, "faceCenterX", "face_center_x", default=0.0),
            face_center_y=_lookup(data, "faceCenterY", "face_center_y", default=0.0),
            face_scale_x=_lookup(data, "faceScaleX", "face_scale_x", default=1.0),
            face_scale_y=_lookup(data, "faceScaleY", "face_scale_y", default=1.0),
            left_eye=_lookup(data, "leftEye", "left_eye", default=[]),
            right_eye=_lookup(data, "rightEye", "right_eye", default=[]),
            scanline_opacity=_lookup(data, "scanlineOpacity", "scanline_opacity", default=1.0),
        )

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "faceAngle": self.face_angle,
            "faceCenterX": self.face_center_x,
            "faceCenterY": self.face_center_y,
            "faceScaleX": self.face_scale_x,
            "faceScaleY": self.face_scale_y,
            "leftEye": list(self.left_eye),
            "rightEye": list(self.right_eye),
            "scanlineOpacity": self.scanline_opacity,
        })
        return data


@dataclass
class VectorHeadAngle(VectorKeyframe):
    duration_time_ms: int = 0
    angle_deg: int = 0
    angle_variability_deg: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        self.duration_time_ms = max(0, _integer(self.duration_time_ms))
        self.angle_deg = _integer(self.angle_deg)
        self.angle_variability_deg = max(0, _integer(self.angle_variability_deg))

    @property
    def duration_ms(self) -> int:
        return self.duration_time_ms

    @property
    def angle_rad(self) -> float:
        return math.radians(self.angle_deg)

    @classmethod
    def from_table(cls, table: _Table) -> "VectorHeadAngle":
        return cls(trigger_time_ms=table.u32(0),
                   duration_time_ms=table.u32(1),
                   angle_deg=table.i8(2, 0),
                   angle_variability_deg=table.u8(3))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VectorHeadAngle":
        return cls(
            trigger_time_ms=_lookup(data, "triggerTime_ms", "trigger_time_ms", default=0),
            duration_time_ms=_lookup(data, "durationTime_ms", "duration_time_ms", "duration_ms", default=0),
            angle_deg=_lookup(data, "angle_deg", "angleDeg", default=0),
            angle_variability_deg=_lookup(data, "angleVariability_deg", "angle_variability_deg", default=0),
        )

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "durationTime_ms": self.duration_time_ms,
            "angle_deg": self.angle_deg,
            "angleVariability_deg": self.angle_variability_deg,
        })
        return data


@dataclass
class VectorAudioEventGroup:
    event_ids: List[int] = field(default_factory=list)
    volumes: List[float] = field(default_factory=list)
    probabilities: List[float] = field(default_factory=list)

    def choose(self, rng: Optional[random.Random] = None) -> Optional[Tuple[int, float]]:
        """ Choose an event according to its probability weight. """
        if not self.event_ids:
            return None
        rng = rng or random
        probabilities = list(self.probabilities)
        if len(probabilities) != len(self.event_ids) or sum(max(0.0, value) for value in probabilities) <= 0:
            probabilities = [1.0] * len(self.event_ids)
        total = sum(max(0.0, value) for value in probabilities)
        pick = rng.random() * total
        cumulative = 0.0
        index = len(self.event_ids) - 1
        for i, probability in enumerate(probabilities):
            cumulative += max(0.0, probability)
            if pick <= cumulative:
                index = i
                break
        volume = self.volumes[index] if index < len(self.volumes) else 1.0
        return int(self.event_ids[index]), _float(volume, 1.0)

    @classmethod
    def from_table(cls, table: _Table) -> "VectorAudioEventGroup":
        return cls(event_ids=[int(value) for value in table.vector_scalars(0, "I")],
                   volumes=[_float(value) for value in table.vector_scalars(1, "f")],
                   probabilities=[_float(value) for value in table.vector_scalars(2, "f")])

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VectorAudioEventGroup":
        return cls(event_ids=[_integer(value) for value in _lookup(data, "eventIds", "event_ids", default=[])],
                   volumes=[_float(value) for value in _lookup(data, "volumes", default=[])],
                   probabilities=[_float(value) for value in _lookup(data, "probabilities", default=[])])

    def to_dict(self) -> Dict[str, Any]:
        return {"eventIds": list(self.event_ids),
                "volumes": list(self.volumes),
                "probabilities": list(self.probabilities)}


@dataclass
class VectorAudioState:
    state_group_id: int = 0
    state_id: int = 0

    @classmethod
    def from_table(cls, table: _Table) -> "VectorAudioState":
        return cls(state_group_id=table.u32(0), state_id=table.u32(1))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VectorAudioState":
        return cls(state_group_id=_lookup(data, "stateGroupId", "state_group_id", default=0),
                   state_id=_lookup(data, "stateId", "state_id", default=0))

    def to_dict(self) -> Dict[str, Any]:
        return {"stateGroupId": self.state_group_id, "stateId": self.state_id}


@dataclass
class VectorAudioSwitch:
    switch_group_id: int = 0
    state_id: int = 0

    @classmethod
    def from_table(cls, table: _Table) -> "VectorAudioSwitch":
        return cls(switch_group_id=table.u32(0), state_id=table.u32(1))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VectorAudioSwitch":
        return cls(switch_group_id=_lookup(data, "switchGroupId", "switch_group_id", default=0),
                   state_id=_lookup(data, "stateId", "state_id", default=0))

    def to_dict(self) -> Dict[str, Any]:
        return {"switchGroupId": self.switch_group_id, "stateId": self.state_id}


@dataclass
class VectorAudioParameter:
    parameter_id: int = 0
    value: float = 0.0
    time_ms: int = 0
    curve_type: int = 0

    @classmethod
    def from_table(cls, table: _Table) -> "VectorAudioParameter":
        return cls(parameter_id=table.u32(0), value=table.f32(1),
                   time_ms=table.u32(2), curve_type=table.u8(3))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VectorAudioParameter":
        return cls(parameter_id=_lookup(data, "parameterID", "parameter_id", default=0),
                   value=_lookup(data, "value", default=0.0),
                   time_ms=_lookup(data, "time_ms", "timeMs", default=0),
                   curve_type=_lookup(data, "curveType", "curve_type", default=0))

    def to_dict(self) -> Dict[str, Any]:
        return {"parameterID": self.parameter_id, "value": self.value,
                "time_ms": self.time_ms, "curveType": self.curve_type}


@dataclass
class VectorRobotAudio(VectorKeyframe):
    event_groups: List[VectorAudioEventGroup] = field(default_factory=list)
    states: List[VectorAudioState] = field(default_factory=list)
    switches: List[VectorAudioSwitch] = field(default_factory=list)
    parameters: List[VectorAudioParameter] = field(default_factory=list)

    def choose_event(self, rng: Optional[random.Random] = None) -> Optional[Tuple[int, float]]:
        rng = rng or random
        if not self.event_groups:
            return None
        index = rng.randrange(len(self.event_groups)) if len(self.event_groups) > 1 else 0
        return self.event_groups[index].choose(rng)

    @classmethod
    def from_table(cls, table: _Table) -> "VectorRobotAudio":
        try:
            groups = [VectorAudioEventGroup.from_table(item) for item in table.vector_tables(1)]
            states = [VectorAudioState.from_table(item) for item in table.vector_tables(2)]
            switches = [VectorAudioSwitch.from_table(item) for item in table.vector_tables(3)]
            parameters = [VectorAudioParameter.from_table(item) for item in table.vector_tables(4)]
        except (VectorAnimationError, struct.error):
            # A Cozmo animation uses a different RobotAudio layout.  Let the
            # rest of the clip remain readable instead of rejecting the file.
            groups, states, switches, parameters = [], [], [], []
        return cls(trigger_time_ms=table.u32(0), event_groups=groups,
                   states=states, switches=switches, parameters=parameters)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VectorRobotAudio":
        return cls(
            trigger_time_ms=_lookup(data, "triggerTime_ms", "trigger_time_ms", default=0),
            event_groups=[VectorAudioEventGroup.from_dict(item)
                          for item in _lookup(data, "eventGroups", "event_groups", default=[])],
            states=[VectorAudioState.from_dict(item) for item in _lookup(data, "states", default=[])],
            switches=[VectorAudioSwitch.from_dict(item) for item in _lookup(data, "switches", default=[])],
            parameters=[VectorAudioParameter.from_dict(item)
                        for item in _lookup(data, "parameters", default=[])],
        )

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "eventGroups": [item.to_dict() for item in self.event_groups],
            "states": [item.to_dict() for item in self.states],
            "switches": [item.to_dict() for item in self.switches],
            "parameters": [item.to_dict() for item in self.parameters],
        })
        return data


@dataclass
class VectorBackpackLights(VectorKeyframe):
    duration_time_ms: int = 0
    front: List[float] = field(default_factory=list)
    middle: List[float] = field(default_factory=list)
    back: List[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.duration_time_ms = max(0, _integer(self.duration_time_ms))
        self.front = list(_color_vector(self.front))
        self.middle = list(_color_vector(self.middle))
        self.back = list(_color_vector(self.back))

    @property
    def duration_ms(self) -> int:
        return self.duration_time_ms

    @property
    def front_rgb(self) -> Tuple[int, int, int]:
        return _rgb(self.front)

    @property
    def middle_rgb(self) -> Tuple[int, int, int]:
        return _rgb(self.middle)

    @property
    def back_rgb(self) -> Tuple[int, int, int]:
        return _rgb(self.back)

    @classmethod
    def from_table(cls, table: _Table) -> "VectorBackpackLights":
        return cls(trigger_time_ms=table.u32(0), duration_time_ms=table.u32(1),
                   front=table.vector_scalars(2, "f"),
                   middle=table.vector_scalars(3, "f"),
                   back=table.vector_scalars(4, "f"))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VectorBackpackLights":
        return cls(
            trigger_time_ms=_lookup(data, "triggerTime_ms", "trigger_time_ms", default=0),
            duration_time_ms=_lookup(data, "durationTime_ms", "duration_time_ms", default=0),
            front=_lookup(data, "Front", "front", default=[]),
            middle=_lookup(data, "Middle", "middle", default=[]),
            back=_lookup(data, "Back", "back", default=[]),
        )

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({"durationTime_ms": self.duration_time_ms,
                     "Front": list(self.front),
                     "Middle": list(self.middle),
                     "Back": list(self.back)})
        return data


def _rgb(color: Sequence[float]) -> Tuple[int, int, int]:
    values = list(color)
    while len(values) < 3:
        values.append(0.0)
    # Vector colors are floats in 0..1; a few community exports use 0..255.
    scale = 255.0 if max(values[:3], default=0.0) <= 1.0 else 1.0
    return tuple(int(round(max(0.0, min(1.0, float(value))) * scale)) for value in values[:3])  # type: ignore


@dataclass
class VectorFaceAnimation(VectorKeyframe):
    anim_name: str = ""
    scanline_opacity: float = 1.0

    def __post_init__(self) -> None:
        super().__post_init__()
        self.anim_name = str(self.anim_name or "")
        self.scanline_opacity = _float(self.scanline_opacity, 1.0)

    @classmethod
    def from_table(cls, table: _Table) -> "VectorFaceAnimation":
        return cls(trigger_time_ms=table.u32(0), anim_name=table.string(1),
                   scanline_opacity=table.f32(2, 1.0))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VectorFaceAnimation":
        return cls(trigger_time_ms=_lookup(data, "triggerTime_ms", "trigger_time_ms", default=0),
                   anim_name=_lookup(data, "animName", "anim_name", default=""),
                   scanline_opacity=_lookup(data, "scanlineOpacity", "scanline_opacity", default=1.0))

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({"animName": self.anim_name, "scanlineOpacity": self.scanline_opacity})
        return data


@dataclass
class VectorSpriteBox(VectorKeyframe):
    sprite_box_name: str = ""
    layer: str = ""
    asset_name: str = ""
    render_method: str = "RGBA"
    sprite_seq_end_type: str = ""
    alpha: float = 100.0
    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        self.sprite_box_name = str(self.sprite_box_name or "")
        self.layer = str(self.layer or "")
        self.asset_name = str(self.asset_name or "")
        self.render_method = str(self.render_method or "RGBA")
        self.sprite_seq_end_type = str(self.sprite_seq_end_type or "")
        self.alpha = _float(self.alpha, 100.0)
        self.x = _integer(self.x)
        self.y = _integer(self.y)
        self.width = max(0, _integer(self.width))
        self.height = max(0, _integer(self.height))

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return self.x, self.y, self.x + self.width, self.y + self.height

    @property
    def is_clear(self) -> bool:
        return (self.asset_name == "clear_sprite_box" or
                (self.width == 0 and self.height == 0 and not self.asset_name))

    @classmethod
    def from_table(cls, table: _Table) -> "VectorSpriteBox":
        return cls(trigger_time_ms=table.u32(0),
                   sprite_box_name=table.string(1),
                   layer=table.string(2),
                   asset_name=table.string(3),
                   render_method=table.string(4),
                   sprite_seq_end_type=table.string(5),
                   alpha=table.f32(6, 100.0),
                   x=table.i32(7), y=table.i32(8),
                   width=table.u32(9), height=table.u32(10))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VectorSpriteBox":
        return cls(
            trigger_time_ms=_lookup(data, "triggerTime_ms", "trigger_time_ms", default=0),
            sprite_box_name=_lookup(data, "spriteBoxName", "sprite_box_name", default=""),
            layer=_lookup(data, "layer", default=""),
            asset_name=_lookup(data, "assetName", "asset_name", default=""),
            render_method=_lookup(data, "renderMethod", "render_method", default="RGBA"),
            sprite_seq_end_type=_lookup(data, "spriteSeqEndType", "sprite_seq_end_type", default=""),
            alpha=_lookup(data, "alpha", default=100.0),
            x=_lookup(data, "xPos", "x", default=0),
            y=_lookup(data, "yPos", "y", default=0),
            width=_lookup(data, "width", default=0),
            height=_lookup(data, "height", default=0),
        )

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "spriteBoxName": self.sprite_box_name, "layer": self.layer,
            "assetName": self.asset_name, "renderMethod": self.render_method,
            "spriteSeqEndType": self.sprite_seq_end_type, "alpha": self.alpha,
            "xPos": self.x, "yPos": self.y, "width": self.width, "height": self.height,
        })
        return data


@dataclass
class VectorEvent(VectorKeyframe):
    event_id: str = ""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.event_id = str(self.event_id or "")

    @classmethod
    def from_table(cls, table: _Table) -> "VectorEvent":
        return cls(trigger_time_ms=table.u32(0), event_id=table.string(1))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VectorEvent":
        return cls(trigger_time_ms=_lookup(data, "triggerTime_ms", "trigger_time_ms", default=0),
                   event_id=_lookup(data, "event_id", "eventId", default=""))

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["event_id"] = self.event_id
        return data


@dataclass
class VectorBodyMotion(VectorKeyframe):
    duration_time_ms: int = 0
    radius_mm: str = ""
    speed: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        self.duration_time_ms = max(0, _integer(self.duration_time_ms))
        self.radius_mm = str(self.radius_mm or "")
        self.speed = _integer(self.speed)

    @property
    def duration_ms(self) -> int:
        return self.duration_time_ms

    @property
    def is_straight(self) -> bool:
        return self.radius_mm.upper() == "STRAIGHT"

    @property
    def is_turn_in_place(self) -> bool:
        return self.radius_mm.upper() == "TURN_IN_PLACE"

    @classmethod
    def from_table(cls, table: _Table) -> "VectorBodyMotion":
        return cls(trigger_time_ms=table.u32(0), duration_time_ms=table.u32(1),
                   radius_mm=table.string(2), speed=table.i16(3))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VectorBodyMotion":
        return cls(trigger_time_ms=_lookup(data, "triggerTime_ms", "trigger_time_ms", default=0),
                   duration_time_ms=_lookup(data, "durationTime_ms", "duration_time_ms", default=0),
                   radius_mm=_lookup(data, "radius_mm", "radiusMm", default=""),
                   speed=_lookup(data, "speed", default=0))

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({"durationTime_ms": self.duration_time_ms,
                     "radius_mm": self.radius_mm, "speed": self.speed})
        return data


@dataclass
class VectorRecordHeading(VectorKeyframe):
    @classmethod
    def from_table(cls, table: _Table) -> "VectorRecordHeading":
        return cls(trigger_time_ms=table.u32(0))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VectorRecordHeading":
        return cls(trigger_time_ms=_lookup(data, "triggerTime_ms", "trigger_time_ms", default=0))


@dataclass
class VectorTurnToRecordedHeading(VectorKeyframe):
    duration_time_ms: int = 0
    offset_deg: int = 0
    speed_deg_per_sec: int = 0
    accel_deg_per_sec2: int = 1000
    decel_deg_per_sec2: int = 1000
    tolerance_deg: int = 2
    num_half_revs: int = 0
    use_shortest_dir: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        self.duration_time_ms = max(0, _integer(self.duration_time_ms))
        self.offset_deg = _integer(self.offset_deg)
        self.speed_deg_per_sec = _integer(self.speed_deg_per_sec)
        self.accel_deg_per_sec2 = _integer(self.accel_deg_per_sec2, 1000)
        self.decel_deg_per_sec2 = _integer(self.decel_deg_per_sec2, 1000)
        self.tolerance_deg = max(0, _integer(self.tolerance_deg, 2))
        self.num_half_revs = max(0, _integer(self.num_half_revs))
        self.use_shortest_dir = bool(self.use_shortest_dir)

    @classmethod
    def from_table(cls, table: _Table) -> "VectorTurnToRecordedHeading":
        return cls(trigger_time_ms=table.u32(0), duration_time_ms=table.u32(1),
                   offset_deg=table.i16(2), speed_deg_per_sec=table.i16(3),
                   accel_deg_per_sec2=table.i16(4, 1000),
                   decel_deg_per_sec2=table.i16(5, 1000),
                   tolerance_deg=table.u16(6, 2),
                   num_half_revs=table.u16(7),
                   use_shortest_dir=table.bool(8))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VectorTurnToRecordedHeading":
        return cls(
            trigger_time_ms=_lookup(data, "triggerTime_ms", "trigger_time_ms", default=0),
            duration_time_ms=_lookup(data, "durationTime_ms", "duration_time_ms", default=0),
            offset_deg=_lookup(data, "offset_deg", "offsetDeg", default=0),
            speed_deg_per_sec=_lookup(data, "speed_degPerSec", "speed_deg_per_sec", default=0),
            accel_deg_per_sec2=_lookup(data, "accel_degPerSec2", "accel_deg_per_sec2", default=1000),
            decel_deg_per_sec2=_lookup(data, "decel_degPerSec2", "decel_deg_per_sec2", default=1000),
            tolerance_deg=_lookup(data, "tolerance_deg", "toleranceDeg", default=2),
            num_half_revs=_lookup(data, "numHalfRevs", "num_half_revs", default=0),
            use_shortest_dir=_lookup(data, "useShortestDir", "use_shortest_dir", default=False),
        )

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "durationTime_ms": self.duration_time_ms,
            "offset_deg": self.offset_deg,
            "speed_degPerSec": self.speed_deg_per_sec,
            "accel_degPerSec2": self.accel_deg_per_sec2,
            "decel_degPerSec2": self.decel_deg_per_sec2,
            "tolerance_deg": self.tolerance_deg,
            "numHalfRevs": self.num_half_revs,
            "useShortestDir": self.use_shortest_dir,
        })
        return data


#: Mapping from FlatBuffers/JSON track names to the Python model.
KEYFRAME_TYPES: Dict[str, Any] = {
    "LiftHeightKeyFrame": VectorLiftHeight,
    "ProceduralFaceKeyFrame": VectorProceduralFace,
    "HeadAngleKeyFrame": VectorHeadAngle,
    "RobotAudioKeyFrame": VectorRobotAudio,
    "BackpackLightsKeyFrame": VectorBackpackLights,
    "FaceAnimationKeyFrame": VectorFaceAnimation,
    "EventKeyFrame": VectorEvent,
    "BodyMotionKeyFrame": VectorBodyMotion,
    "RecordHeadingKeyFrame": VectorRecordHeading,
    "TurnToRecordedHeadingKeyFrame": VectorTurnToRecordedHeading,
    "SpriteBoxKeyFrame": VectorSpriteBox,
}

_TRACK_NAMES = {
    VectorLiftHeight: "LiftHeightKeyFrame",
    VectorProceduralFace: "ProceduralFaceKeyFrame",
    VectorHeadAngle: "HeadAngleKeyFrame",
    VectorRobotAudio: "RobotAudioKeyFrame",
    VectorBackpackLights: "BackpackLightsKeyFrame",
    VectorFaceAnimation: "FaceAnimationKeyFrame",
    VectorEvent: "EventKeyFrame",
    VectorBodyMotion: "BodyMotionKeyFrame",
    VectorRecordHeading: "RecordHeadingKeyFrame",
    VectorTurnToRecordedHeading: "TurnToRecordedHeadingKeyFrame",
    VectorSpriteBox: "SpriteBoxKeyFrame",
}

# FlatBuffers slot numbers for each track in the Vector Keyframes table.
_TRACK_SLOTS = {
    "LiftHeightKeyFrame": 0,
    "ProceduralFaceKeyFrame": 1,
    "HeadAngleKeyFrame": 2,
    "RobotAudioKeyFrame": 3,
    "BackpackLightsKeyFrame": 4,
    "FaceAnimationKeyFrame": 5,
    "EventKeyFrame": 6,
    "BodyMotionKeyFrame": 7,
    "RecordHeadingKeyFrame": 8,
    "TurnToRecordedHeadingKeyFrame": 9,
    "SpriteBoxKeyFrame": 10,
}


@dataclass
class VectorClip:
    """ One named Vector animation clip. """

    name: str = ""
    keyframes: List[VectorKeyframe] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.name = str(self.name or "")
        self.keyframes = sorted(list(self.keyframes), key=lambda item: item.trigger_time_ms)

    @property
    def duration_ms(self) -> int:
        """ Length of the clip, including the last keyframe duration. """
        end = 0
        for keyframe in self.keyframes:
            end = max(end, keyframe.trigger_time_ms + int(getattr(keyframe, "duration_time_ms", 0) or 0))
        return end

    @property
    def duration(self) -> float:
        """ Clip length in seconds. """
        return self.duration_ms / 1000.0

    @property
    def tracks(self) -> Dict[str, List[VectorKeyframe]]:
        tracks: Dict[str, List[VectorKeyframe]] = {}
        for keyframe in self.keyframes:
            tracks.setdefault(_TRACK_NAMES[type(keyframe)], []).append(keyframe)
        return tracks

    def keyframes_at(self, time_ms: int, tolerance: int = 0) -> List[VectorKeyframe]:
        return [keyframe for keyframe in self.keyframes
                if abs(keyframe.trigger_time_ms - time_ms) <= tolerance]

    def get(self, track: str) -> List[VectorKeyframe]:
        return list(self.tracks.get(track, []))

    @classmethod
    def from_table(cls, table: _Table) -> "VectorClip":
        name = table.string(0, "")
        keyframes_table = table.table(1)
        keyframes: List[VectorKeyframe] = []
        if keyframes_table is not None:
            for track_name, slot in _TRACK_SLOTS.items():
                cls_type = KEYFRAME_TYPES[track_name]
                for item in keyframes_table.vector_tables(slot):
                    try:
                        keyframes.append(cls_type.from_table(item))
                    except VectorAnimationError:
                        # A malformed optional track should not make the
                        # entire animation unreadable.
                        continue
        return cls(name=name, keyframes=keyframes)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VectorClip":
        name = str(_lookup(data, "Name", "name", default="") or "")
        keyframes_data = _lookup(data, "keyframes", "Keyframes", default={}) or {}
        keyframes: List[VectorKeyframe] = []
        if isinstance(keyframes_data, list):
            for item in keyframes_data:
                kind = _lookup(item, "type", "track", default="")
                if kind in KEYFRAME_TYPES:
                    keyframes.append(KEYFRAME_TYPES[kind].from_dict(item))
        elif isinstance(keyframes_data, dict):
            for track_name, cls_type in KEYFRAME_TYPES.items():
                for item in keyframes_data.get(track_name, []) or []:
                    keyframes.append(cls_type.from_dict(item))
            # Accept the short track names as well.
            short_names = {track_name[:-8]: track_name for track_name in KEYFRAME_TYPES}
            for short_name, track_name in short_names.items():
                if short_name in keyframes_data and track_name not in keyframes_data:
                    cls_type = KEYFRAME_TYPES[track_name]
                    for item in keyframes_data.get(short_name, []) or []:
                        keyframes.append(cls_type.from_dict(item))
        return cls(name=name, keyframes=keyframes)

    def to_dict(self) -> Dict[str, Any]:
        data = {"Name": self.name, "keyframes": {track: [] for track in KEYFRAME_TYPES}}
        for keyframe in self.keyframes:
            data["keyframes"][_TRACK_NAMES[type(keyframe)]].append(keyframe.to_dict())
        return data

    def to_cozmo_clip(self):
        """ Convert the common tracks to a :class:`pycozmo.anim_encoder.AnimClip`.

        Behaviors that only use head, lift, body, lights, faces, audio or
        events are playable by Cozmo after conversion.  Sprite-box tracks are
        dropped because Cozmo has no equivalent.
        """
        from .. import anim_encoder

        result = anim_encoder.AnimClip(name=self.name)
        for keyframe in self.keyframes:
            converted = _cozmo_keyframe(keyframe)
            if converted is not None:
                result.keyframes.append(converted)
        return result

    def __iter__(self) -> Iterator[VectorKeyframe]:
        return iter(self.keyframes)

    def __len__(self) -> int:
        return len(self.keyframes)

    def __repr__(self) -> str:
        return "VectorClip(name={!r}, keyframes={}, duration={:.2f}s)".format(
            self.name, len(self.keyframes), self.duration)


def _cozmo_keyframe(keyframe: VectorKeyframe):
    from .. import anim_encoder

    if isinstance(keyframe, VectorLiftHeight):
        return anim_encoder.AnimLiftHeight(
            trigger_time_ms=keyframe.trigger_time_ms,
            duration_ms=keyframe.duration_time_ms,
            height_mm=keyframe.height_mm,
            variability_mm=keyframe.height_variability_mm)
    if isinstance(keyframe, VectorHeadAngle):
        return anim_encoder.AnimHeadAngle(
            trigger_time_ms=keyframe.trigger_time_ms,
            duration_ms=keyframe.duration_time_ms,
            angle_deg=keyframe.angle_deg,
            variability_deg=keyframe.angle_variability_deg)
    if isinstance(keyframe, VectorBodyMotion):
        return anim_encoder.AnimBodyMotion(
            trigger_time_ms=keyframe.trigger_time_ms,
            duration_ms=keyframe.duration_time_ms,
            radius_mm=keyframe.radius_mm,
            speed=keyframe.speed)
    if isinstance(keyframe, VectorRecordHeading):
        return anim_encoder.AnimRecordHeading(trigger_time_ms=keyframe.trigger_time_ms)
    if isinstance(keyframe, VectorTurnToRecordedHeading):
        return anim_encoder.AnimTurnToRecordedHeading(
            trigger_time_ms=keyframe.trigger_time_ms,
            duration_ms=keyframe.duration_time_ms,
            offset_deg=keyframe.offset_deg,
            speed_deg_per_sec=keyframe.speed_deg_per_sec,
            accel_deg_per_sec_2=keyframe.accel_deg_per_sec2,
            decel_deg_per_sec_2=keyframe.decel_deg_per_sec2,
            tolerance_deg=keyframe.tolerance_deg,
            num_half_revs=keyframe.num_half_revs,
            use_shortest_dir=keyframe.use_shortest_dir)
    if isinstance(keyframe, VectorProceduralFace):
        return anim_encoder.AnimProceduralFace(
            trigger_time_ms=keyframe.trigger_time_ms,
            center_x=keyframe.face_center_x, center_y=keyframe.face_center_y,
            scale_x=keyframe.face_scale_x, scale_y=keyframe.face_scale_y,
            angle=keyframe.face_angle, left_eye=list(keyframe.left_eye),
            right_eye=list(keyframe.right_eye))
    if isinstance(keyframe, VectorFaceAnimation):
        return anim_encoder.AnimFaceAnimation(
            trigger_time_ms=keyframe.trigger_time_ms, anim_name=keyframe.anim_name)
    if isinstance(keyframe, VectorEvent):
        return anim_encoder.AnimEvent(
            trigger_time_ms=keyframe.trigger_time_ms, event_id=keyframe.event_id)
    if isinstance(keyframe, VectorBackpackLights):
        # Vector has three lights; Cozmo has five.  Use the outer Vector
        # lights for Cozmo's left/right LEDs so converted animations still
        # produce a smooth gradient.
        front = anim_encoder.AnimLight(*keyframe.front_rgb)
        middle = anim_encoder.AnimLight(*keyframe.middle_rgb)
        back = anim_encoder.AnimLight(*keyframe.back_rgb)
        return anim_encoder.AnimBackpackLights(
            trigger_time_ms=keyframe.trigger_time_ms,
            duration_ms=keyframe.duration_time_ms,
            left=front, front=front, middle=middle, back=back, right=back)
    if isinstance(keyframe, VectorRobotAudio):
        # Cozmo's audio track is a list of event IDs rather than Vector's
        # weighted event groups.  Represent the first group when possible.
        if keyframe.event_groups:
            group = keyframe.event_groups[0]
            return anim_encoder.AnimRobotAudio(
                trigger_time_ms=keyframe.trigger_time_ms,
                audio_event_ids=list(group.event_ids),
                volume=group.volumes[0] if group.volumes else 1.0,
                probabilities=list(group.probabilities),
                has_alts=len(keyframe.event_groups) > 1)
        return None
    return None


@dataclass
class VectorAnimations:
    """ A Vector ``AnimClips`` file, containing one or more clips. """

    clips: List[VectorClip] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Keep the file order: animation names are not necessarily unique in
        # community exports and playback code often uses the first clip.
        self.clips = list(self.clips)

    @classmethod
    def from_fb_bytes(cls, data: Union[bytes, bytearray, memoryview]) -> "VectorAnimations":
        if not data:
            raise VectorAnimationError("Empty animation file.")
        reader = _FlatBuffer(data)
        try:
            root_offset = reader.u32(0)
            root = reader.table(root_offset)
            clips = [VectorClip.from_table(item) for item in root.vector_tables(0)]
        except (VectorAnimationError, struct.error, IndexError) as exc:
            raise VectorAnimationError("Failed to parse Vector animation: {}".format(exc)) from exc
        if not clips:
            raise VectorAnimationError("Animation file contains no clips.")
        return cls(clips=clips)

    @classmethod
    def from_fb_file(cls, filename: Union[str, os.PathLike]) -> "VectorAnimations":
        with open(filename, "rb") as stream:
            return cls.from_fb_bytes(stream.read())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VectorAnimations":
        clips = (_lookup(data, "clips", "Clips", default=[]) if isinstance(data, dict) else [])
        return cls(clips=[VectorClip.from_dict(item) for item in clips])

    @classmethod
    def from_json_file(cls, filename: Union[str, os.PathLike]) -> "VectorAnimations":
        with open(filename, "r", encoding="utf-8") as stream:
            return cls.from_dict(json.load(stream))

    def to_dict(self) -> Dict[str, Any]:
        return {"clips": [clip.to_dict() for clip in self.clips]}

    def to_json_file(self, filename: Union[str, os.PathLike]) -> None:
        with open(filename, "w", encoding="utf-8") as stream:
            json.dump(self.to_dict(), stream, indent=2)

    @property
    def by_name(self) -> Dict[str, VectorClip]:
        return {clip.name: clip for clip in self.clips}

    def get(self, name: str, default: Any = None) -> Any:
        return self.by_name.get(name, default)

    def __getitem__(self, name: Union[int, str]) -> VectorClip:
        if isinstance(name, int):
            return self.clips[name]
        return self.by_name[name]

    def __contains__(self, name: object) -> bool:
        return name in self.by_name

    def __iter__(self) -> Iterator[VectorClip]:
        return iter(self.clips)

    def __len__(self) -> int:
        return len(self.clips)

    @property
    def names(self) -> List[str]:
        return [clip.name for clip in self.clips]

    def to_cozmo_clips(self):
        from .. import anim_encoder
        return anim_encoder.AnimClips([clip.to_cozmo_clip() for clip in self.clips])


#: Common alias used in user code.
VectorAnimClips = VectorAnimations


def cozmo_clips_to_vector(clips: Any) -> VectorAnimations:
    """ Convert Cozmo :class:`~pycozmo.anim_encoder.AnimClips` to Vector clips. """
    result: List[VectorClip] = []
    for source in getattr(clips, "clips", clips):
        target = VectorClip(name=getattr(source, "name", "clip"))
        for keyframe in getattr(source, "keyframes", []):
            converted = _vector_keyframe(keyframe)
            if converted is not None:
                target.keyframes.append(converted)
        target.keyframes.sort(key=lambda item: item.trigger_time_ms)
        result.append(target)
    return VectorAnimations(clips=result)


def _vector_keyframe(keyframe: Any) -> Optional[VectorKeyframe]:
    from .. import anim_encoder

    if isinstance(keyframe, anim_encoder.AnimLiftHeight):
        return VectorLiftHeight(
            trigger_time_ms=keyframe.trigger_time_ms,
            duration_time_ms=keyframe.duration_ms,
            height_mm=keyframe.height_mm,
            height_variability_mm=keyframe.variability_mm)
    if isinstance(keyframe, anim_encoder.AnimHeadAngle):
        return VectorHeadAngle(
            trigger_time_ms=keyframe.trigger_time_ms,
            duration_time_ms=keyframe.duration_ms,
            angle_deg=keyframe.angle_deg,
            angle_variability_deg=keyframe.variability_deg)
    if isinstance(keyframe, anim_encoder.AnimBodyMotion):
        return VectorBodyMotion(
            trigger_time_ms=keyframe.trigger_time_ms,
            duration_time_ms=keyframe.duration_ms,
            radius_mm=keyframe.radius_mm,
            speed=keyframe.speed)
    if isinstance(keyframe, anim_encoder.AnimRecordHeading):
        return VectorRecordHeading(trigger_time_ms=keyframe.trigger_time_ms)
    if isinstance(keyframe, anim_encoder.AnimTurnToRecordedHeading):
        return VectorTurnToRecordedHeading(
            trigger_time_ms=keyframe.trigger_time_ms,
            duration_time_ms=keyframe.duration_ms,
            offset_deg=keyframe.offset_deg,
            speed_deg_per_sec=keyframe.speed_deg_per_sec,
            accel_deg_per_sec2=keyframe.accel_deg_per_sec_2,
            decel_deg_per_sec2=keyframe.decel_deg_per_sec_2,
            tolerance_deg=keyframe.tolerance_deg,
            num_half_revs=keyframe.num_half_revs,
            use_shortest_dir=keyframe.use_shortest_dir)
    if isinstance(keyframe, anim_encoder.AnimProceduralFace):
        return VectorProceduralFace(
            trigger_time_ms=keyframe.trigger_time_ms,
            face_angle=keyframe.angle,
            face_center_x=keyframe.center_x,
            face_center_y=keyframe.center_y,
            face_scale_x=keyframe.scale_x,
            face_scale_y=keyframe.scale_y,
            left_eye=list(keyframe.left_eye),
            right_eye=list(keyframe.right_eye))
    if isinstance(keyframe, anim_encoder.AnimFaceAnimation):
        return VectorFaceAnimation(trigger_time_ms=keyframe.trigger_time_ms,
                                   anim_name=keyframe.anim_name)
    if isinstance(keyframe, anim_encoder.AnimEvent):
        return VectorEvent(trigger_time_ms=keyframe.trigger_time_ms,
                           event_id=keyframe.event_id)
    if isinstance(keyframe, anim_encoder.AnimBackpackLights):
        def rgb(light: Any) -> List[float]:
            return [light.red / 255.0, light.green / 255.0, light.blue / 255.0]

        return VectorBackpackLights(
            trigger_time_ms=keyframe.trigger_time_ms,
            duration_time_ms=keyframe.duration_ms,
            front=rgb(keyframe.front),
            middle=rgb(keyframe.middle),
            back=rgb(keyframe.back))
    if isinstance(keyframe, anim_encoder.AnimRobotAudio):
        event_ids = list(keyframe.audio_event_ids)
        probabilities = list(keyframe.probabilities)
        if len(probabilities) != len(event_ids):
            probabilities = []
        return VectorRobotAudio(
            trigger_time_ms=keyframe.trigger_time_ms,
            event_groups=[VectorAudioEventGroup(
                event_ids=event_ids,
                volumes=[float(keyframe.volume)] * len(event_ids),
                probabilities=probabilities)])
    return None


def is_vector_animation(data: Union[bytes, bytearray, memoryview, str, os.PathLike]) -> bool:
    """ Return whether ``data`` looks like a Vector (rather than Cozmo) file.

    The root envelope is shared, so this checks for Vector-only tracks.  It
    returns ``False`` for unreadable files rather than raising.
    """
    try:
        if isinstance(data, (str, os.PathLike)):
            with open(data, "rb") as stream:
                raw = stream.read()
        else:
            raw = data
        if not raw:
            return False
        reader = _FlatBuffer(raw)
        root = reader.table(reader.u32(0))
        for clip_table in root.vector_tables(0):
            keyframes = clip_table.table(1)
            if keyframes is None:
                continue
            if keyframes.field(_TRACK_SLOTS["SpriteBoxKeyFrame"]) is not None:
                return True
            # scanlineOpacity is slot 8 of ProceduralFace and does not exist
            # in the original Cozmo schema.
            for face_table in keyframes.vector_tables(_TRACK_SLOTS["ProceduralFaceKeyFrame"]):
                if face_table.field(8) is not None:
                    return True
        return False
    except Exception:
        return False


def load_vector_animations(source: Union[str, os.PathLike, bytes, bytearray, memoryview],
                           ) -> VectorAnimations:
    """ Load from a path, a FlatBuffers byte string, or a JSON string. """
    if isinstance(source, (bytes, bytearray, memoryview)):
        raw = bytes(source)
        if raw.lstrip().startswith(b"{"):
            return VectorAnimations.from_dict(json.loads(raw.decode("utf-8")))
        return VectorAnimations.from_fb_bytes(raw)
    if isinstance(source, (str, os.PathLike)):
        path = os.fspath(source)
        if path.lower().endswith(".json"):
            return VectorAnimations.from_json_file(path)
        return VectorAnimations.from_fb_file(path)
    raise TypeError("Unsupported animation source: {}".format(type(source).__name__))


# ---------------------------------------------------------------------------
# Playback
# ---------------------------------------------------------------------------

@dataclass
class VectorFrame:
    """ Snapshot of the animation tracks at one point in time. """

    time_ms: int
    keyframes: List[VectorKeyframe] = field(default_factory=list)
    lift_height_mm: Optional[int] = None
    head_angle_deg: Optional[int] = None
    body_motion: Optional[VectorBodyMotion] = None
    backpack_lights: Optional[VectorBackpackLights] = None
    face_animation: Optional[VectorFaceAnimation] = None
    procedural_face: Optional[VectorProceduralFace] = None
    sprite_boxes: List[VectorSpriteBox] = field(default_factory=list)
    audio: List[VectorRobotAudio] = field(default_factory=list)
    events: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "time_ms": self.time_ms,
            "lift_height_mm": self.lift_height_mm,
            "head_angle_deg": self.head_angle_deg,
            "backpack_lights": self.backpack_lights.to_dict() if self.backpack_lights else None,
            "face_animation": self.face_animation.anim_name if self.face_animation else None,
            "sprite_boxes": [item.to_dict() for item in self.sprite_boxes],
            "events": list(self.events),
        }


class VectorAnimator:
    """ Turn a :class:`VectorClip` into timed frames and optional playback.

    The animator does not import a Vector SDK.  When a ``client`` is given,
    methods with Cozmo/Vector-like names are called for the tracks it
    supports.  For anything else pass a ``handlers`` dictionary::

        VectorAnimator(clip).play(handlers={
            "sprite": lambda keyframe: ...,
            "audio": lambda keyframe: ...,
            "event": lambda event_id: ...,
        })

    Supported handler keys: ``lift``, ``head``, ``body``, ``lights``,
    ``face``, ``procedural_face``, ``sprite``, ``audio``, ``event`` and
    ``keyframe``.
    """

    def __init__(self, clip: VectorClip, rng: Optional[random.Random] = None) -> None:
        self.clip = clip
        self.rng = rng or random.Random()
        self._stop = threading.Event()

    def iter_frames(self) -> Iterator[VectorFrame]:
        """ Yield one frame for each distinct keyframe timestamp. """
        lift: Optional[int] = None
        head: Optional[int] = None
        lights: Optional[VectorBackpackLights] = None
        face: Optional[VectorFaceAnimation] = None
        procedural: Optional[VectorProceduralFace] = None
        sprites: Dict[str, VectorSpriteBox] = {}
        grouped: Dict[int, List[VectorKeyframe]] = {}
        for keyframe in self.clip.keyframes:
            grouped.setdefault(keyframe.trigger_time_ms, []).append(keyframe)
        for timestamp in sorted(grouped):
            body = None
            audio: List[VectorRobotAudio] = []
            events: List[str] = []
            frame_sprites: List[VectorSpriteBox] = []
            for keyframe in grouped[timestamp]:
                if isinstance(keyframe, VectorLiftHeight):
                    lift = keyframe.height_mm
                elif isinstance(keyframe, VectorHeadAngle):
                    head = keyframe.angle_deg
                elif isinstance(keyframe, VectorBodyMotion):
                    body = keyframe
                elif isinstance(keyframe, VectorBackpackLights):
                    lights = keyframe
                elif isinstance(keyframe, VectorFaceAnimation):
                    face = keyframe
                elif isinstance(keyframe, VectorProceduralFace):
                    procedural = keyframe
                elif isinstance(keyframe, VectorSpriteBox):
                    sprites[keyframe.sprite_box_name] = keyframe
                    frame_sprites.append(keyframe)
                elif isinstance(keyframe, VectorRobotAudio):
                    audio.append(keyframe)
                elif isinstance(keyframe, VectorEvent) and keyframe.event_id:
                    events.append(keyframe.event_id)
            yield VectorFrame(
                time_ms=timestamp, keyframes=list(grouped[timestamp]),
                lift_height_mm=lift, head_angle_deg=head, body_motion=body,
                backpack_lights=lights, face_animation=face,
                procedural_face=procedural, sprite_boxes=frame_sprites,
                audio=audio, events=events)

    def _dispatch(self, keyframe: VectorKeyframe, client: Any,
                  handlers: Dict[str, Callable[[Any], Any]]) -> None:
        callback = handlers.get("keyframe")
        if callback:
            callback(keyframe)
        if isinstance(keyframe, VectorLiftHeight):
            handler = handlers.get("lift")
            if handler:
                handler(keyframe)
            elif client is not None and hasattr(client, "set_lift_height"):
                client.set_lift_height(keyframe.height_mm)
        elif isinstance(keyframe, VectorHeadAngle):
            handler = handlers.get("head")
            if handler:
                handler(keyframe)
            elif client is not None and hasattr(client, "set_head_angle"):
                client.set_head_angle(keyframe.angle_rad)
        elif isinstance(keyframe, VectorBodyMotion):
            handler = handlers.get("body")
            if handler:
                handler(keyframe)
            elif client is not None:
                self._drive(client, keyframe)
        elif isinstance(keyframe, VectorBackpackLights):
            handler = handlers.get("lights")
            if handler:
                handler(keyframe)
            elif client is not None and hasattr(client, "set_backpack_lights"):
                client.set_backpack_lights(keyframe.front_rgb,
                                           keyframe.middle_rgb, keyframe.back_rgb)
        elif isinstance(keyframe, VectorFaceAnimation):
            handler = handlers.get("face")
            if handler:
                handler(keyframe)
            elif client is not None and hasattr(client, "play_face_animation"):
                client.play_face_animation(keyframe.anim_name)
        elif isinstance(keyframe, VectorProceduralFace):
            handler = handlers.get("procedural_face")
            if handler:
                handler(keyframe)
        elif isinstance(keyframe, VectorSpriteBox):
            handler = handlers.get("sprite")
            if handler:
                handler(keyframe)
        elif isinstance(keyframe, VectorRobotAudio):
            handler = handlers.get("audio")
            if handler:
                handler(keyframe)
            elif client is not None:
                selected = keyframe.choose_event(self.rng)
                if selected and hasattr(client, "play_audio_event"):
                    client.play_audio_event(selected[0], volume=selected[1])
        elif isinstance(keyframe, VectorEvent):
            handler = handlers.get("event")
            if handler:
                handler(keyframe.event_id)
            elif client is not None and hasattr(client, "trigger_event"):
                client.trigger_event(keyframe.event_id)

    @staticmethod
    def _drive(client: Any, motion: VectorBodyMotion) -> None:
        seconds = motion.duration_time_ms / 1000.0
        if motion.is_straight and hasattr(client, "drive_straight"):
            client.drive_straight(motion.speed, duration=seconds)
        elif motion.is_turn_in_place and hasattr(client, "turn_in_place_at_speed"):
            client.turn_in_place_at_speed(abs(motion.speed),
                                          direction=1 if motion.speed >= 0 else -1)
            if seconds > 0:
                time.sleep(seconds)
                if hasattr(client, "stop_all_motors"):
                    client.stop_all_motors()
        else:
            try:
                radius = float(motion.radius_mm)
            except (TypeError, ValueError):
                radius = None
            if radius is not None and hasattr(client, "drive_wheels"):
                track_width = 45.0
                left = motion.speed * (radius - track_width / 2.0)
                right = motion.speed * (radius + track_width / 2.0)
                client.drive_wheels(left, right)
                if seconds > 0:
                    time.sleep(seconds)
                    if hasattr(client, "stop_all_motors"):
                        client.stop_all_motors()
            elif hasattr(client, "drive_wheels"):
                client.drive_wheels(motion.speed, motion.speed, duration=seconds)

    def play(self, client: Any = None,
             handlers: Optional[Dict[str, Callable[[Any], Any]]] = None,
             speed: float = 1.0,
             sleep: Callable[[float], None] = time.sleep,
             stop_event: Optional[threading.Event] = None) -> None:
        """ Play the clip synchronously.

        ``speed`` scales time (``2.0`` is twice as fast).  A custom ``sleep``
        function can be supplied for deterministic tests.
        """
        speed = float(speed)
        if speed <= 0:
            raise ValueError("Playback speed must be positive.")
        handlers = dict(handlers or {})
        stop_event = stop_event or self._stop
        previous = 0
        for frame in self.iter_frames():
            if stop_event.is_set():
                break
            delay = (frame.time_ms - previous) / 1000.0 / speed
            if delay > 0:
                sleep(delay)
            previous = frame.time_ms
            for keyframe in frame.keyframes:
                self._dispatch(keyframe, client, handlers)
        if stop_event is self._stop:
            self._stop.clear()

    def play_async(self, client: Any = None,
                   handlers: Optional[Dict[str, Callable[[Any], Any]]] = None,
                   speed: float = 1.0) -> threading.Thread:
        thread = threading.Thread(target=self.play, kwargs={
            "client": client, "handlers": handlers, "speed": speed,
            "stop_event": self._stop,
        }, daemon=True, name="VectorAnimator")
        thread.start()
        return thread

    def stop(self) -> None:
        self._stop.set()


def load_and_play(source: Union[str, os.PathLike, bytes, bytearray, memoryview],
                  clip_name: Optional[str] = None, client: Any = None,
                  handlers: Optional[Dict[str, Callable[[Any], Any]]] = None,
                  **kwargs: Any) -> None:
    """ Load an animation file and play one clip through a client/handlers. """
    animations = load_vector_animations(source)
    clip = animations.get(clip_name) if clip_name else animations.clips[0]
    if clip is None:
        raise KeyError("Animation clip {!r} not found.".format(clip_name))
    VectorAnimator(clip).play(client=client, handlers=handlers, **kwargs)
