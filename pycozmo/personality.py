"""

Personality engine.

The engine combines three simple ideas:

* **Traits** are stable preferences in the range 0..1 (playfulness,
  sociability, curiosity, ...).
* **Drives** are slowly changing needs in the range 0..1.  A value near 1
  means the need is urgent; running a matching behavior satisfies it.
* **Emotions** are short-lived values that decay back to a baseline.

:class:`Personality` turns those values into scores for activities and
behaviors, reacts to robot events, and can be serialized to JSON.  It has no
robot dependency and is therefore easy to test and to reuse with Vector.

"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
import random
import time
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple, Union


__all__ = [
    "TRAIT_NAMES",
    "EMOTION_NAMES",
    "DRIVE_NAMES",
    "DEFAULT_TRAITS",
    "DEFAULT_EMOTIONS",
    "DEFAULT_DRIVES",
    "Drive",
    "Personality",
]


TRAIT_NAMES = (
    "playfulness",
    "sociability",
    "curiosity",
    "bravery",
    "energy",
    "patience",
    "expressiveness",
    "independence",
)

EMOTION_NAMES = (
    "happy",
    "excited",
    "calm",
    "social",
    "brave",
    "curious",
    "frustrated",
    "anxious",
    "tired",
)

DRIVE_NAMES = (
    "social",
    "play",
    "curiosity",
    "rest",
    "safety",
)

DEFAULT_TRAITS: Dict[str, float] = {
    "playfulness": 0.70,
    "sociability": 0.60,
    "curiosity": 0.65,
    "bravery": 0.50,
    "energy": 0.65,
    "patience": 0.55,
    "expressiveness": 0.70,
    "independence": 0.40,
}

DEFAULT_EMOTIONS: Dict[str, float] = {
    "happy": 0.55,
    "excited": 0.25,
    "calm": 0.60,
    "social": 0.35,
    "brave": 0.40,
    "curious": 0.55,
    "frustrated": 0.00,
    "anxious": 0.10,
    "tired": 0.20,
}

DEFAULT_DRIVES: Dict[str, Tuple[float, float]] = {
    # name: (initial value, growth per second)
    "social": (0.25, 0.015),
    "play": (0.35, 0.025),
    "curiosity": (0.30, 0.020),
    "rest": (0.15, 0.010),
    "safety": (0.05, 0.030),
}

DEFAULT_EMOTION_BASELINES: Dict[str, float] = dict(DEFAULT_EMOTIONS)
DEFAULT_EMOTION_BASELINE = 0.0
DEFAULT_EMOTION_DECAY = 0.03


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


@dataclass
class Drive:
    """ A slowly changing need. """

    name: str
    value: float = 0.0
    growth_rate: float = 0.0
    low: float = 0.0
    high: float = 1.0

    def __post_init__(self) -> None:
        self.name = str(self.name)
        self.value = _clamp(self.value, self.low, self.high)
        self.growth_rate = max(0.0, _number(self.growth_rate))

    def update(self, dt: float) -> float:
        """ Increase urgency by ``growth_rate * dt`` seconds. """
        if dt > 0:
            self.value = _clamp(self.value + self.growth_rate * dt, self.low, self.high)
        return self.value

    def satisfy(self, amount: float) -> float:
        """ Reduce urgency.  Negative amounts increase it. """
        self.value = _clamp(self.value - float(amount), self.low, self.high)
        return self.value

    def frustrate(self, amount: float) -> float:
        return self.satisfy(-float(amount))

    def to_dict(self) -> Dict[str, float]:
        return {
            "value": self.value,
            "growth_rate": self.growth_rate,
        }

    @classmethod
    def from_dict(cls, name: str, data: Any) -> "Drive":
        if isinstance(data, Mapping):
            return cls(name=name,
                       value=data.get("value", 0.0),
                       growth_rate=data.get("growth_rate", 0.0))
        return cls(name=name, value=_number(data))


class Personality:
    """ Traits, drives and emotions, plus activity/behavior scoring.

    Args:
        traits: Trait overrides.  Unknown traits are kept as well, which
            allows applications to define their own dimensions.
        drives: Mapping from drive name to either a value or a mapping with
            ``value`` and ``growth_rate``.
        emotions: Initial emotion values.  Missing emotions use the values in
            :data:`DEFAULT_EMOTIONS`.
        baselines: Values emotions decay towards.
        emotion_decay: Fraction of the distance to the baseline removed per
            second.  ``0.03`` means about 3% per second.
        activity_preferences: Mapping from activity name to extra preference.
        behavior_preferences: Mapping from behavior name/prefix to extra
            preference.
        event_responses: Mapping from event name to a response dictionary.
        rng: Optional :class:`random.Random` instance for reproducible choices.
    """

    def __init__(self,
                 traits: Optional[Mapping[str, float]] = None,
                 drives: Optional[Mapping[str, Any]] = None,
                 emotions: Optional[Mapping[str, float]] = None,
                 baselines: Optional[Mapping[str, float]] = None,
                 emotion_decay: float = DEFAULT_EMOTION_DECAY,
                 activity_preferences: Optional[Mapping[str, float]] = None,
                 behavior_preferences: Optional[Mapping[str, float]] = None,
                 event_responses: Optional[Mapping[str, Mapping[str, Any]]] = None,
                 rng: Optional[random.Random] = None) -> None:
        self.traits: Dict[str, float] = dict(DEFAULT_TRAITS)
        if traits:
            for name, value in traits.items():
                self.traits[str(name)] = _clamp(_number(value, self.traits.get(str(name), 0.5)))
        self.drives: Dict[str, Drive] = {}
        for name, (value, growth) in DEFAULT_DRIVES.items():
            self.drives[name] = Drive(name=name, value=value, growth_rate=growth)
        if drives:
            for name, data in drives.items():
                name = str(name)
                existing = self.drives.get(name)
                if isinstance(data, Mapping):
                    self.drives[name] = Drive(
                        name=name,
                        value=data.get("value", existing.value if existing else 0.0),
                        growth_rate=data.get("growth_rate",
                                             existing.growth_rate if existing else 0.0))
                elif existing is not None:
                    existing.value = _clamp(_number(data))
                else:
                    self.drives[name] = Drive(name=name, value=_number(data))
        self.emotions: Dict[str, float] = dict(DEFAULT_EMOTIONS)
        if emotions:
            for name, value in emotions.items():
                self.emotions[str(name)] = _clamp(_number(value))
        self.baselines: Dict[str, float] = {}
        for name in self.emotions:
            default = DEFAULT_EMOTION_BASELINES.get(name, DEFAULT_EMOTION_BASELINE)
            self.baselines[name] = _clamp(_number((baselines or {}).get(name, default)))
        if baselines:
            for name, value in baselines.items():
                self.baselines[str(name)] = _clamp(_number(value))
                self.emotions.setdefault(str(name), self.baselines[str(name)])
        self.emotion_decay = max(0.0, _number(emotion_decay, DEFAULT_EMOTION_DECAY))
        self.activity_preferences = {str(key): _number(value)
                                     for key, value in (activity_preferences or {}).items()}
        self.behavior_preferences = {str(key).lower(): _number(value)
                                     for key, value in (behavior_preferences or {}).items()}
        self.event_responses: Dict[str, Dict[str, Dict[str, float]]] = {}
        self.event_responses.update(self._default_event_responses())
        for name, response in (event_responses or {}).items():
            self.event_responses[str(name)] = {
                "emotions": dict(response.get("emotions", {})),
                "drives": dict(response.get("drives", {})),
            }
        self.rng = rng or random.Random()
        self.last_update = time.monotonic()

    # ------------------------------------------------------------------
    # State updates
    # ------------------------------------------------------------------

    def update(self, dt: Optional[float] = None) -> None:
        """ Advance drive and emotion decay by ``dt`` seconds.

        When ``dt`` is omitted, the time elapsed since the previous call is
        used.  Negative values are ignored.
        """
        now = time.monotonic()
        if dt is None:
            dt = now - self.last_update
        self.last_update = now
        dt = _number(dt)
        if dt <= 0:
            return
        for drive in self.drives.values():
            drive.update(dt)
        factor = min(1.0, self.emotion_decay * dt)
        for name, value in list(self.emotions.items()):
            baseline = self.baselines.get(name, DEFAULT_EMOTION_BASELINE)
            self.emotions[name] = _clamp(value + (baseline - value) * factor)

    def feel(self, emotion: str, amount: float) -> float:
        """ Add ``amount`` to an emotion and return the new value. """
        name = str(emotion)
        self.emotions[name] = _clamp(self.emotions.get(name, 0.0) + float(amount))
        return self.emotions[name]

    def satisfy(self, drive: str, amount: float) -> float:
        """ Satisfy a drive (reduce its urgency) and return the new value. """
        name = str(drive)
        if name not in self.drives:
            self.drives[name] = Drive(name=name, value=0.0)
        return self.drives[name].satisfy(amount)

    def apply_affectors(self, affectors: Mapping[str, float]) -> None:
        """ Apply a flat mapping of emotion/drive names to deltas. """
        for name, value in affectors.items():
            name = str(name)
            if name in self.drives:
                self.drives[name].frustrate(value)
            else:
                self.feel(name, value)

    def react(self, event_name: str, strength: float = 1.0) -> bool:
        """ Apply the response configured for ``event_name``.

        Matching is case-insensitive and accepts ``Evt``-prefixed and
        ``Change``-suffixed event names.  Returns whether a response matched.
        """
        response = self._lookup_response(event_name)
        if response is None:
            return False
        strength = max(0.0, _number(strength, 1.0))
        for name, value in response.get("emotions", {}).items():
            self.feel(name, _number(value) * strength)
        for name, value in response.get("drives", {}).items():
            drive = self.drives.get(str(name))
            if drive is None:
                drive = Drive(name=str(name), value=0.0)
                self.drives[str(name)] = drive
            # Positive values in an event response increase urgency.
            drive.frustrate(_number(value) * strength)
        return True

    def _lookup_response(self, event_name: str) -> Optional[Dict[str, Dict[str, float]]]:
        name = str(event_name)
        if name in self.event_responses:
            return self.event_responses[name]
        base = name[3:] if name.startswith("Evt") else name
        candidates = {name.lower(), base.lower()}
        if base.endswith("Change"):
            candidates.add(base[:-6].lower())
        for key, response in self.event_responses.items():
            if key.lower() in candidates:
                return response
        return None

    # ------------------------------------------------------------------
    # Derived state
    # ------------------------------------------------------------------

    @property
    def mood(self) -> float:
        """ A simple average of the positive and negative emotions. """
        positive = ("happy", "excited", "calm", "social", "brave", "curious")
        negative = ("frustrated", "anxious", "tired")
        pos = sum(self.emotions.get(name, 0.0) for name in positive) / len(positive)
        neg = sum(self.emotions.get(name, 0.0) for name in negative) / len(negative)
        return _clamp(0.5 + (pos - neg) * 0.5)

    @property
    def dominant_drive(self) -> Tuple[str, float]:
        if not self.drives:
            return "", 0.0
        name = max(self.drives, key=lambda key: self.drives[key].value)
        return name, self.drives[name].value

    @property
    def dominant_emotion(self) -> Tuple[str, float]:
        if not self.emotions:
            return "", 0.0
        name = max(self.emotions, key=lambda key: self.emotions[key])
        return name, self.emotions[name]

    # ------------------------------------------------------------------
    # Selection and scoring
    # ------------------------------------------------------------------

    def score_activity(self, activity: Any, context: Optional[Mapping[str, Any]] = None) -> float:
        """ Score an activity object or name. """
        name = self._activity_name(activity)
        context = dict(context or {})
        traits = self.traits
        drives = {key: drive.value for key, drive in self.drives.items()}
        face = bool(context.get("face_visible", context.get("faces", False)))
        cube = bool(context.get("cube_visible", context.get("cubes", False)))
        battery_low = bool(context.get("battery_low", False))
        on_charger = bool(context.get("on_charger", False))
        picked_up = bool(context.get("picked_up", False))

        key = name.lower().replace("_", "").replace("-", "").replace(" ", "")
        score = 0.15
        if "freeplay" in key or "play" in key:
            score += drives.get("play", 0.0) * 1.4
            score += drives.get("curiosity", 0.0) * 1.0
            score += traits.get("playfulness", 0.5) * 0.8
            score += traits.get("energy", 0.5) * 0.25
            score += 0.5 if cube else 0.0
        elif "social" in key:
            score += drives.get("social", 0.0) * 1.6
            score += traits.get("sociability", 0.5) * 1.0
            score += 0.6 if face else 0.0
            score += 0.2 * self.emotions.get("social", 0.0)
        elif "feed" in key or "food" in key:
            score += drives.get("play", 0.0) * 0.5
            score += drives.get("curiosity", 0.0) * 0.8
            score += 0.4 if cube else 0.0
        elif "need" in key or "sleep" in key or "rest" in key:
            score += drives.get("rest", 0.0) * 1.6
            score += drives.get("safety", 0.0) * 0.3
            score += traits.get("expressiveness", 0.5) * 0.35
            score += scalescore(battery_low, 1.0)
        elif "pyramid" in key or "build" in key:
            score += drives.get("play", 0.0) * 0.8
            score += drives.get("curiosity", 0.0) * 0.9
            score += traits.get("patience", 0.5) * 0.6
            score += 0.6 if cube else 0.0
        elif "spark" in key:
            score += drives.get("play", 0.0) * 1.0
            score += traits.get("bravery", 0.5) * 0.7
            score += self.emotions.get("excited", 0.0) * 0.5
        elif "voice" in key or "command" in key:
            score += traits.get("sociability", 0.5) * 0.9
            score += drives.get("social", 0.0) * 0.5
        else:
            score += 0.25 * self.mood

        if on_charger:
            # Being on the charger is a good moment for calm activities.
            if "freeplay" in key or "social" in key or "feed" in key:
                score += context.get("charger_bonus", 0.15)
        if picked_up and ("need" in key or "social" in key):
            score += 0.2
        if context.get("quiet"):
            score -= traits.get("playfulness", 0.5) * 0.25
        score += self.activity_preferences.get(name, 0.0)
        return max(0.0, float(score))

    def score_behavior(self, behavior: Any, context: Optional[Mapping[str, Any]] = None) -> float:
        """ Score a behavior object or ID from its name and the current state. """
        behavior_id = getattr(behavior, "get_id", None)
        if callable(behavior_id):
            behavior_id = behavior_id()
        name = str(behavior_id if behavior_id is not None else behavior)
        context = dict(context or {})
        key = name.lower()
        traits = self.traits
        drives = {drive_name: drive.value for drive_name, drive in self.drives.items()}
        score = 0.1
        play_words = ("play", "dance", "roll", "pop", "wiggle", "anim", "trick", "celebrate")
        social_words = ("hello", "greet", "social", "lookat", "look_at", "happy", "smile", "name")
        explore_words = ("explore", "curious", "look", "scan", "wander", "discover")
        rest_words = ("sleep", "rest", "tired", "settle", "calm", "idle")
        safety_words = ("cliff", "obstacle", "falling", "pickup", "picked", "block", "danger", "react")
        if any(word in key for word in play_words):
            score += drives.get("play", 0.0) * 1.2 + traits.get("playfulness", 0.5) * 0.8
        if any(word in key for word in social_words):
            score += drives.get("social", 0.0) * 1.2 + traits.get("sociability", 0.5) * 0.8
            score += 0.3 if context.get("face_visible") else 0.0
        if any(word in key for word in explore_words):
            score += drives.get("curiosity", 0.0) * 1.3 + traits.get("curiosity", 0.5) * 0.5
        if any(word in key for word in rest_words):
            score += drives.get("rest", 0.0) * 1.4 + traits.get("patience", 0.5) * 0.2
        if any(word in key for word in safety_words):
            score += drives.get("safety", 0.0) * 1.8 + traits.get("bravery", 0.5) * 0.1
        if context.get("cube_visible") and any(word in key for word in ("cube", "block", "build")):
            score += 0.4
        if context.get("battery_low") and any(word in key for word in ("charge", "rest", "sleep")):
            score += 0.8
        for prefix, bonus in self.behavior_preferences.items():
            if prefix and prefix in key:
                score += bonus
        return max(0.0, float(score))

    def choose_activity(self,
                        activities: Iterable[Any],
                        context: Optional[Mapping[str, Any]] = None,
                        rng: Optional[random.Random] = None,
                        temperature: float = 1.0) -> Optional[Any]:
        """ Choose an activity object/name using weighted sampling. """
        return self._choose(activities, self.score_activity, context,
                            rng=rng, temperature=temperature)

    def choose_behavior(self,
                        behaviors: Iterable[Any],
                        context: Optional[Mapping[str, Any]] = None,
                        rng: Optional[random.Random] = None,
                        temperature: float = 1.0) -> Optional[Any]:
        """ Choose a behavior object/name using weighted sampling. """
        return self._choose(behaviors, self.score_behavior, context,
                            rng=rng, temperature=temperature)

    def _choose(self,
                candidates: Iterable[Any],
                scorer: Any,
                context: Optional[Mapping[str, Any]],
                rng: Optional[random.Random],
                temperature: float) -> Optional[Any]:
        items = list(candidates)
        if not items:
            return None
        scores = [max(1e-6, float(scorer(item, context))) for item in items]
        temperature = _number(temperature, 1.0)
        if temperature <= 0:
            return items[max(range(len(items)), key=lambda index: scores[index])]
        rng = rng or self.rng
        highest = max(scores)
        weights = [math.exp((score - highest) / temperature) for score in scores]
        total = sum(weights)
        pick = rng.random() * total
        cumulative = 0.0
        for item, weight in zip(items, weights):
            cumulative += weight
            if pick <= cumulative:
                return item
        return items[-1]

    # ------------------------------------------------------------------
    # Serialization and presets
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "traits": dict(self.traits),
            "drives": {name: drive.to_dict() for name, drive in self.drives.items()},
            "emotions": dict(self.emotions),
            "baselines": dict(self.baselines),
            "emotion_decay": self.emotion_decay,
            "activity_preferences": dict(self.activity_preferences),
            "behavior_preferences": dict(self.behavior_preferences),
            "event_responses": {
                name: {"emotions": dict(response.get("emotions", {})),
                       "drives": dict(response.get("drives", {}))}
                for name, response in self.event_responses.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Personality":
        return cls(
            traits=data.get("traits"),
            drives=data.get("drives"),
            emotions=data.get("emotions"),
            baselines=data.get("baselines"),
            emotion_decay=data.get("emotion_decay", DEFAULT_EMOTION_DECAY),
            activity_preferences=data.get("activity_preferences"),
            behavior_preferences=data.get("behavior_preferences"),
            event_responses=data.get("event_responses"),
        )

    def save(self, filename: Union[str, os.PathLike]) -> None:
        with open(filename, "w", encoding="utf-8") as stream:
            json.dump(self.to_dict(), stream, indent=2)

    @classmethod
    def load(cls, filename: Union[str, os.PathLike]) -> "Personality":
        with open(filename, "r", encoding="utf-8") as stream:
            return cls.from_dict(json.load(stream))

    @classmethod
    def default(cls, **kwargs: Any) -> "Personality":
        return cls(**kwargs)

    @classmethod
    def playful(cls) -> "Personality":
        return cls(traits={"playfulness": 0.95, "energy": 0.85, "curiosity": 0.8,
                           "sociability": 0.7, "patience": 0.35},
                   drives={"play": {"value": 0.55, "growth_rate": 0.04},
                           "social": {"value": 0.3, "growth_rate": 0.015}})

    @classmethod
    def social(cls) -> "Personality":
        return cls(traits={"sociability": 0.95, "playfulness": 0.6,
                           "expressiveness": 0.85, "patience": 0.65},
                   drives={"social": {"value": 0.6, "growth_rate": 0.03}})

    @classmethod
    def curious(cls) -> "Personality":
        return cls(traits={"curiosity": 0.95, "playfulness": 0.7, "bravery": 0.65},
                   drives={"curiosity": {"value": 0.6, "growth_rate": 0.035}})

    @classmethod
    def shy(cls) -> "Personality":
        return cls(traits={"sociability": 0.25, "bravery": 0.2, "curiosity": 0.45,
                           "patience": 0.7},
                   emotions={"anxious": 0.35, "calm": 0.5},
                   baselines={"anxious": 0.2})

    @classmethod
    def calm(cls) -> "Personality":
        return cls(traits={"playfulness": 0.35, "energy": 0.35, "patience": 0.9,
                           "bravery": 0.5},
                   emotions={"calm": 0.85, "anxious": 0.0},
                   baselines={"calm": 0.75})

    @classmethod
    def brave(cls) -> "Personality":
        return cls(traits={"bravery": 0.95, "curiosity": 0.75, "energy": 0.7},
                   drives={"safety": {"value": 0.05, "growth_rate": 0.01}})

    @classmethod
    def tired(cls) -> "Personality":
        return cls(traits={"energy": 0.2, "patience": 0.4, "playfulness": 0.25},
                   drives={"rest": {"value": 0.75, "growth_rate": 0.02}},
                   emotions={"tired": 0.75, "calm": 0.65},
                   baselines={"tired": 0.45})

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _activity_name(activity: Any) -> str:
        for attribute in ("id", "activity_id", "name"):
            value = getattr(activity, attribute, None)
            if value:
                return str(value)
        return str(activity)

    @staticmethod
    def _default_event_responses() -> Dict[str, Dict[str, Dict[str, float]]]:
        return {
            "RobotPickedUp": {
                "emotions": {"anxious": 0.30, "excited": 0.10},
                "drives": {"safety": 0.20},
            },
            "RobotFalling": {
                "emotions": {"anxious": 0.55, "frustrated": 0.15},
                "drives": {"safety": 0.45},
            },
            "CliffDetected": {
                "emotions": {"anxious": 0.45, "frustrated": 0.10},
                "drives": {"safety": 0.50},
            },
            "PlacedOnCharger": {
                "emotions": {"calm": 0.25, "tired": 0.10, "anxious": -0.15},
                "drives": {"rest": -0.20, "safety": -0.15},
            },
            "FaceInitialDetection": {
                "emotions": {"social": 0.25, "happy": 0.15, "curious": 0.10},
                "drives": {"social": -0.15},
            },
            "FaceDetected": {
                "emotions": {"social": 0.15, "happy": 0.10},
                "drives": {"social": -0.08},
            },
            "SmileDetected": {
                "emotions": {"happy": 0.25, "social": 0.15},
                "drives": {"social": -0.10},
            },
            "FrownDetected": {
                "emotions": {"frustrated": 0.10, "anxious": 0.10},
                "drives": {"social": 0.05},
            },
            "PetInitialDetection": {
                "emotions": {"excited": 0.25, "curious": 0.20},
                "drives": {"curiosity": -0.15},
            },
            "CubeInitialDetection": {
                "emotions": {"curious": 0.20, "excited": 0.15},
                "drives": {"play": -0.10, "curiosity": -0.10},
            },
            "CubeTapped": {
                "emotions": {"excited": 0.30, "happy": 0.15},
                "drives": {"play": -0.20},
            },
            "UnexpectedMovement": {
                "emotions": {"anxious": 0.20, "curious": 0.10},
                "drives": {"safety": 0.15},
            },
            "BehaviorDone": {
                "emotions": {"happy": 0.05},
            },
        }


def scalescore(condition: bool, amount: float) -> float:
    """ Tiny helper used by :meth:`Personality.score_activity`. """
    return float(amount) if condition else 0.0
