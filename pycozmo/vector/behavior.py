"""

Animation-driven behaviors for Vector clips.

A :class:`VectorBehavior` wraps a :class:`~pycozmo.vector.VectorClip` and
plays it through a Cozmo-like client or user-supplied handlers.  Unlike a
plain animation player it tracks completion, supports looping and cooldowns,
and can be selected by a small priority-based :class:`VectorBehaviorSystem`.

The module has no Vector SDK dependency, so the same clips can drive a Cozmo
robot (using the methods supported by :class:`VectorAnimator`) or be used for
offline playback and tests.

"""

from __future__ import annotations

import random
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Union

from .animation import (VectorAnimations, VectorAnimator, VectorClip,
                        load_vector_animations)


__all__ = [
    "VectorBehaviorError",
    "VectorBehavior",
    "VectorBehaviorSystem",
]


class VectorBehaviorError(ValueError):
    """ Raised for invalid Vector behavior configuration. """


def _clip_name(value: Any) -> str:
    return str(value or "").strip()


class VectorBehavior:
    """ One animation-driven behavior.

    Args:
        name: Behavior name.
        clip: Animation clip or a clip name resolved by the caller.
        client: Optional client with Cozmo/Vector-like playback methods.
        handlers: Optional :class:`VectorAnimator` handler mapping.
        speed: Playback speed multiplier.
        priority: Larger values are preferred by :class:`VectorBehaviorSystem`.
        cooldown: Seconds before the behavior may run again.
        loop: Replay the clip until :meth:`stop` is called.
        on_done: Optional callback invoked with this behavior after a run.
    """

    def __init__(self,
                 name: str,
                 clip: VectorClip,
                 client: Any = None,
                 handlers: Optional[Dict[str, Callable[[Any], Any]]] = None,
                 speed: float = 1.0,
                 priority: float = 0.0,
                 cooldown: float = 0.0,
                 loop: bool = False,
                 on_done: Optional[Callable[["VectorBehavior"], Any]] = None,
                 rng: Optional[random.Random] = None) -> None:
        if not isinstance(clip, VectorClip):
            raise VectorBehaviorError("clip must be a VectorClip instance.")
        self.name = str(name or clip.name or "behavior")
        self.clip = clip
        self.client = client
        self.handlers = dict(handlers or {})
        self.speed = float(speed)
        if self.speed <= 0:
            raise VectorBehaviorError("Playback speed must be positive.")
        self.priority = float(priority)
        self.cooldown = max(0.0, float(cooldown))
        self.loop = bool(loop)
        self.on_done = on_done
        self.rng = rng or random.Random()
        self.runs = 0
        self.last_started: Optional[float] = None
        self.last_finished: Optional[float] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()

    @property
    def running(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    def available(self, now: Optional[float] = None) -> bool:
        """ Return whether the behavior may start now. """
        if self.running:
            return False
        if self.last_finished is None or self.cooldown <= 0:
            return True
        now = time.monotonic() if now is None else float(now)
        return now - self.last_finished >= self.cooldown

    def start(self, wait: bool = False) -> threading.Thread:
        """ Start playback in a background thread.

        Returns the worker thread.  Use :meth:`play` for synchronous playback.
        """
        with self._lock:
            if self.running:
                raise VectorBehaviorError(
                    "Behavior {!r} is already running.".format(self.name))
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._run, daemon=True, name="VectorBehavior-{}".format(self.name))
            thread = self._thread
            thread.start()
        if wait:
            thread.join()
        return thread

    def play(self) -> None:
        """ Play the clip synchronously in the current thread. """
        if self.running:
            raise VectorBehaviorError(
                "Behavior {!r} is already running.".format(self.name))
        self._run()

    def _run(self) -> None:
        self.last_started = time.monotonic()
        try:
            while not self._stop.is_set():
                animator = VectorAnimator(self.clip, rng=self.rng)
                animator.play(client=self.client, handlers=self.handlers,
                              speed=self.speed, stop_event=self._stop)
                if not self.loop or self._stop.is_set():
                    break
        finally:
            self.last_finished = time.monotonic()
            self.runs += 1
            callback = self.on_done
            if callback is not None and not self._stop.is_set():
                try:
                    callback(self)
                except Exception:
                    # A user callback must not hide playback completion.
                    pass

    def stop(self, timeout: Optional[float] = 5.0) -> None:
        """ Request stop and optionally wait for the worker thread. """
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive() and timeout is not None:
            thread.join(timeout)

    def wait(self, timeout: Optional[float] = None) -> bool:
        """ Wait for the current run to finish.  Returns ``True`` if it did. """
        thread = self._thread
        if thread is None:
            return True
        thread.join(timeout)
        return not thread.is_alive()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "clip": self.clip.name,
            "priority": self.priority,
            "cooldown": self.cooldown,
            "loop": self.loop,
            "speed": self.speed,
        }

    def __repr__(self) -> str:
        return "VectorBehavior(name={!r}, clip={!r}, priority={})".format(
            self.name, self.clip.name, self.priority)


class VectorBehaviorSystem:
    """ Select and play a set of :class:`VectorBehavior` objects.

    Behaviors are chosen by descending priority, then by insertion order.
    Behaviors that are running or cooling down are skipped.
    """

    def __init__(self, animations: Optional[VectorAnimations] = None,
                 rng: Optional[random.Random] = None) -> None:
        self.animations = animations or VectorAnimations()
        self.rng = rng or random.Random()
        self.behaviors: Dict[str, VectorBehavior] = {}

    def add(self, behavior: VectorBehavior, replace: bool = True) -> VectorBehavior:
        if not isinstance(behavior, VectorBehavior):
            raise VectorBehaviorError("Expected a VectorBehavior instance.")
        if behavior.name in self.behaviors and not replace:
            raise VectorBehaviorError(
                "Behavior {!r} already exists.".format(behavior.name))
        self.behaviors[behavior.name] = behavior
        return behavior

    def add_clip(self, name: str, clip: Union[str, VectorClip],
                 **kwargs: Any) -> VectorBehavior:
        if isinstance(clip, str):
            resolved = self.animations.get(clip)
            if resolved is None:
                raise VectorBehaviorError(
                    "Animation clip {!r} was not found.".format(clip))
            clip = resolved
        return self.add(VectorBehavior(name, clip, rng=self.rng, **kwargs))

    def from_dict(self, data: Dict[str, Any]) -> VectorBehavior:
        """ Add a behavior from a dictionary.

        Accepted keys are ``name``/``behaviorID``, ``clip``/``clipName``,
        ``priority``, ``cooldown``, ``loop``, ``speed`` and the runtime
        options passed to :class:`VectorBehavior`.
        """
        if not isinstance(data, dict):
            raise VectorBehaviorError("Behavior configuration must be a dictionary.")
        name = _clip_name(data.get("name", data.get("behaviorID", "")))
        clip_name = _clip_name(data.get("clip", data.get("clipName", "")))
        if not name:
            name = clip_name
        if not clip_name:
            raise VectorBehaviorError("Behavior configuration needs a clip name.")
        return self.add_clip(
            name, clip_name,
            priority=data.get("priority", 0.0),
            cooldown=data.get("cooldown", 0.0),
            loop=data.get("loop", False),
            speed=data.get("speed", 1.0),
        )

    def get(self, name: str) -> Optional[VectorBehavior]:
        return self.behaviors.get(name)

    def remove(self, name: str) -> bool:
        behavior = self.behaviors.pop(name, None)
        if behavior is not None:
            behavior.stop()
            return True
        return False

    def available(self, now: Optional[float] = None) -> List[VectorBehavior]:
        behaviors = [behavior for behavior in self.behaviors.values()
                     if behavior.available(now)]
        return sorted(behaviors, key=lambda behavior: behavior.priority,
                      reverse=True)

    def choose(self, now: Optional[float] = None) -> Optional[VectorBehavior]:
        available = self.available(now)
        return available[0] if available else None

    def play(self, name: Optional[str] = None, client: Any = None,
             handlers: Optional[Dict[str, Callable[[Any], Any]]] = None,
             now: Optional[float] = None, wait: bool = True) -> Optional[VectorBehavior]:
        """ Choose and start a behavior, returning it when one is available. """
        behavior = self.get(name) if name is not None else self.choose(now)
        if behavior is None or not behavior.available(now):
            return None
        if client is not None:
            behavior.client = client
        if handlers is not None:
            behavior.handlers.update(handlers)
        behavior.start(wait=wait)
        return behavior

    def stop(self, timeout: Optional[float] = 5.0) -> None:
        for behavior in list(self.behaviors.values()):
            behavior.stop(timeout)

    def __contains__(self, name: object) -> bool:
        return name in self.behaviors

    def __len__(self) -> int:
        return len(self.behaviors)

    def __iter__(self):
        return iter(self.behaviors.values())


def load_vector_behaviors(source: Any,
                          animations: Optional[VectorAnimations] = None,
                          rng: Optional[random.Random] = None) -> VectorBehaviorSystem:
    """ Load behaviors from a JSON list or an animation file.

    ``source`` may be a :class:`VectorAnimations` instance, a path to a
    ``.json`` behavior list, or a path/bytes animation file.  When no explicit
    behavior list is supplied, every clip becomes a behavior with the clip
    name as its behavior name.
    """
    if isinstance(source, VectorBehaviorSystem):
        return source
    if isinstance(source, VectorAnimations):
        system = VectorBehaviorSystem(source, rng=rng)
        for clip in source:
            system.add_clip(clip.name, clip)
        return system
    if isinstance(source, (str, bytes, bytearray, memoryview)) and \
            isinstance(source, str) and source.lower().endswith(".json"):
        import json
        with open(source, "r", encoding="utf-8") as stream:
            data = json.load(stream)
        if isinstance(data, dict):
            data = data.get("behaviors", [])
        system = VectorBehaviorSystem(animations or VectorAnimations(), rng=rng)
        for item in data:
            system.from_dict(item)
        return system
    animations = animations or load_vector_animations(source)
    return load_vector_behaviors(animations, rng=rng)


__all__ += ["load_vector_behaviors"]
