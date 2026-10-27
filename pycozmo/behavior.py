"""

Behavior representation and reading.

"""

import os
import random
import time
from typing import Dict, Optional, Any

from . import event
from . import client
from . import logger
from .json_loader import get_json_files, load_json_file


__all__ = [
    "ReactionTrigger",
    "Behavior",
    "BehaviorPlayAnim",
    "BehaviorPlayArbitraryAnim",
    "BehaviorReactToCliff",
    "BehaviorDriveOffCharger",

    "load_behaviors",
    "load_reaction_trigger_behavior_map",
]


def _normalize_triggers(value: Any) -> list:
    """ Accept a single trigger, a list, or a comma-separated string. """
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, dict):
        return [value] if value else []
    return [item for item in value if item]


class ReactionTrigger:
    """ Reaction trigger representation class. """
    __slots__ = [
        "name",
        "behavior_id",
        "should_resume_last",
    ]

    def __init__(self, name: str, behavior_id: str, should_resume_last: Optional[bool] = False):
        self.name = str(name)
        self.behavior_id = str(behavior_id)
        self.should_resume_last = bool(should_resume_last)

    @classmethod
    def from_json(cls, data: Dict):
        return cls(name=data['reactionTrigger'],
                   behavior_id=data['behaviorID'],
                   should_resume_last=data.get('genericStrategyParams', {}).get('shouldResumeLast'))


class Behavior(event.Dispatcher):
    """ Behavior representation class. """

    def __init__(self, cli: client.Client, conf: Any) -> None:
        super().__init__()
        self.cli = cli
        self.conf = conf

    def get_id(self) -> str:
        return self.conf["behaviorID"]

    def activate(self) -> None:
        logger.warning("Behavior '{}' not implemented.".format(self.get_id()))
        self.cli.conn.post_event(event.EvtBehaviorDone, self.cli)

    def deactivate(self) -> None:
        pass


class BehaviorPlayAnim(Behavior):
    """ Play a sequence of animation triggers.

    ``animTriggers`` in the behavior configuration may be a single trigger
    name or a list.  Each trigger is played when the previous animation
    completes; after the last one :class:`~pycozmo.event.EvtBehaviorDone` is
    posted.
    """

    def __init__(self, cli: client.Client, conf: Any):
        super().__init__(cli, conf)
        self.anim_triggers = _normalize_triggers(conf.get("animTriggers", []))
        self.current_trigger = 0
        self._active = False
        self._done = False
        self.add_handler(event.EvtAnimationCompleted, self._on_animation_completed)

    def activate(self) -> None:
        self.current_trigger = 0
        self._active = True
        self._done = False
        if self.anim_triggers:
            self._play_next()
        else:
            self._finish()

    def _play_next(self) -> None:
        if not self._active or self._done:
            return
        if self.current_trigger >= len(self.anim_triggers):
            self._finish()
            return
        anim_trigger = self.anim_triggers[self.current_trigger]
        self.current_trigger += 1
        self.cli.play_anim_group(anim_trigger)

    def _on_animation_completed(self, cli) -> None:
        del cli
        if self._active and not self._done:
            self._play_next()

    def _finish(self) -> None:
        if self._done:
            return
        self._done = True
        self._active = False
        self.cli.conn.post_event(event.EvtBehaviorDone, self.cli)

    def deactivate(self) -> None:
        self._active = False
        self.cli.cancel_anim()


class BehaviorPlayArbitraryAnim(BehaviorPlayAnim):
    """ Play a random selection from the configured animation triggers. """

    def __init__(self, cli: client.Client, conf: Any, rng: Optional[random.Random] = None):
        super().__init__(cli, conf)
        self.rng = rng or random.Random()
        self._all_anim_triggers = list(self.anim_triggers)

    def activate(self) -> None:
        triggers = list(self._all_anim_triggers)
        if triggers:
            self.rng.shuffle(triggers)
            count = self.conf.get("numAnimations", self.conf.get("num_animations"))
            try:
                count = int(count)
            except (TypeError, ValueError):
                count = self.rng.randint(1, len(triggers))
            count = max(1, min(len(triggers), count))
            self.anim_triggers = triggers[:count]
        super().activate()


class BehaviorReactToCliff(BehaviorPlayAnim):
    """ ReactToCliff behavior - stop and play the reaction animation. """

    def activate(self) -> None:
        if hasattr(self.cli, "stop_all_motors"):
            self.cli.stop_all_motors()
        self.anim_triggers = ["ReactToCliff"]
        super().activate()


class BehaviorDriveOffCharger(Behavior):
    """ Drive backwards off the charging contacts.

    The optional ``extraDistanceToDrive_mm`` configuration value is travelled
    at ``driveSpeed_mmps`` (default ``50`` mm/s) after the contacts are
    cleared.  Motion is time based because the robot does not report charger
    contact state separately from the charging status.
    """

    def __init__(self, cli: client.Client, conf: Any) -> None:
        super().__init__(cli, conf)
        self._active = False
        self._cancelled = False

    def activate(self) -> None:
        self._active = True
        self._cancelled = False
        speed = abs(float(self.conf.get("driveSpeed_mmps", 50.0)))
        extra_distance = max(0.0, float(self.conf.get("extraDistanceToDrive_mm", 0.0)))
        contact_time = max(0.0, float(self.conf.get("timeToDriveOffContacts_sec", 1.5)))

        wake_trigger = self.conf.get("wakeUpAnimTrigger")
        if wake_trigger and self.conf.get("playWakeUpAnim", False):
            try:
                self.cli.play_anim_group(wake_trigger)
            except Exception as exc:  # pragma: no cover - hardware/asset dependent
                logger.warning("Failed to play wake-up animation '{}': {}".format(wake_trigger, exc))

        try:
            if hasattr(self.cli, "drive_off_charger_contacts"):
                self.cli.drive_off_charger_contacts(speed=-speed, duration=contact_time)
            elif hasattr(self.cli, "drive_straight"):
                self.cli.drive_straight(-speed, duration=contact_time)

            if self._active and not self._cancelled and extra_distance > 0:
                duration = extra_distance / max(speed, 1.0)
                if hasattr(self.cli, "drive_straight"):
                    self.cli.drive_straight(-speed, duration=duration)
                elif hasattr(self.cli, "drive_wheels"):
                    self.cli.drive_wheels(-speed, -speed, duration=duration)
        except Exception as exc:
            logger.error("DriveOffCharger failed: {}".format(exc))
        finally:
            self._active = False
            if not self._cancelled:
                self.cli.conn.post_event(event.EvtBehaviorDone, self.cli)

    def deactivate(self) -> None:
        self._cancelled = True
        self._active = False
        if hasattr(self.cli, "stop_all_motors"):
            self.cli.stop_all_motors()


#: Mapping used by :func:`get_behavior_class_from_dict`.  Applications may add
#: their own behavior classes to this dictionary.
BEHAVIOR_CLASSES = {
    "PlayAnim": BehaviorPlayAnim,
    "PlayArbitraryAnim": BehaviorPlayArbitraryAnim,
    "ReactToCliff": BehaviorReactToCliff,
    "DriveOffCharger": BehaviorDriveOffCharger,
}


def get_behavior_class_from_dict(data):
    """ Choose a behavior class, based on the behaviorClass JSON attribute. """
    class_name = data.get("behaviorClass")
    if class_name in BEHAVIOR_CLASSES:
        return BEHAVIOR_CLASSES[class_name]
    # Some resource files use a fully qualified name or a different case.
    normalized = str(class_name or "").split(".")[-1].lower()
    for key, value in BEHAVIOR_CLASSES.items():
        if key.lower() == normalized:
            return value
    return Behavior


def load_behaviors(resource_dir: str, cli: client.Client) -> Dict[str, Behavior]:

    start_time = time.perf_counter()

    behavior_files = get_json_files(
        resource_dir, [os.path.join('cozmo_resources', 'config', 'engine', 'behaviorSystem', 'behaviors')])
    behaviors = {}
    for filename in behavior_files:
        data = load_json_file(filename)
        cls = get_behavior_class_from_dict(data)
        behaviors[data['behaviorID']] = cls(cli, data)

    logger.debug("Loaded {} behaviors in {:.02f} s.".format(len(behaviors), time.perf_counter() - start_time))

    return behaviors


def load_reaction_trigger_behavior_map(resource_dir: str) -> Dict[str, ReactionTrigger]:

    start_time = time.perf_counter()

    reaction_trigger_behavior_map = {}
    filename = os.path.join(resource_dir, 'cozmo_resources', 'config',
                            'engine', 'behaviorSystem', 'reactionTrigger_behavior_map.json')

    json_data = load_json_file(filename)
    for trigger in json_data['reactionTriggerBehaviorMap']:
        reaction_trigger_behavior_map[trigger['reactionTrigger']] = ReactionTrigger.from_json(trigger)

    logger.debug("Loaded {} entry reaction trigger behavior map in {:.02f} s.".format(
        len(reaction_trigger_behavior_map), time.perf_counter() - start_time))

    return reaction_trigger_behavior_map
