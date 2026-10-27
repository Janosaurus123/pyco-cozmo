"""

Activity representation and reading.

"""

import os
import random
import time
from typing import Any, Dict, List, Optional

from . import logger
from .emotions import DecayGraph, Node
from .json_loader import get_json_files, load_json_file


__all__ = [
    "BehaviorChooser",
    "Objective",
    "Activity",
    "BehaviorsActivity",
    "VoiceCommandActivity",
    "FeedingActivity",
    "FreeplayActivity",
    "SparkedActivity",
    "PyramidActivity",
    "SocializeActivity",
    "NeedsActivity",
    "from_dict",

    "load_activities",
]


class BehaviorChooser:
    """ Choose behavior IDs according to a behavior-system strategy.

    Supported strategies:

    ``Selection``
        Use the configured order as-is.
    ``StrictPriority``
        Try behaviors in the configured order, one per iteration.
    ``Scoring``
        Weighted random order without replacement.  Scores are reduced by a
        per-behavior repetition penalty.
    """

    __slots__ = [
        "choice_type",
        "behaviors",
        "iteration",
        "rng",
        "__dict__",
    ]

    def __init__(self,
                 choice_type: str,
                 behaviors: List,
                 rng: Optional[random.Random] = None) -> None:

        self.choice_type = str(choice_type)
        self.behaviors = list(behaviors or [])
        self.rng = rng or random.Random()
        self.iteration = 0

        if self.choice_type == 'Scoring':
            self.init_scores()
            self.init_repetition_penalty()

    @classmethod
    def from_json(cls, data: Dict):
        return cls(
            choice_type=data.get('type', 'Selection'),
            behaviors=data.get('behaviors', [])
        )

    def reset(self):
        self.iteration = 0
        if self.choice_type == 'Scoring':
            self.init_scores()

    def init_scores(self) -> None:
        self.behavior_scores = []
        self.behavior_names = []
        self.behavior_repetitions = []
        for behavior in self.behaviors:
            scoring = behavior.get('scoring', {}) if isinstance(behavior, dict) else {}
            self.behavior_scores.append(float(scoring.get('flatScore', 1.0)))
            name = behavior.get('behaviorID', behavior) if isinstance(behavior, dict) else behavior
            self.behavior_names.append(str(name))
            self.behavior_repetitions.append(0)
        self.total_score = sum(self.behavior_scores)

    def init_repetition_penalty(self) -> None:
        self.repetition_penaltys = []
        for behavior in self.behaviors:
            scoring = behavior.get('scoring', {}) if isinstance(behavior, dict) else {}
            penalty = scoring.get('repetitionPenalty')
            if penalty and 'nodes' in penalty:
                self.repetition_penaltys.append(
                    DecayGraph([Node(x=n['x'], y=n['y']) for n in penalty['nodes']]))
            else:
                self.repetition_penaltys.append(DecayGraph([Node(x=1, y=0)]))

    def _index(self, ref: Any) -> int:
        if isinstance(ref, int):
            return ref
        name = ref
        if isinstance(ref, dict):
            name = ref.get('behaviorID', ref)
        name = str(name)
        try:
            return self.behavior_names.index(name)
        except ValueError:
            return -1

    def apply_repetition_penalty(self, ref) -> None:
        if self.choice_type != 'Scoring':
            return
        idx = self._index(ref)
        if idx < 0:
            return
        self.behavior_repetitions[idx] += 1
        self.behavior_scores[idx] -= self.repetition_penaltys[idx].get_increment(self.behavior_repetitions[idx])
        self.behavior_scores[idx] = max(0, self.behavior_scores[idx])
        self.total_score = sum(self.behavior_scores)

    def _names(self) -> List[str]:
        return [str(behavior.get('behaviorID', behavior)
                    if isinstance(behavior, dict) else behavior)
                for behavior in self.behaviors]

    def get_sorted_choices(self) -> Optional[List[str]]:
        """ Return available behavior IDs in the order they should be tried. """
        names = self._names()
        if self.choice_type == 'Selection':
            return names
        if self.choice_type == 'StrictPriority':
            if self.iteration < len(names):
                return names[self.iteration:]
            return None
        if self.choice_type == 'Scoring':
            if not names:
                return None
            remaining = list(names)
            scores = dict(zip(self.behavior_names, self.behavior_scores))
            ordered: List[str] = []
            while remaining:
                weights = [max(1e-9, float(scores.get(name, 1.0))) for name in remaining]
                total = sum(weights)
                pick = self.rng.random() * total
                cumulative = 0.0
                index = len(remaining) - 1
                for i, weight in enumerate(weights):
                    cumulative += weight
                    if pick <= cumulative:
                        index = i
                        break
                ordered.append(remaining.pop(index))
            return ordered
        raise ValueError('Unknown choice type: {}'.format(self.choice_type))

    def choose(self, available: Optional[Any] = None) -> Optional[str]:
        """ Return the first available behavior, notifying the chooser.

        ``available`` may be a callable, a collection of behavior IDs, or
        ``None`` to accept every configured behavior.
        """
        choices = self.get_sorted_choices()
        if not choices:
            return None
        for behavior_id in choices:
            if self.is_available(behavior_id, available):
                self.iteration += 1
                self.apply_repetition_penalty(behavior_id)
                return behavior_id
        return None

    @staticmethod
    def is_available(behavior_id: str, available: Optional[Any]) -> bool:
        if available is None:
            return True
        if callable(available):
            try:
                return bool(available(behavior_id))
            except TypeError:
                return bool(available())
        try:
            return behavior_id in available
        except TypeError:
            return True


class Objective:
    __slots__ = [
        "objective",
        "behavior_id",
        "ignore_if_locked",
        "probability_to_require_objective",
        "random_completions_needed_min",
        "random_completions_needed_max",
    ]

    def __init__(self,
                 objective: str,
                 behavior_id: str,
                 ignore_if_locked: str,
                 probability_to_require_objective: float,
                 random_completions_needed_min: Optional[int] = 0,
                 random_completions_needed_max: Optional[int] = 0) -> None:
        self.objective = str(objective)
        self.behavior_id = str(behavior_id)
        self.ignore_if_locked = str(ignore_if_locked)
        self.probability_to_require_objective = float(probability_to_require_objective)
        self.random_completions_needed_min = \
            int(random_completions_needed_min) if random_completions_needed_min is not None else 0
        self.random_completions_needed_max = \
            int(random_completions_needed_max) if random_completions_needed_max is not None else 0

    @classmethod
    def from_json(cls, data: Dict):
        return cls(objective=data['objective'],
                   behavior_id=data['behaviorID'],
                   ignore_if_locked=data['ignoreIfLocked'],
                   probability_to_require_objective=data['probabilityToRequireObjective'],
                   random_completions_needed_min=data.get('randomCompletionsNeededMin', 0),
                   random_completions_needed_max=data.get('randomCompletionsNeededMax', 0))


class Activity:
    """ Activity representation class. """

    __slots__ = [
        "id",
        "type",
        "strategy",
    ]

    def __init__(self,
                 activity_id: str,
                 activity_type: str,
                 strategy: str) -> None:

        self.id = str(activity_id)
        self.type = str(activity_type)
        self.strategy = str(strategy)

    def available(self, context: Optional[Dict[str, Any]] = None) -> bool:
        """ Return whether this activity can run in ``context``. """
        return True

    def choose(self, context: Optional[Dict[str, Any]] = None,
               rng: Optional[random.Random] = None) -> Optional[str]:
        """ Return the next behavior or sub-activity, if this activity has one.

        The base activity has no behavior chooser and returns ``None``.
        """
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "activityID": self.id,
            "activityType": self.type,
            "activityStrategy": {"type": self.strategy},
        }


class BehaviorsActivity(Activity):
    __slots__ = [
        "behavior_chooser"
    ]

    def __init__(self,
                 behavior_chooser: BehaviorChooser,
                 *args, **kwargs):
        self.behavior_chooser = behavior_chooser
        super().__init__(*args, **kwargs)

    @classmethod
    def from_json(cls, data: Dict):
        return cls(
            BehaviorChooser.from_json(data['behaviorChooser'])
            if 'behaviorChooser' in data else None,
            activity_id=data['activityID'],
            activity_type=data['activityType'],
            strategy=data['activityStrategy']['type'])

    def choose(self, context: Optional[Dict[str, Any]] = None,
               rng: Optional[random.Random] = None) -> Optional[str]:
        if self.behavior_chooser is None:
            return None
        available = context.get("available_behaviors") if context else None
        return self.behavior_chooser.choose(available)

    def choose_behaviors(self, context: Optional[Dict[str, Any]] = None) -> List[str]:
        if self.behavior_chooser is None:
            return []
        return self.behavior_chooser.get_sorted_choices() or []

    def available(self, context: Optional[Dict[str, Any]] = None) -> bool:
        return self.behavior_chooser is not None and bool(self.behavior_chooser.behaviors)


class VoiceCommandActivity(Activity):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @classmethod
    def from_json(cls, data: Dict):
        return cls(
            activity_id=data['activityID'],
            activity_type=data['activityType'],
            strategy=data['activityStrategy']['type'])


class FeedingActivity(Activity):
    __slots__ = [
        "universal_chooser",
    ]

    def __init__(self,
                 universal_chooser: List[str],
                 *args, **kwargs) -> None:
        self.universal_chooser = universal_chooser
        super().__init__(*args, **kwargs)

    @classmethod
    def from_json(cls, data: Dict):
        return cls(
            universal_chooser=data['universalChooser']['behaviors']
            if 'universalChooser' in data else None,
            activity_id=data['activityID'],
            activity_type=data['activityType'],
            strategy=data['activityStrategy']['type'])

    def choose(self, context: Optional[Dict[str, Any]] = None,
               rng: Optional[random.Random] = None) -> Optional[str]:
        choices = list(self.universal_chooser or ())
        available = context.get("available_behaviors") if context else None
        if available is not None and not callable(available):
            choices = [item for item in choices if item in available]
        if not choices:
            return None
        return (rng or random).choice(choices)

    def available(self, context: Optional[Dict[str, Any]] = None) -> bool:
        return bool(self.universal_chooser)


class FreeplayActivity(Activity):
    __slots__ = [
        "cube_only_activity",
        "face_only_activity",
        "face_and_cube_activity",
        "no_face_no_cube_activity",
        "sub_activities",
    ]

    def __init__(self,
                 cube_only_activity: str,
                 face_only_activity: str,
                 face_and_cube_activity: str,
                 no_face_no_cube_activity: str,
                 sub_activities: List,
                 *args, **kwargs) -> None:
        self.cube_only_activity = str(cube_only_activity)
        self.face_only_activity = str(face_only_activity)
        self.face_and_cube_activity = str(face_and_cube_activity)
        self.no_face_no_cube_activity = str(no_face_no_cube_activity)
        self.sub_activities = sub_activities

        super().__init__(*args, **kwargs)

    @classmethod
    def from_json(cls, data: Dict):
        return cls(
            cube_only_activity=data['desiredActivityNames']['cubeOnlyActivityName'],
            face_only_activity=data['desiredActivityNames']['faceOnlyActivityName'],
            face_and_cube_activity=data['desiredActivityNames']['faceAndCubeActivityName'],
            no_face_no_cube_activity=data['desiredActivityNames']['noFaceNoCubeActivityName'],
            sub_activities=data.get('subActivities'),
            activity_id=data['activityID'],
            activity_type=data['activityType'],
            strategy=data['activityStrategy']['type'])

    def choose_sub_activity(self, context: Optional[Dict[str, Any]] = None) -> str:
        """ Pick the sub-activity that matches the current face/cube situation. """
        context = context or {}
        face = bool(context.get("face_visible", context.get("faces", False)))
        cube = bool(context.get("cube_visible", context.get("cubes", False)))
        if face and cube:
            return self.face_and_cube_activity
        if face:
            return self.face_only_activity
        if cube:
            return self.cube_only_activity
        return self.no_face_no_cube_activity

    def choose(self, context: Optional[Dict[str, Any]] = None,
               rng: Optional[random.Random] = None) -> Optional[str]:
        return self.choose_sub_activity(context)

    def available(self, context: Optional[Dict[str, Any]] = None) -> bool:
        return True


class SparkedActivity(Activity):
    __slots__ = [
        "require_spark",
        "min_time",
        "max_time",
        "reps",
        "behavior_objective",
        "soft_spark_trigger",
        "behavior_chooser",
        "sub_activity_delegate",
        "spark_success_trigger",
        "spark_fail_trigger",
        "drive_start_trigger",
        "drive_loop_trigger",
        "drive_stop_trigger",
    ]

    def __init__(self,
                 require_spark: str,
                 min_time_secs: float,
                 max_time_secs: float,
                 reps: int,
                 behavior_objective: str,
                 soft_spark_trigger: str,
                 behavior_chooser: Optional[BehaviorChooser] = None,
                 sub_activity_delegate: Optional[Activity] = None,
                 spark_success_trigger: Optional[str] = None,
                 spark_fail_trigger: Optional[str] = None,
                 drive_start_trigger: Optional[str] = None,
                 drive_loop_trigger: Optional[str] = None,
                 drive_stop_trigger: Optional[str] = None,
                 *args, **kwargs) -> None:
        self.require_spark = str(require_spark)
        self.min_time = float(min_time_secs)
        self.max_time = float(max_time_secs)
        self.reps = int(reps)
        self.behavior_objective = str(behavior_objective)
        self.soft_spark_trigger = str(soft_spark_trigger)

        self.behavior_chooser = behavior_chooser
        self.sub_activity_delegate = sub_activity_delegate
        self.spark_success_trigger = spark_success_trigger
        self.spark_fail_trigger = spark_fail_trigger

        self.drive_start_trigger = drive_start_trigger
        self.drive_loop_trigger = drive_loop_trigger
        self.drive_stop_trigger = drive_stop_trigger

        super().__init__(*args, **kwargs)

    @classmethod
    def from_json(cls, data: Dict):
        if 'subActivityDelegate' in data:
            sub_act_delegate = from_dict(data['subActivityDelegate'])
        else:
            sub_act_delegate = None
        return cls(
            require_spark=data['requireSpark'],
            min_time_secs=data['minTimeSecs'],
            max_time_secs=data['maxTimeSecs'],
            reps=data['numberOfRepetitions'],
            behavior_objective=data['behaviorObjective'],
            soft_spark_trigger=data['softSparkTrigger'],
            behavior_chooser=BehaviorChooser.from_json(data['behaviorChooser'])
            if 'behaviorChooser' in data else None,
            sub_activity_delegate=sub_act_delegate,
            spark_success_trigger=data.get('sparksSuccessTrigger'),
            spark_fail_trigger=data.get('sparksFailTrigger'),
            drive_start_trigger=data.get('driveStartAnimTrigger'),
            drive_loop_trigger=data.get('driveLoopAnimTrigger'),
            drive_stop_trigger=data.get('driveStopAnimTrigger'),
            activity_id=data['activityID'],
            activity_type=data['activityType'],
            strategy=data['activityStrategy']['type'])

    def available(self, context: Optional[Dict[str, Any]] = None) -> bool:
        required = self.require_spark.lower()
        if required in ("", "none", "false", "no", "nospark"):
            return True
        if required in ("true", "yes", "spark", "required", "requirespark"):
            return bool((context or {}).get("spark", True))
        return True

    def choose(self, context: Optional[Dict[str, Any]] = None,
               rng: Optional[random.Random] = None) -> Optional[str]:
        if not self.available(context):
            return None
        if self.sub_activity_delegate is not None and (context or {}).get("use_sub_activity"):
            return self.sub_activity_delegate.choose(context, rng)
        if self.behavior_chooser is not None:
            available = (context or {}).get("available_behaviors")
            return self.behavior_chooser.choose(available)
        return None


class PyramidActivity(Activity):
    __slots__ = [
        "setup_chooser",
        "build_chooser",
        "interlude_chooser",
        "needs_action_id",
    ]

    def __init__(self,
                 setup_chooser: BehaviorChooser,
                 build_chooser: BehaviorChooser,
                 interlude_chooser: Optional[BehaviorChooser] = None,
                 needs_action_id: Optional[str] = None,
                 *args, **kwargs):
        self.setup_chooser = setup_chooser
        self.build_chooser = build_chooser
        self.interlude_chooser = interlude_chooser
        self.needs_action_id = needs_action_id

        super().__init__(*args, **kwargs)

    @classmethod
    def from_json(cls, data: Dict):
        return cls(
            setup_chooser=BehaviorChooser.from_json(data['setupChooser']),
            build_chooser=BehaviorChooser.from_json(data['buildChooser']),
            interlude_chooser=BehaviorChooser.from_json(data['interludeBehaviorChooser'])
            if 'interludeBehaviorChooser' in data else None,
            needs_action_id=data.get('needsActionID'),
            activity_id=data['activityID'],
            activity_type=data['activityType'],
            strategy=data['activityStrategy']['type'])

    def choose(self, context: Optional[Dict[str, Any]] = None,
               rng: Optional[random.Random] = None) -> Optional[str]:
        context = context or {}
        if context.get("pyramid_built") and self.interlude_chooser is not None:
            return self.interlude_chooser.choose(context.get("available_behaviors"))
        chooser = self.build_chooser if context.get("pyramid_started") else self.setup_chooser
        if chooser is None:
            return None
        return chooser.choose(context.get("available_behaviors"))


class SocializeActivity(Activity):
    __slots__ = [
        "behavior_chooser",
        "interlude_chooser",
        "max_face_iterations",
        "required_objectives",
        "face_iterations",
    ]

    def __init__(self,
                 behavior_chooser: BehaviorChooser,
                 interlude_chooser: BehaviorChooser,
                 max_face_iterations: int,
                 required_objectives: List[Objective],
                 *args, **kwargs):
        self.behavior_chooser = behavior_chooser
        self.interlude_chooser = interlude_chooser
        self.max_face_iterations = int(max_face_iterations)
        self.required_objectives = required_objectives
        self.face_iterations = 0

        super().__init__(*args, **kwargs)

    @classmethod
    def from_json(cls, data: Dict):
        return cls(
            behavior_chooser=BehaviorChooser.from_json(data['behaviorChooser']),
            interlude_chooser=BehaviorChooser.from_json(data['interludeBehaviorChooser']),
            max_face_iterations=data['maxNumFindFacesSearchIterations'],
            required_objectives=[Objective.from_json(d) for d in data.get('requiredObjectives', [])],
            activity_id=data['activityID'],
            activity_type=data['activityType'],
            strategy=data['activityStrategy']['type'])

    def choose(self, context: Optional[Dict[str, Any]] = None,
               rng: Optional[random.Random] = None) -> Optional[str]:
        context = context or {}
        face = bool(context.get("face_visible", context.get("faces", False)))
        if face:
            self.face_iterations += 1
            chooser = self.behavior_chooser
        elif self.face_iterations >= self.max_face_iterations:
            chooser = self.interlude_chooser
        else:
            chooser = self.behavior_chooser
        if chooser is None:
            return None
        return chooser.choose(context.get("available_behaviors"))

    def reset(self) -> None:
        self.face_iterations = 0
        for chooser in (self.behavior_chooser, self.interlude_chooser):
            if chooser is not None:
                chooser.reset()


class NeedsActivity(Activity):
    __slots__ = [
        "behavior_chooser"
    ]

    def __init__(self,
                 behavior_chooser: BehaviorChooser,
                 *args, **kwargs):
        self.behavior_chooser = behavior_chooser
        super().__init__(*args, **kwargs)

    @classmethod
    def from_json(cls, data: Dict):
        return cls(
            behavior_chooser=BehaviorChooser.from_json(data['behaviorChooser']),
            activity_id=data['activityID'],
            activity_type=data['activityType'],
            strategy=data['activityStrategy']['type'])

    def choose(self, context: Optional[Dict[str, Any]] = None,
               rng: Optional[random.Random] = None) -> Optional[str]:
        if self.behavior_chooser is None:
            return None
        return self.behavior_chooser.choose((context or {}).get("available_behaviors"))


def from_dict(info: Dict) -> Activity:
    if info['activityType'] == 'VoiceCommand':
        return VoiceCommandActivity.from_json(info)

    elif info['activityType'] == 'BehaviorsOnly':
        return BehaviorsActivity.from_json(info)

    elif info['activityType'] == 'Feeding':
        return FeedingActivity.from_json(info)

    elif info['activityType'] == 'Freeplay':
        return FreeplayActivity.from_json(info)

    elif info['activityType'] == 'Sparked':
        return SparkedActivity.from_json(info)

    elif info['activityType'] == 'BuildPyramid':
        return PyramidActivity.from_json(info)

    elif info['activityType'] == 'Socialize':
        return SocializeActivity.from_json(info)

    elif info['activityType'] == 'NeedsExpression':
        return NeedsActivity.from_json(info)

    else:
        return Activity(
            activity_id=info['activityID'],
            activity_type=info['activityType'],
            strategy=info['activityStrategy']['type']
        )


def _activity_items(json_data: Any) -> List[Dict]:
    """ Extract activity dictionaries from the resource file shapes. """
    if isinstance(json_data, list):
        return [item for item in json_data if isinstance(item, dict) and 'activityID' in item]
    if not isinstance(json_data, dict):
        return []
    if 'activityID' in json_data:
        return [json_data]
    for key in ('activities', 'activityList'):
        value = json_data.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict) and 'activityID' in item]
    return []


def load_activities(resource_dir: str) -> Dict[str, Activity]:
    """ Load activity map from Cozmo resources.

    The behavior-system activity files are loaded along with the needs-action
    and trick-weight configuration files, which contain activity references
    in several different shapes.  Non-activity sections are ignored rather
    than causing the whole resource set to fail.
    """

    start_time = time.perf_counter()

    activity_folders = [
        os.path.join('cozmo_resources', 'config', 'engine', 'behaviorSystem', 'activities_config.json'),
        os.path.join('cozmo_resources', 'config', 'engine', 'behaviorSystem', 'behavior_system_config.json'),
        os.path.join('cozmo_resources', 'config', 'engine', 'behaviorSystem', 'activities'),
        os.path.join('cozmo_resources', 'config', 'engine', 'needs_action_config.json'),
        os.path.join('cozmo_resources', 'config', 'engine', 'do_a_trick_weights.json'),
    ]
    activity_files = get_json_files(resource_dir, activity_folders)

    activities = {}

    for filename in activity_files:
        try:
            json_data = load_json_file(filename)
        except Exception as exc:
            logger.warning("Failed to load activity file '{}': {}".format(filename, exc))
            continue
        for activity in _activity_items(json_data):
            try:
                activities[activity['activityID']] = from_dict(activity)
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning("Skipping malformed activity in '{}': {}".format(filename, exc))

    logger.debug("Loaded {} activities in {:.02f} s.".format(len(activities), time.perf_counter() - start_time))

    return activities
