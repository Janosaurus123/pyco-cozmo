"""

Brain class - high level behavior and emotion engine.

"""

from typing import Any, Dict, Optional
from threading import Thread
from queue import Queue, Empty
import os
import time

from . import logger, logger_reaction, logger_behavior
from . import client
from . import event
from . import emotions
from . import behavior
from . import activity
from . import personality as personality_module
from . import util
from . import robot


__all__ = [
    "Brain"
]


class Brain:
    """ Cozmo robot brain class. """

    def __init__(self,
                 cli: client.Client,
                 personality: Optional[personality_module.Personality] = None,
                 vision: bool = True,
                 camera_color: bool = False):
        super().__init__()

        self.cli = cli
        self.personality = personality or personality_module.Personality.default()
        self.vision_enabled = bool(vision)
        self.camera_color = bool(camera_color)
        self.vision_pipeline = None
        self.last_vision_result = None
        self._vision_presence = {"face": False, "pet": False, "cube": False,
                                 "platform": False, "motion": False}
        self._vision_frames = 0

        util.check_assets()

        start_time = time.perf_counter()
        resource_dir = str(util.get_cozmo_asset_dir())
        self.resource_dir = resource_dir
        self.activities = activity.load_activities(resource_dir)
        self.behaviors = behavior.load_behaviors(resource_dir, self.cli)
        self.reaction_trigger_beahvior_map = behavior.load_reaction_trigger_behavior_map(resource_dir)
        self.emotion_types = emotions.load_emotion_types(resource_dir)
        self.emotion_events = emotions.load_emotion_events(resource_dir)
        self.cli.load_anims()
        logger.info("Loaded resources in {:.02f} s.".format(time.perf_counter() - start_time))

        self.cli.add_handler(event.EvtBehaviorDone, self.on_behavior_done)
        self.cli.add_handler(event.EvtCliffDetectedChange, self.on_cliff_detected)
        self.cli.add_handler(event.EvtRobotOrientationChange, self.on_robot_orientation_change)
        self.cli.add_handler(event.EvtRobotPickedUpChange, self.on_robot_picked_up_change)
        self.cli.add_handler(event.EvtRobotFallingChange, self.on_robot_falling_change)
        self.cli.add_handler(event.EvtRobotOnChargerChange, self.on_robot_on_charger_change)
        # Correctly spelled alias for the historical attribute name.
        self.reaction_trigger_behavior_map = self.reaction_trigger_beahvior_map

        # Reaction trigger queue
        self.reaction_queue = Queue()

        self.stop_flag = False
        self.reaction_thread = Thread(daemon=True, name="ReactionThread", target=self.reaction_thread_run)
        self.heartbeat_thread = Thread(daemon=True, name="HeartbeatThread", target=self.heartbeat_thread_run)

        # Current activity
        self.activity = self.activities.get("Freeplay") or next(iter(self.activities.values()), None)
        # Current behavior
        self.behavior = None    # type: Optional[behavior.Behavior]
        self.auto_behavior = True
        self._resume_behavior_id = None
        self._camera_handler = None
        self._last_heartbeat = time.monotonic()

    def start(self) -> None:
        """ Start the brain threads and enable the robot-side features. """
        self.stop_flag = False
        if self.reaction_thread is None or not self.reaction_thread.is_alive():
            self.reaction_thread = Thread(daemon=True, name="ReactionThread",
                                          target=self.reaction_thread_run)
            self.reaction_thread.start()
        if self.heartbeat_thread is None or not self.heartbeat_thread.is_alive():
            self.heartbeat_thread = Thread(daemon=True, name="HeartbeatThread",
                                           target=self.heartbeat_thread_run)
            self.heartbeat_thread.start()

        if hasattr(self.cli, "enable_stop_on_cliff"):
            self.cli.enable_stop_on_cliff(True)
        if self.vision_enabled and hasattr(self.cli, "enable_camera"):
            self.cli.enable_camera(True, color=self.camera_color)
        if self._camera_handler is None:
            self._camera_handler = self.cli.add_handler(
                event.EvtNewRawCameraImage, self.on_camera_image)

        if getattr(self.cli, "is_on_charger", False):
            self.post_reaction("PlacedOnCharger")

    def stop(self) -> None:
        """ Stop the brain threads and disable camera streaming. """
        self.stop_flag = True
        if hasattr(self.cli, "enable_camera"):
            try:
                self.cli.enable_camera(False)
            except Exception:
                pass
        if self._camera_handler is not None:
            self.cli.del_handler(event.EvtNewRawCameraImage, self._camera_handler)
            self._camera_handler = None
        if self.heartbeat_thread:
            self.heartbeat_thread.join()
            self.heartbeat_thread = None
        if self.reaction_thread:
            self.reaction_thread.join()
            self.reaction_thread = None

    def on_behavior_done(self, cli: client.Client) -> None:
        if self.behavior:
            logger_reaction.info("Done.")
            self.personality.react("BehaviorDone")
            self.deactivate_behavior()

    def on_cliff_detected(self, cli: client.Client, state: bool) -> None:
        if state:
            self.personality.react("CliffDetected")
            if not cli.robot_picked_up and cli.robot_moving:
                self.post_reaction("CliffDetected")

    def on_robot_orientation_change(self, cli: client.Client, orientation: robot.RobotOrientation) -> None:
        del cli
        orientation_name = getattr(orientation, "name", str(orientation)).upper()
        mapping = {
            "ON_THREADS": "ReturnedToTreads",
            "ON_BACK": "RobotOnBack",
            "ON_FACE": "RobotOnFace",
            "ON_LEFT_SIDE": "RobotOnSide",
            "ON_RIGHT_SIDE": "RobotOnSide",
        }
        trigger = mapping.get(orientation_name)
        if trigger:
            self.personality.react(trigger)
            self.post_reaction(trigger)

    def on_robot_picked_up_change(self, cli: client.Client, state: bool) -> None:
        if state:
            self.personality.react("RobotPickedUp")
            self.post_reaction("RobotPickedUp")

    def on_robot_falling_change(self, cli: client.Client, state: bool):
        if state:
            self.personality.react("RobotFalling")
            self.post_reaction("RobotFalling")

    def on_robot_on_charger_change(self, cli: client.Client, state: bool) -> None:
        if state:
            self.personality.react("PlacedOnCharger")
            self.post_reaction("PlacedOnCharger")

    def _ensure_vision_pipeline(self):
        """ Lazily create the off-board vision pipeline. """
        if self.vision_pipeline is not None:
            return self.vision_pipeline
        from .vision import VisionConfig, VisionPipeline

        config = VisionConfig(
            motion=True,
            faces=True,
            expressions=True,
            cubes=True,
            pets=True,
            robots=True,
            platforms=True,
            track=True,
        )
        self._apply_vision_config_file(config)
        self.vision_pipeline = VisionPipeline(config=config)
        return self.vision_pipeline

    def _apply_vision_config_file(self, config: Any) -> None:
        """ Apply a subset of ``vision_config.json`` to ``config``. """
        filename = os.path.join(self.resource_dir, 'cozmo_resources', 'config',
                                'engine', 'vision_config.json')
        if not os.path.exists(filename):
            return
        try:
            from .json_loader import load_json_file
            data = load_json_file(filename)
        except Exception as exc:
            logger.debug("Failed to load vision configuration: {}".format(exc))
            return
        if not isinstance(data, dict):
            return
        mapping = {
            "motionDetection": "motion",
            "faceDetection": "faces",
            "petDetection": "pets",
            "cubeDetection": "cubes",
            "platformDetection": "platforms",
            "markerDetection": "markers",
            "arucoDetection": "aruco",
            "groundDetection": "ground",
            "robotDetection": "robots",
            "expressionEstimation": "expressions",
        }
        for key, attribute in mapping.items():
            value = data.get(key)
            if isinstance(value, dict):
                value = value.get("enabled", value.get("enable"))
            if isinstance(value, bool):
                setattr(config, attribute, value)

    def _vision_context(self) -> Dict[str, Any]:
        result = self.last_vision_result
        return {
            "face_visible": bool(result and result.faces),
            "pet_visible": bool(result and result.pets),
            "cube_visible": bool(result and result.cubes),
            "platform_visible": bool(result and result.platforms),
            "on_charger": bool(getattr(self.cli, "is_on_charger", False)),
            "picked_up": bool(getattr(self.cli, "robot_picked_up", False)),
            "battery_low": float(getattr(self.cli, "battery_voltage", 4.0)) < 3.5,
            "available_behaviors": set(self.behaviors),
        }

    def _post_if_known(self, trigger: str) -> None:
        if trigger in self.reaction_trigger_beahvior_map:
            self.post_reaction(trigger)
        else:
            logger.debug("No reaction behavior for '{}'.".format(trigger))

    def _update_presence(self, name: str, present: bool, trigger: str) -> None:
        previous = self._vision_presence.get(name, False)
        self._vision_presence[name] = present
        if present and not previous:
            self._post_if_known(trigger)

    def on_camera_image(self, cli: client.Client, new_im) -> None:
        """ Run off-board vision and convert detections into reactions. """
        del cli
        if not self.vision_enabled:
            return
        try:
            pipeline = self._ensure_vision_pipeline()
            result = pipeline.process(new_im)
        except Exception as exc:
            logger.warning("Vision processing failed: {}".format(exc))
            return

        self.last_vision_result = result
        self._vision_frames += 1

        self._update_presence("face", bool(result.faces), "FaceInitialDetection")
        self._update_presence("pet", bool(result.pets), "PetInitialDetection")
        self._update_presence("cube", bool(result.cubes), "CubeInitialDetection")
        self._update_presence("platform", bool(result.platforms), "PlatformInitialDetection")

        smile = any(face.smile > 0.35 for face in result.faces)
        frown = any(face.smile < -0.35 for face in result.faces)
        self._update_presence("smile", smile, "SmileDetected")
        self._update_presence("frown", frown, "FrownDetected")

        if result.motion and not getattr(self.cli, "robot_moving", False):
            self._update_presence("motion", True, "UnexpectedMovement")
        else:
            self._vision_presence["motion"] = False

        if result.faces:
            self.personality.react("FaceDetected")
        if result.pets:
            self.personality.react("PetInitialDetection")
        if result.cubes:
            self.personality.react("CubeInitialDetection")

    def post_reaction(self, reaction_trigger: str) -> None:
        """ Post a reaction trigger to the reaction trigger queue. """
        logger_reaction.debug("Posting {}".format(reaction_trigger))
        self.reaction_queue.put(reaction_trigger)

    def reaction_thread_run(self) -> None:
        """ Reaction thread loop. Reaction trigger queue processing. """
        while not self.stop_flag:
            try:
                reaction_trigger = self.reaction_queue.get(timeout=0.05)
            except Empty:
                continue
            except Exception as e:
                logger.error("Failed to get from reaction trigger queue. {}".format(e))
                continue

            try:
                self.process_reaction(reaction_trigger)
            except Exception as e:
                logger.error("Failed to dispatch reaction trigger '{}'. {}".format(reaction_trigger, e))
                continue

    def process_reaction(self, reaction_trigger: str) -> None:
        logger_reaction.info("Processing {}".format(reaction_trigger))
        self.personality.react(reaction_trigger)
        reaction = self.reaction_trigger_beahvior_map.get(reaction_trigger)
        if reaction:
            if reaction.should_resume_last and self.behavior is not None:
                self._resume_behavior_id = self.behavior.get_id()
            self.activate_behavior(reaction.behavior_id, allow_auto=False)
        else:
            logger_reaction.error("Failed to find reaction for {}.".format(reaction_trigger))

    def activate_behavior(self, behavior_id: str, allow_auto: bool = True) -> None:
        behavior = self.behaviors.get(behavior_id)
        if behavior:
            self.deactivate_behavior(allow_auto=False)
            logger_behavior.info("Activating {}".format(behavior_id))
            self.behavior = behavior
            self.cli.activate_behavior(self.behavior)
        else:
            logger_reaction.error("Failed to find behavior {}.".format(behavior_id))
            if allow_auto:
                self._choose_next_behavior()

    def deactivate_behavior(self, allow_auto: bool = True) -> None:
        if self.behavior:
            logger_behavior.info("Deactivating {}".format(self.behavior.get_id()))
            self.cli.deactivate_behavior(self.behavior)
            self.behavior = None
        if not allow_auto or not self.auto_behavior or self.stop_flag:
            return
        if self._resume_behavior_id:
            behavior_id = self._resume_behavior_id
            self._resume_behavior_id = None
            if behavior_id in self.behaviors:
                self.activate_behavior(behavior_id, allow_auto=False)
                return
        self._choose_next_behavior()

    def _choose_next_behavior(self) -> None:
        """ Ask the current activity (then personality) for the next behavior. """
        if not self.behaviors or self.stop_flag:
            return
        context = self._vision_context()
        choice = None
        if self.activity is not None:
            try:
                choice = self.activity.choose(context, self.personality.rng)
            except Exception as exc:
                logger.error("Activity '{}' failed to choose: {}".format(self.activity.id, exc))
        if choice in self.activities and choice != getattr(self.activity, "id", None):
            self.activity = self.activities[choice]
            try:
                choice = self.activity.choose(context, self.personality.rng)
            except Exception as exc:
                logger.error("Activity '{}' failed to choose: {}".format(choice, exc))
                choice = None
        if choice not in self.behaviors:
            choice = self.personality.choose_behavior(
                list(self.behaviors.values()), context, rng=self.personality.rng)
            if choice is not None:
                choice = choice.get_id()
        if choice in self.behaviors:
            self.activate_behavior(choice, allow_auto=False)

    def set_activity(self, activity_id: str) -> bool:
        """ Switch the current activity by ID. """
        if activity_id not in self.activities:
            return False
        if self.activity is not None and self.activity.id == activity_id:
            return True
        self.activity = self.activities[activity_id]
        return True

    def heartbeat_thread_run(self) -> None:
        """ Heartbeat thread loop. """

        # Raise head.
        angle = (robot.MAX_HEAD_ANGLE.radians - robot.MIN_HEAD_ANGLE.radians) / 2.0
        self.cli.set_head_angle(angle)

        cnt = 1
        timer = util.FPSTimer(robot.FRAME_RATE)
        self._last_heartbeat = time.monotonic()
        while not self.stop_flag:

            now = time.monotonic()
            dt = now - self._last_heartbeat
            self._last_heartbeat = now
            self.personality.update(dt)
            self.update_emotion_types(dt)

            if cnt % (30 * 60) == 0:
                self.post_reaction("Hiccup")

            cnt += 1
            timer.sleep()

    def update_emotion_types(self, dt: Optional[float] = None) -> None:
        """ Update resource emotion types from the personality state. """
        for name, emotion_type in self.emotion_types.items():
            key = name.lower()
            if key in self.personality.emotions:
                emotion_type.set(self.personality.emotions[key])
            else:
                emotion_type.update(dt)
