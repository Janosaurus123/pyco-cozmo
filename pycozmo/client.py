"""

Cozmo protocol client and high-level API.

"""

from threading import Event, Timer
from typing import Optional, Tuple
import json
import math
import time
import io

import numpy as np
from PIL import Image

from . import logger, logger_robot, logger_animation
from . import protocol_encoder
from .protocol_base import UnknownCommand
from . import event
from . import camera
from . import object
from . import util
from . import robot
from . import exception
from . import filter
from . import protocol_declaration
from . import conn
from . import lights
from . import image_encoder
from . import anim
from . import anim_encoder
from . import audio
from . import anim_controller
from . import robot_debug


__all__ = [
    "Client",
]


class Client(event.Dispatcher):
    """ Cozmo protocol client and high-level API class. """

    def __init__(self,
                 robot_addr: Optional[Tuple[str, int]] = None,
                 protocol_log_messages: Optional[list] = None,
                 auto_initialize: bool = True,
                 enable_animations: bool = True,
                 enable_procedural_face: bool = True) -> None:
        super().__init__()
        # Whether to automatically initialize the robot when connection is established.
        self.auto_initialize = bool(auto_initialize)

        self.conn = conn.Connection(robot_addr, protocol_log_messages)
        self.conn.add_child_dispatcher(self)
        self.anim_controller = anim_controller.AnimationController(self)
        self.anim_controller.enable_animations(auto_initialize and enable_animations)
        self.anim_controller.enable_procedural_face(auto_initialize and enable_animations and enable_procedural_face)

        self.serial_number_head = None
        self.robot_fw_sig = None
        self.serial_number = None
        self.body_hw_version = None
        self.body_color = None
        # Robot state
        # Heading in X-Y plane.
        self.pose_frame_id = 0
        self.pose = util.Pose(0.0, 0.0, 0.0, angle_z=util.Angle(radians=0.0), origin_id=1)
        self.pose_pitch = util.Angle(radians=0.0)
        self.head_angle = util.Angle(radians=robot.MIN_HEAD_ANGLE.radians)
        self.left_wheel_speed = util.Speed(mmps=0.0)
        self.right_wheel_speed = util.Speed(mmps=0.0)
        self.lift_position = robot.LiftPosition(height=robot.MIN_LIFT_HEIGHT)
        self.battery_voltage = 0.0
        self.accel = util.Vector3(0.0, 0.0, 0.0)
        self.gyro = util.Vector3(0.0, 0.0, 0.0)
        self.robot_status = 0
        self.robot_orientation = robot.RobotOrientation.ON_THREADS
        self.robot_picked_up = False
        self.robot_moving = False
        self.cliff_data_raw = (0, 0, 0, 0)
        self.backpack_touch_sensor_raw = 0
        self.current_path_segment = 0
        self.lift_motor_enabled = None
        self._lift_was_commanded = False
        self._manual_image_timer = None
        # Animation state
        self.num_anim_bytes_played = 0
        self.num_audio_frames_played = 0
        self.enabled_anim_tracks = 0
        self.tag = 0
        self.client_drop_count = 0
        # Camera state
        self.last_image_timestamp = None
        self._latest_image = None
        # Object state
        self.available_objects = dict()
        self.connected_objects = dict()
        # Filters
        self.packet_type_filter = filter.Filter()
        self.packet_type_filter.deny_ids({protocol_declaration.PacketType.PING.value})
        self.packet_id_filter = filter.Filter()
        self._reset_partial_state()
        # Animations
        self._clip_metadata = {}
        self._clips = {}
        self._ppclips = {}
        self._next_anim_id = 1
        self.animation_groups = {}

    def start(self) -> None:
        logger.debug("Starting client...")
        self.add_handler(protocol_encoder.HardwareInfo, self._on_hardware_info)
        self.add_handler(protocol_encoder.FirmwareSignature, self._on_firmware_signature)
        self.add_handler(protocol_encoder.BodyInfo, self._on_body_info)
        self.add_handler(protocol_encoder.ImageChunk, self._on_image_chunk)
        self.add_handler(protocol_encoder.RobotState, self._on_robot_state)
        self.add_handler(protocol_encoder.AnimationState, self._on_animation_state)
        self.add_handler(protocol_encoder.ObjectAvailable, self._on_object_available)
        self.add_handler(protocol_encoder.ObjectConnectionState, self._on_object_connection_state)
        self.add_handler(protocol_encoder.DebugData, self._on_debug_data)
        self.add_handler(UnknownCommand, self._on_unknown_command)
        self.add_handler(event.EvtRobotPickedUpChange, self._on_robot_picked_up)
        self.add_handler(event.EvtRobotWheelsMovingChange, self._on_robot_moving)
        self.conn.start()

    def stop(self) -> None:
        logger.debug("Stopping client...")
        self.conn.stop()
        self.anim_controller.stop()
        self.del_all_handlers()

    def connect(self) -> None:
        logger.debug("Connecting...")
        self.conn.connect()

    def disconnect(self) -> None:
        logger.debug("Disconnecting...")
        self.conn.disconnect()

    def _enable_robot(self):
        # Enable
        pkt = protocol_encoder.Enable()
        self.conn.send(pkt)
        self.conn.send(pkt)  # This repetition seems to trigger BodyInfo

    def _initialize_robot(self):
        # Set world frame origin to (0,0,0), frame ID to 0, and origin ID to 1.
        pkt = protocol_encoder.SetOrigin()
        self.conn.send(pkt)
        # Set timestamp to 0. Also enables RobotState and ObjectAvailable events. Requires Enable (0x25).
        pkt = protocol_encoder.SyncTime()
        self.conn.send(pkt)

        # TODO: Proper waiting for motor calibration to complete.
        time.sleep(0.5)

        self.anim_controller.start()

        self.dispatch(event.EvtRobotReady, self)

    def _on_hardware_info(self, cli, pkt: protocol_encoder.HardwareInfo):
        del cli
        self.serial_number_head = pkt.serial_number_head

    def _on_firmware_signature(self, cli, pkt: protocol_encoder.FirmwareSignature):
        del cli
        self.robot_fw_sig = json.loads(pkt.signature)
        logger.info("Firmware version %s.", self.robot_fw_sig["version"])
        if self.robot_fw_sig.get("build") == "FACTORY":
            logger.warning("Factory/recovery firmware detected. Functionality is degraded.")
        elif self.robot_fw_sig["version"] < protocol_declaration.FIRMWARE_VERSION:
            logger.warning(
                "Old firmware detected. PyCozmo works best with v{}. Functionality may be degraded.".format(
                    protocol_declaration.FIRMWARE_VERSION))
        self._enable_robot()

    def _on_body_info(self, cli, pkt: protocol_encoder.BodyInfo):
        del cli
        self.serial_number = pkt.serial_number
        self.body_hw_version = pkt.body_hw_version
        self.body_color = pkt.body_color
        logger.info("Body S/N 0x%08x, HW version %i, color %i.",
                    self.serial_number, self.body_hw_version, self.body_color.value)
        if self.auto_initialize:
            self._initialize_robot()
        self.dispatch(event.EvtRobotFound, self)

    def wait_for(self, evt, timeout: Optional[float] = None):
        e = Event()
        self.add_handler(evt, lambda cli: e.set(), one_shot=True)
        if not e.wait(timeout):
            raise exception.Timeout("Timeout waiting for event {}".format(evt))

    def wait_for_robot(self, timeout: float = 5.0) -> None:
        if not self.robot_fw_sig:
            try:
                self.wait_for(event.EvtRobotFound, timeout=timeout)
            except exception.Timeout as e:
                raise exception.ConnectionTimeout("Failed to connect to Cozmo.") from e

        if not self.serial_number:
            try:
                self.wait_for(event.EvtRobotReady, timeout=timeout)
            except exception.Timeout as e:
                raise exception.ConnectionTimeout("Failed to initialize Cozmo.") from e

    def _reset_partial_state(self):
        self._partial_image_timestamp = None
        self._partial_data = None
        self._partial_image_id = None
        self._partial_invalid = False
        self._partial_size = 0
        self._partial_image_encoding = None
        self._partial_image_resolution = None
        self._last_chunk_id = -1

    def _on_image_chunk(self, cli, pkt: protocol_encoder.ImageChunk):
        del cli
        # The timestamp is refreshed from every chunk.  Older firmware reports
        # zero in the first chunk of an image and the real timestamp in the
        # later chunks (see upstream issue #55).
        if pkt.frame_timestamp:
            self._partial_image_timestamp = pkt.frame_timestamp
        if self._partial_image_id is not None and pkt.chunk_id == 0:
            if not self._partial_invalid:
                logger.debug("Lost final chunk of image - discarding.")
            self._partial_image_id = None

        if self._partial_image_id is None:
            if pkt.chunk_id != 0:
                if not self._partial_invalid:
                    logger.debug("Received chunk of broken image.")
                self._partial_invalid = True
                return
            # Discard any previous in-progress image
            self._reset_partial_state()
            self._partial_image_timestamp = pkt.frame_timestamp
            self._partial_image_id = pkt.image_id
            self._partial_image_encoding = protocol_encoder.ImageEncoding(pkt.image_encoding)
            self._partial_image_resolution = protocol_encoder.ImageResolution(pkt.image_resolution)

            image_resolution = protocol_encoder.ImageResolution(pkt.image_resolution)
            width, height = camera.RESOLUTIONS[image_resolution]
            max_size = width * height * 3  # 3 bytes per pixel (RGB)
            self._partial_data = np.empty(max_size, dtype=np.uint8)

        if pkt.chunk_id != (self._last_chunk_id + 1) or pkt.image_id != self._partial_image_id:
            logger.debug("Image missing chunks - discarding (last_chunk_id=%d partial_image_id=%s).",
                         self._last_chunk_id, self._partial_image_id)
            self._reset_partial_state()
            self._partial_invalid = True
            return

        offset = self._partial_size
        self._partial_data[offset:offset + len(pkt.data)] = np.frombuffer(pkt.data, dtype=np.uint8)
        self._partial_size += len(pkt.data)
        self._last_chunk_id = pkt.chunk_id

        if pkt.chunk_id == pkt.image_chunk_count - 1:
            self._process_completed_image()
            self._reset_partial_state()

    def _process_completed_image(self):
        data = self._partial_data[0:self._partial_size]

        # The first byte of the image is whether or not it is in color
        is_color_image = data[0] != 0

        if self._partial_image_encoding == protocol_encoder.ImageEncoding.JPEGMinimizedGray:
            width, height = camera.RESOLUTIONS[self._partial_image_resolution]

            if is_color_image:
                # Color images are half width
                width = width // 2
                data = camera.minicolor_to_jpeg(data, width, height)
            else:
                data = camera.minigray_to_jpeg(data, width, height)

        image = Image.open(io.BytesIO(data)).convert('RGB')

        # Color images need to be resized to the proper resolution
        if is_color_image:
            size = camera.RESOLUTIONS[self._partial_image_resolution]
            image = image.resize(size)

        self._latest_image = image
        self.last_image_timestamp = self._partial_image_timestamp
        self.dispatch(event.EvtNewRawCameraImage, self, image)

    def _on_robot_state(self, cli, pkt: protocol_encoder.RobotState):
        del cli
        self.pose_frame_id = pkt.pose_frame_id
        self.pose = util.Pose(pkt.pose_x, pkt.pose_y, pkt.pose_z,
                              angle_z=util.Angle(radians=pkt.pose_angle_rad), origin_id=pkt.pose_origin_id)
        self.pose_pitch = util.Angle(radians=pkt.pose_pitch_rad)
        self.head_angle = util.Angle(radians=pkt.head_angle_rad)
        self.left_wheel_speed = util.Speed(mmps=pkt.lwheel_speed_mmps)
        self.right_wheel_speed = util.Speed(mmps=pkt.rwheel_speed_mmps)
        self.lift_position = robot.LiftPosition(height=util.Distance(mm=pkt.lift_height_mm))
        self.battery_voltage = pkt.battery_voltage
        self.accel = util.Vector3(pkt.accel_x, pkt.accel_y, pkt.accel_z)
        self.gyro = util.Vector3(pkt.gyro_x, pkt.gyro_y, pkt.gyro_z)
        self.cliff_data_raw = tuple(pkt.cliff_data_raw)
        self.backpack_touch_sensor_raw = pkt.backpack_touch_sensor_raw
        self.current_path_segment = pkt.curr_path_segment
        old_status = self.robot_status
        self.robot_status = pkt.status
        self.dispatch(event.EvtRobotStateUpdated, self)
        # Dispatch status flag change events.
        for flag, evt in event.STATUS_EVENTS.items():
            if (old_status & flag) != (pkt.status & flag):
                state = (pkt.status & flag) != 0
                logger.debug("%s: %i", robot.RobotStatusFlagNames[flag], state)
                self.dispatch(evt, self, state)
        # Orientation
        if pkt.pose_angle_rad < -0.4:
            robot_orientation = robot.RobotOrientation.ON_LEFT_SIDE
        elif pkt.pose_angle_rad > 0.4:
            robot_orientation = robot.RobotOrientation.ON_RIGHT_SIDE
        elif pkt.pose_pitch_rad < -1.0:
            robot_orientation = robot.RobotOrientation.ON_FACE
        elif pkt.pose_pitch_rad > 1.0:
            robot_orientation = robot.RobotOrientation.ON_BACK
        else:
            robot_orientation = robot.RobotOrientation.ON_THREADS
        if self.robot_orientation != robot_orientation:
            self.robot_orientation = robot_orientation
            self.dispatch(event.EvtRobotOrientationChange, self, robot_orientation)

    def _on_robot_picked_up(self, cli, state):
        del cli
        if state:
            self.robot_picked_up = True
        else:
            # Robot put down - reset world frame origin.
            self.robot_picked_up = False
            pkt = protocol_encoder.SetOrigin(
                pose_frame_id=self.pose_frame_id + 1, pose_origin_id=self.pose.origin_id + 1)
            self.conn.send(pkt)

    def _on_robot_moving(self, cli, state):
        self.robot_moving = state

    def _on_animation_state(self, cli, pkt: protocol_encoder.AnimationState):
        del cli
        self.num_anim_bytes_played = pkt.num_anim_bytes_played
        self.num_audio_frames_played = pkt.num_audio_frames_played
        self.enabled_anim_tracks = pkt.enabled_anim_tracks
        self.tag = pkt.tag
        self.client_drop_count = pkt.client_drop_count

    def _on_object_available(self, cli, pkt: protocol_encoder.ObjectAvailable):
        del cli
        factory_id = pkt.factory_id
        object_type = protocol_encoder.ObjectType(pkt.object_type)
        obj = object.Object(factory_id=factory_id, object_type=object_type)
        if factory_id not in self.available_objects:
            self.available_objects[factory_id] = obj
            logger.debug("Object of type %s with S/N 0x%08x available.", str(obj.object_type), obj.factory_id)

    def _on_object_connection_state(self, cli, pkt: protocol_encoder.ObjectConnectionState):
        del cli
        if pkt.connected:
            # Connected
            self.connected_objects[pkt.object_id] = {
                "factory_id": pkt.factory_id,
                "object_type": pkt.object_type,
            }
        else:
            # Disconnected
            if pkt.object_id in self.connected_objects:
                del self.connected_objects[pkt.object_id]

    def _on_debug_data(self, cli, pkt: protocol_encoder.DebugData):
        del cli
        msg = robot_debug.get_debug_message(pkt.name_id, pkt.format_id, pkt.args)
        logger_robot.log(robot_debug.get_log_level(pkt.level), msg)

    def _on_unknown_command(self, cli, pkt: UnknownCommand):
        """Handle protocol commands which are not (yet) formally declared.

        Command 0xd8 is emitted by newer firmware and indicates whether the
        lift motor is enabled.  When it is disabled while the application is
        trying to move the lift, the robot has most likely detected that the
        motor is overpowered (for example, when the lift is blocked).
        """
        del cli
        if pkt.id == 0xd8 and pkt.data:
            enabled = bool(pkt.data[-1])
            if self.lift_motor_enabled is not enabled:
                self.lift_motor_enabled = enabled
                logger.debug("Lift motor enabled: %s", enabled)
                self.dispatch(event.EvtLiftMotorEnabledChange, self, enabled)
                if not enabled and self._lift_was_commanded:
                    self.dispatch(event.EvtLiftOverpowered, self)
                    self._lift_was_commanded = False

    def _status_flag(self, flag: int) -> bool:
        return bool(self.robot_status & flag)

    @property
    def is_moving(self) -> bool:
        return self._status_flag(robot.RobotStatusFlag.IS_MOVING)

    @property
    def is_carrying_block(self) -> bool:
        return self._status_flag(robot.RobotStatusFlag.IS_CARRYING_BLOCK)

    @property
    def is_picking_or_placing(self) -> bool:
        return self._status_flag(robot.RobotStatusFlag.IS_PICKING_OR_PLACING)

    @property
    def is_picked_up(self) -> bool:
        return self._status_flag(robot.RobotStatusFlag.IS_PICKED_UP)

    @property
    def is_falling(self) -> bool:
        return self._status_flag(robot.RobotStatusFlag.IS_FALLING)

    @property
    def is_animating(self) -> bool:
        return self._status_flag(robot.RobotStatusFlag.IS_ANIMATING)

    @property
    def is_pathing(self) -> bool:
        return self._status_flag(robot.RobotStatusFlag.IS_PATHING)

    @property
    def is_on_charger(self) -> bool:
        return self._status_flag(robot.RobotStatusFlag.IS_ON_CHARGER)

    @property
    def latest_image(self):
        """ The most recent camera image, or ``None`` before the first frame. """
        return self._latest_image

    @property
    def is_charging(self) -> bool:
        return self._status_flag(robot.RobotStatusFlag.IS_CHARGING)

    @property
    def is_cliff_detected(self) -> bool:
        return self._status_flag(robot.RobotStatusFlag.CLIFF_DETECTED)

    @property
    def lift_in_position(self) -> bool:
        return self._status_flag(robot.RobotStatusFlag.LIFT_IN_POS)

    @property
    def head_in_position(self) -> bool:
        return self._status_flag(robot.RobotStatusFlag.HEAD_IN_POS)

    @property
    def wheels_moving(self) -> bool:
        return self._status_flag(robot.RobotStatusFlag.ARE_WHEELS_MOVING)

    @property
    def accelerometer(self) -> util.Vector3:
        """Acceleration along the x, y and z axes."""
        return self.accel

    @property
    def gyro_reading(self) -> util.Vector3:
        """Angular velocity around the x, y and z axes."""
        return self.gyro

    @property
    def battery_voltage_volts(self) -> float:
        return self.battery_voltage

    def set_head_angle(self, angle: float, accel: float = 10.0, max_speed: float = 10.0,
                       duration: float = 0.0):
        pkt = protocol_encoder.SetHeadAngle(angle_rad=angle, accel_rad_per_sec2=accel,
                                            max_speed_rad_per_sec=max_speed, duration_sec=duration)
        self.conn.send(pkt)

    def move_head(self, speed: float) -> None:
        pkt = protocol_encoder.MoveHead(speed_rad_per_sec=speed)
        self.conn.send(pkt)

    def set_lift_height(self, height: float, accel: float = 10.0, max_speed: float = 10.0,
                        duration: float = 0.0):
        self._lift_was_commanded = True
        pkt = protocol_encoder.SetLiftHeight(height_mm=height, accel_rad_per_sec2=accel,
                                             max_speed_rad_per_sec=max_speed, duration_sec=duration)
        self.conn.send(pkt)

    def move_lift(self, speed: float) -> None:
        self._lift_was_commanded = True
        pkt = protocol_encoder.MoveLift(speed_rad_per_sec=speed)
        self.conn.send(pkt)

    def start_motor_calibration(self, head: bool = True, lift: bool = True) -> None:
        """Start calibration of the head and/or lift motors."""
        pkt = protocol_encoder.StartMotorCalibration(head=head, lift=lift)
        self.conn.send(pkt)

    def drive_wheels(self, lwheel_speed: float, rwheel_speed: float,
                     lwheel_acc: Optional[float] = 0.0, rwheel_acc: Optional[float] = 0.0,
                     duration: Optional[float] = None) -> None:
        pkt = protocol_encoder.DriveWheels(lwheel_speed_mmps=lwheel_speed, rwheel_speed_mmps=rwheel_speed,
                                           lwheel_accel_mmps2=lwheel_acc, rwheel_accel_mmps2=rwheel_acc)
        self.conn.send(pkt)
        if duration is not None:
            time.sleep(duration)
            self.stop_all_motors()

    def stop_all_motors(self) -> None:
        self._lift_was_commanded = False
        pkt = protocol_encoder.StopAllMotors()
        self.conn.send(pkt)

    def drive_straight(self, speed: float, duration: Optional[float] = None,
                       lwheel_speed: Optional[float] = None,
                       rwheel_speed: Optional[float] = None) -> None:
        """Drive both wheels at the same speed.

        Args:
            speed: Forward speed in mm/s (negative values drive backwards).
            duration: If provided, stop the motors after this many seconds.
            lwheel_speed: Optional override for the left wheel.
            rwheel_speed: Optional override for the right wheel.
        """
        self.drive_wheels(
            lwheel_speed if lwheel_speed is not None else speed,
            rwheel_speed if rwheel_speed is not None else speed,
            duration=duration)

    def turn_in_place(self, angle: float, speed: float = 1.0,
                      accel: float = 2.0, tolerance: float = 0.01) -> None:
        """Turn to an absolute heading in radians.

        Args:
            angle: Target heading in radians.
            speed: Angular speed in radians per second.
            accel: Angular acceleration in radians per second squared.
            tolerance: Heading tolerance in radians.
        """
        pkt = protocol_encoder.TurnInPlace(
            angle_rad=angle, speed_rad_per_sec=speed, accel_rad_per_sec2=accel,
            angle_tolerance_rad=tolerance, unknown4=0, unknown5=0,
            is_absolute=True, action_id=0)
        self.conn.send(pkt)

    def turn_in_place_at_speed(self, speed: float, direction: int = 1,
                               accel: float = 2.0) -> None:
        """Rotate continuously at a given wheel speed.

        Args:
            speed: Wheel speed in mm/s.
            direction: 1 for counter-clockwise, -1 for clockwise.
            accel: Wheel acceleration in mm/s2.
        """
        pkt = protocol_encoder.TurnInPlaceAtSpeed(
            wheel_speed_mmps=abs(speed), wheel_accel_mmps2=accel,
            direction=1 if direction >= 0 else -1)
        self.conn.send(pkt)

    def drive_wheel_motors(self, lwheel_speed: float, rwheel_speed: float,
                           duration: Optional[float] = None) -> None:
        """Alias for :meth:`drive_wheels`, provided for SDK compatibility."""
        self.drive_wheels(lwheel_speed, rwheel_speed, duration=duration)

    def enable_stop_on_cliff(self, enable: bool = True) -> None:
        pkt = protocol_encoder.EnableStopOnCliff(enable=enable)
        self.conn.send(pkt)

    def set_camera_params(self, gain: float = 0.0, exposure_ms: int = 0,
                          auto_exposure_enabled: bool = True) -> None:
        pkt = protocol_encoder.SetCameraParams(
            gain=gain, exposure_ms=exposure_ms,
            auto_exposure_enabled=auto_exposure_enabled)
        self.conn.send(pkt)

    def set_manual_exposure(self, gain: float, exposure_ms: int) -> None:
        """Set a fixed camera gain and exposure time."""
        self.set_camera_params(gain=gain, exposure_ms=exposure_ms,
                               auto_exposure_enabled=False)

    def set_robot_volume(self, level: int) -> None:
        """Alias for :meth:`set_volume`."""
        self.set_volume(level)

    def drive_off_charger_contacts(self, speed: float = -50.0,
                                   duration: float = 1.5) -> None:
        """Drive backwards off the charging contacts.

        The robot does not expose a dedicated command for this, so the motion
        is time based.  Cliff detection is temporarily disabled because the
        charger contacts can otherwise be mistaken for a cliff.
        """
        was_on_charger = self.is_on_charger
        self.enable_stop_on_cliff(False)
        try:
            self.drive_wheels(speed, speed, duration=duration)
        finally:
            self.enable_stop_on_cliff(True)
        if was_on_charger:
            logger.debug("Drove off charger contacts.")

    def drive_onto_charger(self, speed: float = 50.0, timeout: float = 5.0) -> bool:
        """Drive forward until the robot reports that it is on the charger.

        Returns ``True`` if the charger was reached before ``timeout``.
        """
        if self.is_on_charger:
            return True
        reached = Event()
        result = False
        handler = self.add_handler(
            event.EvtRobotOnChargerChange,
            lambda _cli, state: reached.set() if state else None)
        self.enable_stop_on_cliff(False)
        try:
            self.drive_wheels(speed, speed, lwheel_acc=20.0, rwheel_acc=20.0)
            result = reached.wait(timeout)
        finally:
            self.stop_all_motors()
            self.enable_stop_on_cliff(True)
            self.del_handler(event.EvtRobotOnChargerChange, handler)
        return bool(result)

    def backup_onto_charger(self, speed: float = -50.0, timeout: float = 5.0) -> bool:
        """Back up onto the charger, returning whether it was reached."""
        return self.drive_onto_charger(speed=speed, timeout=timeout)

    def wait_for_pose(self, pose: util.Pose, tolerance: float = 10.0,
                      angle_tolerance: float = 0.1, timeout: float = 30.0) -> None:
        """Wait until the robot reaches a pose within the given tolerances."""
        reached = Event()

        def on_state(_cli):
            dx = self.pose.position.x - pose.position.x
            dy = self.pose.position.y - pose.position.y
            da = self.pose.rotation.angle_z.radians - pose.rotation.angle_z.radians
            da = math.atan2(math.sin(da), math.cos(da))
            if math.hypot(dx, dy) <= tolerance and abs(da) <= angle_tolerance:
                reached.set()

        handler = self.add_handler(event.EvtRobotStateUpdated, on_state)
        try:
            if not reached.wait(timeout):
                raise exception.Timeout("Timeout waiting for pose.")
        finally:
            self.del_handler(event.EvtRobotStateUpdated, handler)

    def go_to_pose(self, pose: util.Pose, relative_to_robot: bool = False,
                   timeout: Optional[float] = 60.0,
                   speed: float = 100.0, turn_speed: float = 40.0) -> None:
        """ Move to a specific pose (position and orientation). """

        if relative_to_robot:
            pose = util.Pose(self.pose.position.x, self.pose.position.y,
                             self.pose.position.z, angle_z=self.pose.rotation.angle_z).define_pose_relative_this(pose)

        pkt = protocol_encoder.AppendPathSegLine(
            from_x=self.pose.position.x, from_y=self.pose.position.y,
            to_x=pose.position.x, to_y=pose.position.y,
            speed_mmps=speed, accel_mmps2=20.0, decel_mmps2=20.0)
        self.conn.send(pkt)
        pkt = protocol_encoder.AppendPathSegPointTurn(
            x=pose.position.x, y=pose.position.y,
            angle_rad=pose.rotation.angle_z.radians,
            angle_tolerance_rad=0.01,
            speed_mmps=turn_speed, accel_mmps2=20.0, decel_mmps2=20.0)
        self.conn.send(pkt)
        event_id = (self.tag + 1) & 0xffff
        pkt = protocol_encoder.ExecutePath(event_id=event_id)
        self.conn.send(pkt)

        e = Event()

        def event_wait(_, pkt2: protocol_encoder.PathFollowingEvent):
            if pkt2.event_id == event_id and pkt2.event_type != protocol_encoder.PathEventType.PATH_STARTED:
                e.set()

        handler = self.add_handler(protocol_encoder.PathFollowingEvent, event_wait)
        try:
            if not e.wait(timeout):
                raise exception.Timeout("Timeout waiting for path following to complete.")
        finally:
            self.del_handler(protocol_encoder.PathFollowingEvent, handler)

    def set_backpack_lights(self, left_light, front_light, center_light, rear_light, right_light) -> None:
        pkt = protocol_encoder.LightStateCenter(states=(front_light, center_light, rear_light))
        self.conn.send(pkt)
        pkt = protocol_encoder.LightStateSide(states=(left_light, right_light))
        self.conn.send(pkt)

    def set_center_backpack_lights(self, light) -> None:
        self.set_backpack_lights(lights.off_light, light, light, light, lights.off_light)

    def set_all_backpack_lights(self, light) -> None:
        self.set_backpack_lights(light, light, light, light, light)

    def set_backpack_lights_off(self) -> None:
        self.set_backpack_lights(lights.off_light, lights.off_light, lights.off_light,
                                 lights.off_light, lights.off_light)

    def set_head_light(self, enable: bool) -> None:
        pkt = protocol_encoder.SetHeadLight(enable=enable)
        self.conn.send(pkt)

    def enable_camera(self, enable: bool = True, color: bool = False) -> None:
        """ Enable or disable camera image streaming in color or grayscale. """
        image_send_mode = protocol_encoder.ImageSendMode.Stream if enable else protocol_encoder.ImageSendMode.Off
        pkt = protocol_encoder.EnableCamera(image_send_mode=image_send_mode)
        self.conn.send(pkt)
        pkt = protocol_encoder.EnableColorImages(enable=color)
        self.conn.send(pkt)

    def _cancel_manual_image_timer(self) -> None:
        if self._manual_image_timer is not None:
            self._manual_image_timer.cancel()
            self._manual_image_timer = None

    def clear_screen(self) -> None:
        """Clear the OLED display and resume procedural faces."""
        self._cancel_manual_image_timer()
        self.anim_controller.clear_image()
        pkt = protocol_encoder.DisplayImage(image=b"\x3f\x3f")
        self.anim_controller.display_image(pkt, manual=False)

    def display_image(self, im: Image, duration: Optional[float] = None) -> None:
        """Display a still image on the robot's OLED display.

        If ``duration`` is provided the image is cleared after that many
        seconds.  Without a duration the image remains until
        :meth:`clear_screen` is called or another image is displayed.  The
        procedural face generator is suspended while a manual image is shown
        (upstream issue #48).
        """
        self._cancel_manual_image_timer()
        encoder = image_encoder.ImageEncoder(im)
        buf = bytes(encoder.encode())
        pkt = protocol_encoder.DisplayImage(image=buf)
        self.anim_controller.display_image(pkt, manual=True)
        if duration is not None:
            self._manual_image_timer = Timer(duration, self.clear_screen)
            self._manual_image_timer.daemon = True
            self._manual_image_timer.start()

    def _load_clips(self, fspec: str) -> None:

        start_time = time.perf_counter()

        if fspec.endswith(".bin"):
            clips = anim_encoder.AnimClips.from_fb_file(fspec)
        elif fspec.endswith(".json"):
            clips = anim_encoder.AnimClips.from_json_file(fspec)
        else:
            raise ValueError("Unsupported animation file format.")
        for clip in clips.clips:
            self._clips[clip.name] = clip

        logger.debug("Loaded {} in {:.02f} s.".format(fspec, time.perf_counter() - start_time))

    def play_anim_ppclip(self, ppclip: anim.PreprocessedClip) -> None:

        # Ensure no other animation is playing.
        self.cancel_anim()

        # Start animation.
        pkt = protocol_encoder.StartAnimation(anim_id=self._next_anim_id)
        self.anim_controller.play_anim_frame(None, None, (pkt, ))
        self._next_anim_id += 1

        # Send frames to the animation controller.
        frames = list(sorted(ppclip.keyframes.keys()))
        num_frames = len(frames)
        time_ms = 0
        for i in range(num_frames):
            audio_pkt = None
            image_pkt = None
            pkts = []
            for action in ppclip.keyframes[frames[i]]:
                if isinstance(action, protocol_encoder.OutputAudio):
                    audio_pkt = action
                elif isinstance(action, protocol_encoder.DisplayImage):
                    image_pkt = action
                elif isinstance(action, protocol_encoder.Packet):
                    pkts.append(action)
            self.anim_controller.play_anim_frame(audio_pkt, image_pkt, pkts)
            time_ms += 33

            # Pause.
            if i < num_frames - 1:
                target_ms = time_ms + frames[i + 1] - frames[i]
                while target_ms > time_ms:
                    self.anim_controller.play_anim_frame(None, None, None)
                    time_ms += 33

        # End animation.
        pkt = protocol_encoder.EndAnimation()
        self.anim_controller.play_anim_frame(None, None, (pkt, ))

    def play_anim(self, name: str) -> None:
        if not self._clip_metadata:
            raise ValueError("Animations not loaded.")
        elif name not in self._clip_metadata:
            raise ValueError("Unknown clip name.")

        if name not in self._ppclips:
            if name not in self._clips:
                self._load_clips(self._clip_metadata[name].fspec)
            clip = self._clips[name]
            self._ppclips[name] = anim.PreprocessedClip.from_anim_clip(clip)

        ppclip = self._ppclips[name]
        self.play_anim_ppclip(ppclip)

    def cancel_anim(self) -> None:
        self.anim_controller.cancel_anim()

    def play_anim_group(self, anim_group_name: str) -> None:
        logger_animation.info("Playing animation group {}".format(anim_group_name))
        animation_group = self.animation_groups.get(anim_group_name)
        if not animation_group:
            logger_animation.error("Failed to find animation group {}.".format(anim_group_name))
            return
        member = animation_group.choose_member()
        logger_animation.info("Playing animation {}".format(member.name))
        self.play_anim(member.name)

    def load_anims(self) -> None:
        util.check_assets()
        anim_dir = str(util.get_cozmo_anim_dir())
        self._clip_metadata = anim_encoder.get_clip_metadata(anim_dir)
        self._clips = {}
        resource_dir = str(util.get_cozmo_asset_dir())
        self.animation_groups = anim.load_animation_groups(resource_dir)

    def get_anim_names(self) -> set:
        return set(self._clip_metadata.keys())

    @property
    def anim_names(self) -> set:
        return self.get_anim_names()

    def set_volume(self, level: int) -> None:
        """ Set audio output volume to a level in the range 0-65535. """
        pkt = protocol_encoder.SetRobotVolume(level=level)
        self.conn.send(pkt)

    def play_audio(self, fspec) -> None:
        """Play a WAVE file or raw PCM/ulaw sample data through the speaker.

        ``fspec`` may be a filename, a file-like object, or raw 22 kHz,
        16-bit mono WAVE bytes.
        """
        if isinstance(fspec, (bytes, bytearray, memoryview)):
            import io as _io
            pkts = audio.load_wav(_io.BytesIO(bytes(fspec)))
        elif hasattr(fspec, "read"):
            pkts = audio.load_wav(fspec)
        else:
            pkts = audio.load_wav(fspec)
        self.anim_controller.play_audio(pkts)

    def play_audio_samples(self, samples) -> None:
        """Play already encoded 8-bit u-law samples."""
        from . import audio as _audio
        self.anim_controller.play_audio(_audio.samples_to_packets(samples))

    def say(self, text: str, voice: Optional[str] = None, rate: Optional[int] = None) -> None:
        """Speak ``text`` through the robot's speaker.

        Requires an optional TTS backend (espeak-ng, pyttsx3, macOS ``say`` or
        Windows SAPI).  See :mod:`pycozmo.tts` for details.
        """
        from . import tts
        data = tts.synthesize_wav(text, voice=voice, rate=rate)
        self.play_audio(data)

    def activate_behavior(self, behavior):
        self.add_child_dispatcher(behavior)
        behavior.activate()

    def deactivate_behavior(self, behavior):
        self.del_child_dispatcher(behavior)
        behavior.deactivate()

    def enable_animations(self, enabled: bool = True) -> None:
        self.anim_controller.enable_animations(enabled)

    def enable_procedural_face(self, enabled: bool = True) -> None:
        self.anim_controller.enable_procedural_face(enabled)
