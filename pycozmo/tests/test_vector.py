"""Tests for the Vector animation reader, converter and player."""

import json
import os
import random
import tempfile
import unittest

import flatbuffers

from pycozmo import anim_encoder
from pycozmo.vector import (KEYFRAME_TYPES, VectorAnimationError,
                            VectorAnimations, VectorAnimator,
                            VectorBackpackLights, VectorBehavior,
                            VectorBehaviorError, VectorBehaviorSystem,
                            VectorBodyMotion, VectorClip, VectorEvent,
                            VectorFaceAnimation, VectorFrame, VectorHeadAngle,
                            VectorLiftHeight, VectorProceduralFace,
                            VectorRobotAudio, VectorSpriteBox,
                            VectorTurnToRecordedHeading, cozmo_clips_to_vector,
                            is_vector_animation, load_and_play,
                            load_vector_animations, load_vector_behaviors)


# ---------------------------------------------------------------------------
# Synthetic FlatBuffers fixture
# ---------------------------------------------------------------------------

def scalar_vector(builder, values, prepend, elem_size):
    builder.StartVector(elem_size, len(values), elem_size)
    for value in reversed(values):
        prepend(value)
    return builder.EndVector()


def offset_vector(builder, offsets):
    builder.StartVector(4, len(offsets), 4)
    for offset in reversed(offsets):
        builder.PrependUOffsetTRelative(offset)
    return builder.EndVector()


def table(builder, fields, num_fields):
    builder.StartObject(num_fields)
    for slot in sorted(fields, reverse=True):
        kind, value, default = fields[slot]
        if kind == "offset":
            builder.PrependUOffsetTRelativeSlot(slot, value, default)
        elif kind == "u32":
            builder.PrependUint32Slot(slot, value, default)
        elif kind == "u8":
            builder.PrependUint8Slot(slot, value, default)
        elif kind == "i8":
            builder.PrependInt8Slot(slot, value, default)
        elif kind == "i16":
            builder.PrependInt16Slot(slot, value, default)
        elif kind == "u16":
            builder.PrependUint16Slot(slot, value, default)
        elif kind == "i32":
            builder.PrependInt32Slot(slot, value, default)
        elif kind == "f32":
            builder.PrependFloat32Slot(slot, value, default)
        elif kind == "bool":
            builder.PrependBoolSlot(slot, value, default)
        else:
            raise ValueError(kind)
    return builder.EndObject()


def build_vector_file():
    """Build a small but complete Vector ``AnimClips`` FlatBuffers file."""
    builder = flatbuffers.Builder(2048)

    name = builder.CreateString("vector_test")
    face_name = builder.CreateString("anim_face_hello")
    event_id = builder.CreateString("test_event")
    asset = builder.CreateString("sprite.png")
    sb_name = builder.CreateString("main")
    layer = builder.CreateString("Overlay")
    render = builder.CreateString("RGBA")
    end_type = builder.CreateString("loop")
    straight = builder.CreateString("STRAIGHT")

    front = scalar_vector(builder, [1.0, 0.25, 0.0], builder.PrependFloat32, 4)
    middle = scalar_vector(builder, [0.0, 0.5, 1.0], builder.PrependFloat32, 4)
    back = scalar_vector(builder, [0.0, 0.0, 0.1], builder.PrependFloat32, 4)

    event_ids = scalar_vector(builder, [42, 43], builder.PrependUint32, 4)
    volumes = scalar_vector(builder, [0.8, 0.6], builder.PrependFloat32, 4)
    probabilities = scalar_vector(builder, [0.7, 0.3], builder.PrependFloat32, 4)
    group = table(builder, {
        0: ("offset", event_ids, 0),
        1: ("offset", volumes, 0),
        2: ("offset", probabilities, 0),
    }, 3)
    groups = offset_vector(builder, [group])

    state = table(builder, {0: ("u32", 1, 0), 1: ("u32", 2, 0)}, 2)
    states = offset_vector(builder, [state])
    switch = table(builder, {0: ("u32", 3, 0), 1: ("u32", 4, 0)}, 2)
    switches = offset_vector(builder, [switch])
    parameter = table(builder, {
        0: ("u32", 5, 0), 1: ("f32", 0.5, 0.0),
        2: ("u32", 50, 0), 3: ("u8", 1, 0),
    }, 4)
    parameters = offset_vector(builder, [parameter])

    left_eye = scalar_vector(builder, [0.1] * 19, builder.PrependFloat32, 4)
    right_eye = scalar_vector(builder, [0.2] * 19, builder.PrependFloat32, 4)

    lift0 = table(builder, {
        0: ("u32", 0, 0), 1: ("u32", 500, 0),
        2: ("u8", 50, 0), 3: ("u8", 5, 0),
    }, 4)
    lift1 = table(builder, {
        0: ("u32", 700, 0), 1: ("u32", 100, 0),
        2: ("u8", 100, 0), 3: ("u8", 0, 0),
    }, 4)
    head = table(builder, {
        0: ("u32", 0, 0), 1: ("u32", 500, 0),
        2: ("i8", -10, 0), 3: ("u8", 2, 0),
    }, 4)
    lights = table(builder, {
        0: ("u32", 0, 0), 1: ("u32", 500, 0),
        2: ("offset", front, 0), 3: ("offset", middle, 0),
        4: ("offset", back, 0),
    }, 5)
    face_anim = table(builder, {
        0: ("u32", 0, 0), 1: ("offset", face_name, 0),
        2: ("f32", 0.75, 1.0),
    }, 3)
    event = table(builder, {0: ("u32", 150, 0), 1: ("offset", event_id, 0)}, 2)
    body = table(builder, {
        0: ("u32", 250, 0), 1: ("u32", 1000, 0),
        2: ("offset", straight, 0), 3: ("i16", 100, 0),
    }, 4)
    record = table(builder, {0: ("u32", 300, 0)}, 1)
    turn = table(builder, {
        0: ("u32", 400, 0), 1: ("u32", 600, 0),
        2: ("i16", 90, 0), 3: ("i16", 45, 0),
        4: ("i16", 800, 1000), 5: ("i16", 700, 1000),
        6: ("u16", 3, 2), 7: ("u16", 1, 0), 8: ("bool", True, False),
    }, 9)
    procedural = table(builder, {
        0: ("u32", 0, 0), 1: ("f32", 0.0, 0.0),
        2: ("f32", 0.0, 0.0), 3: ("f32", 0.0, 0.0),
        4: ("f32", 1.0, 1.0), 5: ("f32", 1.0, 1.0),
        6: ("offset", left_eye, 0), 7: ("offset", right_eye, 0),
        8: ("f32", 0.25, 1.0),
    }, 9)
    audio = table(builder, {
        0: ("u32", 100, 0), 1: ("offset", groups, 0),
        2: ("offset", states, 0), 3: ("offset", switches, 0),
        4: ("offset", parameters, 0),
    }, 5)
    sprite = table(builder, {
        0: ("u32", 120, 0), 1: ("offset", sb_name, 0),
        2: ("offset", layer, 0), 3: ("offset", asset, 0),
        4: ("offset", render, 0), 5: ("offset", end_type, 0),
        6: ("f32", 90.0, 100.0), 7: ("i32", 10, 0),
        8: ("i32", 20, 0), 9: ("u32", 30, 0), 10: ("u32", 40, 0),
    }, 11)

    tracks = {
        0: [lift0, lift1], 1: [procedural], 2: [head], 3: [audio],
        4: [lights], 5: [face_anim], 6: [event], 7: [body],
        8: [record], 9: [turn], 10: [sprite],
    }
    fields = {}
    for slot, items in tracks.items():
        fields[slot] = ("offset", offset_vector(builder, items), 0)
    keyframes = table(builder, fields, 11)
    clip = table(builder, {0: ("offset", name, 0), 1: ("offset", keyframes, 0)}, 2)
    clips = offset_vector(builder, [clip])
    root = table(builder, {0: ("offset", clips, 0)}, 1)
    builder.Finish(root)
    return bytes(builder.Output())


class FakeClient:

    def __init__(self):
        self.calls = []

    def set_lift_height(self, height):
        self.calls.append(("lift", height))

    def set_head_angle(self, angle):
        self.calls.append(("head", angle))

    def set_backpack_lights(self, front, middle, back):
        self.calls.append(("lights", front, middle, back))

    def play_face_animation(self, name):
        self.calls.append(("face", name))

    def play_audio_event(self, event_id, volume=1.0):
        self.calls.append(("audio", event_id, volume))

    def trigger_event(self, event_id):
        self.calls.append(("event", event_id))

    def drive_straight(self, speed, duration=0):
        self.calls.append(("drive", speed, duration))


class TestVectorParsing(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data = build_vector_file()

    def test_is_vector_animation(self):
        self.assertTrue(is_vector_animation(self.data))
        self.assertFalse(is_vector_animation(b""))
        self.assertFalse(is_vector_animation(b"not an animation"))

    def test_parses_all_tracks(self):
        animations = VectorAnimations.from_fb_bytes(self.data)
        self.assertEqual(animations.names, ["vector_test"])
        self.assertIn("vector_test", animations)
        self.assertEqual(animations["vector_test"].name, "vector_test")
        self.assertEqual(animations[0].name, "vector_test")
        self.assertEqual(len(animations), 1)
        clip = animations["vector_test"]
        self.assertEqual(clip.duration_ms, 1250)
        self.assertAlmostEqual(clip.duration, 1.25)
        self.assertEqual(len(clip), 12)
        self.assertEqual(len(clip.tracks), len(KEYFRAME_TYPES))
        self.assertEqual(len(clip.get("LiftHeightKeyFrame")), 2)
        self.assertEqual(clip.keyframes_at(0)[0].trigger_time_ms, 0)
        self.assertEqual(clip.keyframes_at(0, tolerance=5)[0].trigger_time_ms, 0)
        self.assertIn("VectorClip(name='vector_test'", repr(clip))

    def test_lift_and_head_values(self):
        clip = VectorAnimations.from_fb_bytes(self.data)["vector_test"]
        lift = clip.get("LiftHeightKeyFrame")[0]
        self.assertIsInstance(lift, VectorLiftHeight)
        self.assertEqual((lift.height_mm, lift.height_variability_mm), (50, 5))
        self.assertEqual(lift.duration_ms, 500)
        head = clip.get("HeadAngleKeyFrame")[0]
        self.assertIsInstance(head, VectorHeadAngle)
        self.assertEqual(head.angle_deg, -10)
        self.assertAlmostEqual(head.angle_rad, -10 * 3.141592653589793 / 180)

    def test_body_turn_and_event_values(self):
        clip = VectorAnimations.from_fb_bytes(self.data)["vector_test"]
        body = clip.get("BodyMotionKeyFrame")[0]
        self.assertIsInstance(body, VectorBodyMotion)
        self.assertTrue(body.is_straight)
        self.assertFalse(body.is_turn_in_place)
        self.assertEqual((body.radius_mm, body.speed), ("STRAIGHT", 100))
        turn = clip.get("TurnToRecordedHeadingKeyFrame")[0]
        self.assertIsInstance(turn, VectorTurnToRecordedHeading)
        self.assertEqual(turn.offset_deg, 90)
        self.assertEqual(turn.accel_deg_per_sec2, 800)
        self.assertEqual(turn.decel_deg_per_sec2, 700)
        self.assertEqual(turn.tolerance_deg, 3)
        self.assertEqual(turn.num_half_revs, 1)
        self.assertTrue(turn.use_shortest_dir)
        event = clip.get("EventKeyFrame")[0]
        self.assertIsInstance(event, VectorEvent)
        self.assertEqual(event.event_id, "test_event")

    def test_audio_values(self):
        clip = VectorAnimations.from_fb_bytes(self.data)["vector_test"]
        audio = clip.get("RobotAudioKeyFrame")[0]
        self.assertIsInstance(audio, VectorRobotAudio)
        self.assertEqual(len(audio.event_groups), 1)
        group = audio.event_groups[0]
        self.assertEqual(group.event_ids, [42, 43])
        self.assertAlmostEqual(group.volumes[0], 0.8, places=6)
        self.assertAlmostEqual(group.volumes[1], 0.6, places=6)
        self.assertAlmostEqual(group.probabilities[0], 0.7, places=6)
        self.assertAlmostEqual(group.probabilities[1], 0.3, places=6)
        selected = group.choose(random.Random(0))
        self.assertIn(selected[0], (42, 43))
        self.assertTrue(min(abs(selected[1] - 0.8), abs(selected[1] - 0.6)) < 1e-6)
        self.assertIn(audio.choose_event(random.Random(0))[0], (42, 43))
        self.assertEqual(audio.states[0].state_group_id, 1)
        self.assertEqual(audio.switches[0].switch_group_id, 3)
        self.assertEqual(audio.parameters[0].parameter_id, 5)
        self.assertIsNone(VectorRobotAudio().choose_event())

    def test_face_lights_and_sprite_values(self):
        clip = VectorAnimations.from_fb_bytes(self.data)["vector_test"]
        face = clip.get("FaceAnimationKeyFrame")[0]
        self.assertIsInstance(face, VectorFaceAnimation)
        self.assertEqual(face.anim_name, "anim_face_hello")
        self.assertEqual(face.scanline_opacity, 0.75)
        procedural = clip.get("ProceduralFaceKeyFrame")[0]
        self.assertIsInstance(procedural, VectorProceduralFace)
        self.assertEqual(len(procedural.left_eye), 19)
        self.assertEqual(len(procedural.face_params), 43)
        lights = clip.get("BackpackLightsKeyFrame")[0]
        self.assertIsInstance(lights, VectorBackpackLights)
        self.assertEqual(lights.front_rgb, (255, 64, 0))
        self.assertEqual(lights.middle_rgb, (0, 128, 255))
        self.assertEqual(lights.back_rgb, (0, 0, 26))
        sprite = clip.get("SpriteBoxKeyFrame")[0]
        self.assertIsInstance(sprite, VectorSpriteBox)
        self.assertEqual(sprite.bbox, (10, 20, 40, 60))
        self.assertFalse(sprite.is_clear)
        self.assertTrue(VectorSpriteBox().is_clear)

    def test_json_roundtrip(self):
        animations = VectorAnimations.from_fb_bytes(self.data)
        data = animations.to_dict()
        restored = VectorAnimations.from_dict(data)
        self.assertEqual(restored.to_dict(), data)
        self.assertEqual(len(restored["vector_test"]), len(animations["vector_test"]))
        directory = tempfile.mkdtemp()
        filename = os.path.join(directory, "vector_test.json")
        try:
            animations.to_json_file(filename)
            from_file = VectorAnimations.from_json_file(filename)
            self.assertEqual(from_file.to_dict(), data)
            from_source = load_vector_animations(filename)
            self.assertEqual(from_source.to_dict(), data)
            json_bytes = json.dumps(data).encode("utf-8")
            self.assertEqual(load_vector_animations(json_bytes).to_dict(), data)
        finally:
            if os.path.exists(filename):
                os.unlink(filename)
            os.rmdir(directory)

    def test_load_from_binary_file(self):
        directory = tempfile.mkdtemp()
        filename = os.path.join(directory, "vector_test.bin")
        try:
            with open(filename, "wb") as stream:
                stream.write(self.data)
            animations = load_vector_animations(filename)
            self.assertEqual(animations.names, ["vector_test"])
            self.assertTrue(is_vector_animation(filename))
        finally:
            if os.path.exists(filename):
                os.unlink(filename)
            os.rmdir(directory)


class TestVectorConversion(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.clip = VectorAnimations.from_fb_bytes(build_vector_file())["vector_test"]

    def test_to_cozmo_clip(self):
        clip = self.clip.to_cozmo_clip()
        self.assertIsInstance(clip, anim_encoder.AnimClip)
        self.assertEqual(clip.name, "vector_test")
        kinds = {type(keyframe) for keyframe in clip.keyframes}
        self.assertIn(anim_encoder.AnimTurnToRecordedHeading, kinds)
        self.assertIn(anim_encoder.AnimBodyMotion, kinds)
        self.assertIn(anim_encoder.AnimRobotAudio, kinds)
        self.assertIn(anim_encoder.AnimProceduralFace, kinds)
        # Sprite boxes have no Cozmo equivalent, so the clip has one keyframe
        # fewer than the Vector source.
        self.assertEqual(len(clip.keyframes), len(self.clip.keyframes) - 1)
        turn = next(keyframe for keyframe in clip.keyframes
                    if isinstance(keyframe, anim_encoder.AnimTurnToRecordedHeading))
        self.assertEqual(turn.accel_deg_per_sec_2, 800)
        self.assertEqual(turn.decel_deg_per_sec_2, 700)
        self.assertEqual(turn.offset_deg, 90)
        body = next(keyframe for keyframe in clip.keyframes
                    if isinstance(keyframe, anim_encoder.AnimBodyMotion))
        self.assertEqual(body.radius_mm, "STRAIGHT")
        audio = next(keyframe for keyframe in clip.keyframes
                     if isinstance(keyframe, anim_encoder.AnimRobotAudio))
        self.assertEqual(audio.audio_event_ids, [42, 43])
        self.assertAlmostEqual(audio.volume, 0.8)
        lights = next(keyframe for keyframe in clip.keyframes
                      if isinstance(keyframe, anim_encoder.AnimBackpackLights))
        self.assertEqual((lights.left.red, lights.left.green, lights.left.blue),
                         (255, 64, 0))
        self.assertEqual((lights.right.red, lights.right.green, lights.right.blue),
                         (0, 0, 26))

    def test_animations_to_cozmo_clips(self):
        animations = VectorAnimations.from_fb_bytes(build_vector_file())
        clips = animations.to_cozmo_clips()
        self.assertIsInstance(clips, anim_encoder.AnimClips)
        self.assertEqual(len(clips.clips), 1)
        self.assertEqual(clips.clips[0].name, "vector_test")

    def test_cozmo_roundtrip(self):
        source = self.clip.to_cozmo_clip()
        animations = cozmo_clips_to_vector(anim_encoder.AnimClips([source]))
        self.assertEqual(animations.names, ["vector_test"])
        clip = animations["vector_test"]
        turn = clip.get("TurnToRecordedHeadingKeyFrame")[0]
        self.assertEqual(turn.accel_deg_per_sec2, 800)
        self.assertEqual(turn.decel_deg_per_sec2, 700)
        self.assertEqual(turn.offset_deg, 90)
        body = clip.get("BodyMotionKeyFrame")[0]
        self.assertEqual(body.radius_mm, "STRAIGHT")
        lights = clip.get("BackpackLightsKeyFrame")[0]
        self.assertEqual(lights.front_rgb, (255, 64, 0))
        self.assertEqual(lights.back_rgb, (0, 0, 26))
        self.assertEqual(len(clip.get("LiftHeightKeyFrame")), 2)
        self.assertIn("EventKeyFrame", clip.to_dict()["keyframes"])

    def test_cozmo_clip_accepts_plain_list(self):
        source = self.clip.to_cozmo_clip()
        animations = cozmo_clips_to_vector([source])
        self.assertEqual(animations.names, ["vector_test"])


class TestVectorFrames(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.clip = VectorAnimations.from_fb_bytes(build_vector_file())["vector_test"]

    def test_iter_frames(self):
        frames = list(VectorAnimator(self.clip).iter_frames())
        self.assertTrue(all(isinstance(frame, VectorFrame) for frame in frames))
        self.assertEqual([frame.time_ms for frame in frames],
                         [0, 100, 120, 150, 250, 300, 400, 700])
        first = frames[0]
        self.assertEqual(first.lift_height_mm, 50)
        self.assertEqual(first.head_angle_deg, -10)
        self.assertIsNotNone(first.backpack_lights)
        self.assertIsNotNone(first.face_animation)
        self.assertIsNotNone(first.procedural_face)
        self.assertEqual(first.events, [])
        audio_frame = frames[1]
        self.assertEqual(len(audio_frame.audio), 1)
        self.assertEqual(frames[2].sprite_boxes[0].sprite_box_name, "main")
        self.assertEqual(frames[3].events, ["test_event"])
        self.assertEqual(frames[4].body_motion.radius_mm, "STRAIGHT")
        self.assertIsNotNone(frames[5].keyframes[0])
        self.assertEqual(frames[7].lift_height_mm, 100)
        payload = first.to_dict()
        self.assertEqual(payload["time_ms"], 0)
        self.assertEqual(payload["face_animation"], "anim_face_hello")
        self.assertEqual(payload["lift_height_mm"], 50)


class TestVectorPlayback(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.clip = VectorAnimations.from_fb_bytes(build_vector_file())["vector_test"]

    def test_play_dispatches_to_client(self):
        client = FakeClient()
        sleeps = []
        VectorAnimator(self.clip).play(client=client, sleep=sleeps.append)
        self.assertEqual(sleeps[0], 0.1)
        self.assertIn(("lift", 50), client.calls)
        self.assertIn(("lift", 100), client.calls)
        self.assertIn(("face", "anim_face_hello"), client.calls)
        self.assertIn(("event", "test_event"), client.calls)
        self.assertIn(("drive", 100, 1.0), client.calls)
        self.assertTrue(any(call[0] == "head" for call in client.calls))
        self.assertTrue(any(call[0] == "lights" for call in client.calls))
        self.assertTrue(any(call[0] == "audio" for call in client.calls))

    def test_play_handlers_override_client(self):
        calls = []
        handlers = {
            "keyframe": lambda keyframe: calls.append(("keyframe", type(keyframe))),
            "lift": lambda keyframe: calls.append(("lift", keyframe.height_mm)),
            "head": lambda keyframe: calls.append(("head", keyframe.angle_deg)),
            "body": lambda keyframe: calls.append(("body", keyframe.radius_mm)),
            "lights": lambda keyframe: calls.append(("lights", keyframe.front_rgb)),
            "face": lambda keyframe: calls.append(("face", keyframe.anim_name)),
            "procedural_face": lambda keyframe: calls.append(("procedural", len(keyframe.left_eye))),
            "sprite": lambda keyframe: calls.append(("sprite", keyframe.sprite_box_name)),
            "audio": lambda keyframe: calls.append(("audio", len(keyframe.event_groups))),
            "event": lambda event_id: calls.append(("event", event_id)),
        }
        VectorAnimator(self.clip).play(handlers=handlers, sleep=lambda _: None)
        self.assertIn(("lift", 50), calls)
        self.assertIn(("lift", 100), calls)
        self.assertIn(("face", "anim_face_hello"), calls)
        self.assertIn(("sprite", "main"), calls)
        self.assertIn(("procedural", 19), calls)
        self.assertIn(("event", "test_event"), calls)
        self.assertIn(("body", "STRAIGHT"), calls)
        self.assertTrue(any(call[0] == "keyframe" for call in calls))

    def test_speed_scales_sleep(self):
        sleeps = []
        VectorAnimator(self.clip).play(speed=2.0, sleep=sleeps.append)
        self.assertAlmostEqual(sleeps[0], 0.05)
        with self.assertRaises(ValueError):
            VectorAnimator(self.clip).play(speed=0)

    def test_stop_event(self):
        import threading
        stop = threading.Event()
        stop.set()
        sleeps = []
        VectorAnimator(self.clip).play(stop_event=stop, sleep=sleeps.append)
        self.assertEqual(sleeps, [])

    def test_play_async(self):
        import threading
        client = FakeClient()
        animator = VectorAnimator(self.clip)
        thread = animator.play_async(client=client, speed=100.0)
        self.assertIsInstance(thread, threading.Thread)
        thread.join(timeout=10)
        self.assertFalse(thread.is_alive())
        self.assertIn(("lift", 50), client.calls)

    def test_load_and_play(self):
        directory = tempfile.mkdtemp()
        filename = os.path.join(directory, "vector_test.bin")
        calls = []
        try:
            with open(filename, "wb") as stream:
                stream.write(build_vector_file())
            load_and_play(filename, clip_name="vector_test",
                          handlers={"event": calls.append}, sleep=lambda _: None)
            self.assertEqual(calls, ["test_event"])
            with self.assertRaises(KeyError):
                load_and_play(filename, clip_name="missing", sleep=lambda _: None)
        finally:
            if os.path.exists(filename):
                os.unlink(filename)
            os.rmdir(directory)


class TestVectorBehaviors(unittest.TestCase):

    def setUp(self):
        self.animations = VectorAnimations.from_fb_bytes(build_vector_file())
        self.clip = VectorClip("beep", [VectorEvent(0, "beep")])

    def test_behavior_play_and_done(self):
        events = []
        done = []
        obj = VectorBehavior("Beep", self.clip, handlers={"event": events.append},
                             on_done=done.append)
        self.assertFalse(obj.running)
        self.assertTrue(obj.available())
        thread = obj.start(wait=True)
        self.assertFalse(thread.is_alive())
        self.assertEqual(events, ["beep"])
        self.assertEqual(done, [obj])
        self.assertEqual(obj.runs, 1)
        self.assertIsNotNone(obj.last_started)
        self.assertIsNotNone(obj.last_finished)
        self.assertIn("Beep", repr(obj))
        self.assertEqual(obj.to_dict()["clip"], "beep")

    def test_behavior_sync_play(self):
        events = []
        obj = VectorBehavior("Beep", self.clip, handlers={"event": events.append})
        obj.play()
        self.assertEqual(events, ["beep"])
        self.assertEqual(obj.runs, 1)

    def test_behavior_cooldown(self):
        obj = VectorBehavior("Beep", self.clip, cooldown=10.0)
        self.assertTrue(obj.available(now=0))
        obj.last_finished = 100.0
        self.assertFalse(obj.available(now=105))
        self.assertTrue(obj.available(now=110))
        obj.cooldown = 0
        self.assertTrue(obj.available(now=100))

    def test_behavior_loop_and_stop(self):
        import threading
        calls = []
        enough = threading.Event()

        def on_event(name):
            calls.append(name)
            if len(calls) >= 3:
                enough.set()

        obj = VectorBehavior("Loop", self.clip, handlers={"event": on_event},
                             loop=True)
        thread = obj.start()
        self.assertTrue(enough.wait(5))
        obj.stop()
        thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertFalse(obj.running)
        self.assertGreaterEqual(len(calls), 3)
        self.assertEqual(obj.runs, 1)

    def test_behavior_errors(self):
        with self.assertRaises(VectorBehaviorError):
            VectorBehavior("Bad", "not a clip")
        with self.assertRaises(VectorBehaviorError):
            VectorBehavior("Bad", self.clip, speed=0)
        obj = VectorBehavior("Beep", self.clip)
        thread = obj.start()
        thread.join(5)
        self.assertFalse(obj.running)

    def test_behavior_system(self):
        system = VectorBehaviorSystem(self.animations)
        low = system.add_clip("low", self.clip, priority=1)
        high = system.add_clip("high", self.clip, priority=5)
        self.assertEqual(len(system), 2)
        self.assertIn("low", system)
        self.assertEqual([behavior.name for behavior in system.available()],
                         ["high", "low"])
        self.assertIs(system.choose(), high)
        high.last_finished = 100.0
        high.cooldown = 10.0
        self.assertIs(system.choose(now=105), low)
        self.assertIs(system.choose(now=110), high)
        self.assertIs(system.get("missing"), None)
        self.assertTrue(system.remove("low"))
        self.assertFalse(system.remove("low"))

    def test_behavior_system_play_and_dict(self):
        system = VectorBehaviorSystem(self.animations)
        behavior = system.from_dict({
            "behaviorID": "Test", "clipName": "vector_test",
            "priority": 2, "cooldown": 3, "speed": 100.0,
        })
        self.assertEqual(behavior.name, "Test")
        self.assertEqual(behavior.priority, 2)
        self.assertEqual(behavior.cooldown, 3)
        events = []
        played = system.play("Test", handlers={"event": events.append})
        self.assertIs(played, behavior)
        self.assertEqual(events, ["test_event"])
        self.assertIsNone(system.play("missing"))
        with self.assertRaises(VectorBehaviorError):
            system.from_dict("nope")
        with self.assertRaises(VectorBehaviorError):
            system.from_dict({"name": "x"})
        with self.assertRaises(VectorBehaviorError):
            system.add_clip("missing", "not_a_clip")
        with self.assertRaises(VectorBehaviorError):
            system.add("not a behavior")
        with self.assertRaises(VectorBehaviorError):
            system.add(behavior, replace=False)

    def test_load_vector_behaviors(self):
        system = load_vector_behaviors(self.animations)
        self.assertEqual(len(system), 1)
        self.assertIn("vector_test", system)
        self.assertEqual(system.get("vector_test").clip.name, "vector_test")
        self.assertIs(load_vector_behaviors(system), system)

        directory = tempfile.mkdtemp()
        filename = os.path.join(directory, "behaviors.json")
        try:
            with open(filename, "w", encoding="utf-8") as stream:
                json.dump([{"name": "Json", "clip": "vector_test",
                            "priority": 4}], stream)
            loaded = load_vector_behaviors(filename, animations=self.animations)
            self.assertEqual(list(loaded.behaviors), ["Json"])
            self.assertEqual(loaded.get("Json").priority, 4)
        finally:
            if os.path.exists(filename):
                os.unlink(filename)
            os.rmdir(directory)


class TestVectorErrors(unittest.TestCase):

    def test_invalid_sources(self):
        with self.assertRaises(VectorAnimationError):
            VectorAnimations.from_fb_bytes(b"")
        with self.assertRaises(VectorAnimationError):
            VectorAnimations.from_fb_bytes(b"\x00" * 32)
        with self.assertRaises(TypeError):
            load_vector_animations(42)
        self.assertTrue(issubclass(VectorAnimationError, ValueError))

    def test_empty_animation_file(self):
        builder = flatbuffers.Builder(64)
        clips = offset_vector(builder, [])
        root = table(builder, {0: ("offset", clips, 0)}, 1)
        builder.Finish(root)
        with self.assertRaises(VectorAnimationError):
            VectorAnimations.from_fb_bytes(bytes(builder.Output()))

    def test_clip_from_dict_accepts_short_names(self):
        clip = VectorClip.from_dict({
            "Name": "short",
            "keyframes": {
                "LiftHeight": [{"triggerTime_ms": 10, "durationTime_ms": 20,
                                "height_mm": 30}],
                "Event": [{"triggerTime_ms": 40, "event_id": "evt"}],
            },
        })
        self.assertEqual(clip.name, "short")
        self.assertEqual(len(clip.get("LiftHeightKeyFrame")), 1)
        self.assertEqual(clip.get("EventKeyFrame")[0].event_id, "evt")
        self.assertEqual(clip.duration_ms, 40)


if __name__ == "__main__":
    unittest.main()
