
import json
import os
import random
import tempfile
import unittest

from pycozmo.personality import (DEFAULT_DRIVES, DRIVE_NAMES, EMOTION_NAMES,
                                 TRAIT_NAMES, Drive, Personality)


class TestDrive(unittest.TestCase):

    def test_growth_and_satisfaction(self):
        drive = Drive("play", value=0.5, growth_rate=0.1)
        self.assertAlmostEqual(drive.update(2.0), 0.7)
        self.assertAlmostEqual(drive.satisfy(0.2), 0.5)
        self.assertAlmostEqual(drive.frustrate(0.25), 0.75)
        # Values are clamped to the configured range.
        drive.update(100.0)
        self.assertEqual(drive.value, 1.0)
        drive.satisfy(100.0)
        self.assertEqual(drive.value, 0.0)

    def test_roundtrip(self):
        drive = Drive.from_dict("rest", {"value": 0.4, "growth_rate": 0.02})
        self.assertEqual(drive.name, "rest")
        self.assertEqual(drive.to_dict(), {"value": 0.4, "growth_rate": 0.02})
        self.assertEqual(Drive.from_dict("rest", 0.7).value, 0.7)


class TestPersonality(unittest.TestCase):

    def test_defaults(self):
        personality = Personality()
        self.assertEqual(set(personality.traits), set(TRAIT_NAMES))
        self.assertEqual(set(personality.emotions), set(EMOTION_NAMES))
        self.assertEqual(set(personality.drives), set(DRIVE_NAMES))
        for name, (value, growth) in DEFAULT_DRIVES.items():
            self.assertAlmostEqual(personality.drives[name].value, value)
            self.assertAlmostEqual(personality.drives[name].growth_rate, growth)

    def test_traits_are_clamped(self):
        personality = Personality(traits={"playfulness": 5.0,
                                          "sociability": -2.0,
                                          "custom": 0.4})
        self.assertEqual(personality.traits["playfulness"], 1.0)
        self.assertEqual(personality.traits["sociability"], 0.0)
        self.assertEqual(personality.traits["custom"], 0.4)

    def test_update_grows_drives_and_decays_emotions(self):
        personality = Personality(emotions={"happy": 1.0, "frustrated": 0.8},
                                  baselines={"happy": 0.5, "frustrated": 0.0},
                                  emotion_decay=0.5)
        personality.update(2.0)
        # 50% of the distance is removed per second, clamped at 100%.
        self.assertAlmostEqual(personality.emotions["happy"], 0.5)
        self.assertAlmostEqual(personality.emotions["frustrated"], 0.0)
        self.assertGreater(personality.drives["play"].value, 0.35)

    def test_negative_dt_is_ignored(self):
        personality = Personality()
        before = personality.drives["play"].value
        personality.update(-1.0)
        self.assertEqual(personality.drives["play"].value, before)

    def test_feel_and_satisfy(self):
        personality = Personality()
        self.assertAlmostEqual(personality.feel("happy", 0.1), 0.65)
        self.assertAlmostEqual(personality.feel("happy", -5.0), 0.0)
        self.assertAlmostEqual(personality.satisfy("play", 0.2), 0.15)
        # Unknown drives are created on demand.
        self.assertEqual(personality.satisfy("new_drive", 0.0), 0.0)

    def test_apply_affectors(self):
        personality = Personality()
        before_play = personality.drives["play"].value
        personality.apply_affectors({"play": 0.2, "happy": -0.5})
        self.assertGreater(personality.drives["play"].value, before_play)
        self.assertAlmostEqual(personality.emotions["happy"], 0.05)

    def test_react_event_names(self):
        personality = Personality()
        anxious = personality.emotions["anxious"]
        safety = personality.drives["safety"].value
        self.assertTrue(personality.react("EvtRobotPickedUp"))
        self.assertGreater(personality.emotions["anxious"], anxious)
        self.assertGreater(personality.drives["safety"].value, safety)
        self.assertTrue(personality.react("FaceInitialDetectionChange"))
        self.assertFalse(personality.react("NoSuchEvent"))

    def test_custom_event_response(self):
        personality = Personality(event_responses={
            "MyEvent": {"emotions": {"excited": 0.2},
                        "drives": {"curiosity": 0.3}}})
        curiosity = personality.drives["curiosity"].value
        self.assertTrue(personality.react("MyEvent", strength=0.5))
        self.assertAlmostEqual(personality.emotions["excited"], 0.35)
        self.assertGreater(personality.drives["curiosity"].value, curiosity)

    def test_mood_and_dominance(self):
        personality = Personality(emotions={"happy": 1.0, "tired": 0.0},
                                  drives={"play": 1.0})
        self.assertGreater(personality.mood, 0.5)
        self.assertEqual(personality.dominant_drive[0], "play")
        self.assertEqual(personality.dominant_emotion[0], "happy")

    def test_score_activity(self):
        personality = Personality(traits={"playfulness": 1.0, "sociability": 0.0},
                                  drives={"play": 1.0, "social": 0.0})
        freeplay = personality.score_activity("Freeplay", {"cube_visible": True})
        social = personality.score_activity("Socialize", {"face_visible": True})
        self.assertGreater(freeplay, social)
        # Preferences are additive and names are normalized.
        personality.activity_preferences["Freeplay"] = 10.0
        self.assertGreater(personality.score_activity("Freeplay"),
                           personality.score_activity("Socialize"))

    def test_score_behavior(self):
        personality = Personality(traits={"playfulness": 1.0},
                                  drives={"play": 1.0, "rest": 0.0})
        self.assertGreater(personality.score_behavior("PlayAnim"),
                           personality.score_behavior("Sleep"))
        self.assertGreater(personality.score_behavior("Dance"), 0.0)
        personality.behavior_preferences["dance"] = 5.0
        self.assertGreater(personality.score_behavior("Dance"),
                           personality.score_behavior("PlayAnim"))

    def test_choose_activity(self):
        personality = Personality(traits={"playfulness": 1.0},
                                  drives={"play": 1.0})
        activities = ["Socialize", "Freeplay", "Sleep"]
        choice = personality.choose_activity(
            activities, rng=random.Random(0), temperature=0.0)
        self.assertEqual(choice, "Freeplay")
        # Empty input returns None.
        self.assertIsNone(personality.choose_activity([]))
        # A seeded RNG makes weighted choices reproducible.
        first = Personality().choose_activity(activities, rng=random.Random(42))
        second = Personality().choose_activity(activities, rng=random.Random(42))
        self.assertEqual(first, second)

    def test_choose_behavior_objects(self):
        class Behavior:
            def get_id(self):
                return "Dance"

        personality = Personality(drives={"play": 1.0})
        behavior = Behavior()
        self.assertIs(personality.choose_behavior([behavior], temperature=0.0),
                      behavior)

    def test_save_and_load(self):
        personality = Personality.playful()
        personality.feel("excited", 0.1)
        filename = os.path.join(tempfile.mkdtemp(), "personality.json")
        personality.save(filename)
        with open(filename, "r", encoding="utf-8") as stream:
            data = json.load(stream)
        self.assertIn("traits", data)
        loaded = Personality.load(filename)
        self.assertEqual(loaded.to_dict(), personality.to_dict())
        self.assertIsInstance(loaded, Personality)
        os.unlink(filename)

    def test_from_dict_and_presets(self):
        data = Personality.shy().to_dict()
        restored = Personality.from_dict(data)
        self.assertEqual(restored.traits["sociability"], 0.25)
        self.assertEqual(restored.baselines["anxious"], 0.2)
        for name in ("playful", "social", "curious", "shy", "calm", "brave", "tired"):
            preset = getattr(Personality, name)()
            self.assertIsInstance(preset, Personality)
        self.assertEqual(Personality.default().to_dict(), Personality().to_dict())


if __name__ == "__main__":
    unittest.main()
