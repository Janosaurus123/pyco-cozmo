"""Tests for behaviors, activity choosers and resource loaders."""

import json
import os
import random
import tempfile
import unittest

from pycozmo import activity, behavior, event


class FakeConnection:

    def __init__(self):
        self.events = []

    def post_event(self, evt, cli):
        self.events.append((evt, cli))


class FakeBehaviorClient:

    def __init__(self):
        self.conn = FakeConnection()
        self.animations = []
        self.cancelled = 0
        self.stopped = 0
        self.contacts = []
        self.straight = []
        self.wheels = []
        self.on_contact = None

    def play_anim_group(self, name):
        self.animations.append(name)

    def cancel_anim(self):
        self.cancelled += 1

    def stop_all_motors(self):
        self.stopped += 1

    def drive_off_charger_contacts(self, speed, duration=0):
        self.contacts.append((speed, duration))
        if self.on_contact is not None:
            self.on_contact()

    def drive_straight(self, speed, duration=0):
        self.straight.append((speed, duration))

    def drive_wheels(self, left, right, duration=0):
        self.wheels.append((left, right, duration))


class TestBehaviorPlayAnim(unittest.TestCase):

    def test_sequence(self):
        cli = FakeBehaviorClient()
        obj = behavior.BehaviorPlayAnim(cli, {"behaviorID": "B", "animTriggers": ["A", "B"]})
        self.assertEqual(obj.get_id(), "B")
        obj.activate()
        self.assertEqual(cli.animations, ["A"])
        obj.dispatch(event.EvtAnimationCompleted, cli)
        self.assertEqual(cli.animations, ["A", "B"])
        obj.dispatch(event.EvtAnimationCompleted, cli)
        self.assertEqual(len(cli.conn.events), 1)
        self.assertIs(cli.conn.events[0][0], event.EvtBehaviorDone)
        self.assertFalse(obj._active)
        # Late animation events are ignored after completion.
        obj.dispatch(event.EvtAnimationCompleted, cli)
        self.assertEqual(cli.animations, ["A", "B"])

    def test_single_trigger_and_empty_trigger(self):
        cli = FakeBehaviorClient()
        obj = behavior.BehaviorPlayAnim(cli, {"behaviorID": "B", "animTriggers": "Only"})
        self.assertEqual(obj.anim_triggers, ["Only"])
        obj.activate()
        self.assertEqual(cli.animations, ["Only"])
        empty = behavior.BehaviorPlayAnim(cli, {"behaviorID": "E"})
        empty.activate()
        self.assertEqual(len(cli.conn.events), 1)

    def test_deactivate_cancels(self):
        cli = FakeBehaviorClient()
        obj = behavior.BehaviorPlayAnim(cli, {"behaviorID": "B", "animTriggers": ["A", "B"]})
        obj.activate()
        obj.deactivate()
        obj.dispatch(event.EvtAnimationCompleted, cli)
        self.assertEqual(cli.animations, ["A"])
        self.assertEqual(cli.cancelled, 1)
        self.assertEqual(cli.conn.events, [])

    def test_activate_twice_restarts(self):
        cli = FakeBehaviorClient()
        obj = behavior.BehaviorPlayAnim(cli, {"behaviorID": "B", "animTriggers": ["A", "B"]})
        obj.activate()
        obj.dispatch(event.EvtAnimationCompleted, cli)
        obj.activate()
        self.assertEqual(cli.animations, ["A", "B", "A"])
        self.assertEqual(obj.current_trigger, 1)


class TestBehaviorArbitraryAnim(unittest.TestCase):

    def test_plays_requested_number(self):
        cli = FakeBehaviorClient()
        obj = behavior.BehaviorPlayArbitraryAnim(
            cli, {"behaviorID": "B", "animTriggers": ["A", "B", "C", "D"],
                  "numAnimations": 2}, rng=random.Random(3))
        obj.activate()
        self.assertEqual(len(obj.anim_triggers), 2)
        self.assertEqual(len(cli.animations), 1)
        obj.dispatch(event.EvtAnimationCompleted, cli)
        obj.dispatch(event.EvtAnimationCompleted, cli)
        self.assertEqual(len(cli.animations), 2)
        self.assertTrue(set(cli.animations) <= {"A", "B", "C", "D"})
        self.assertEqual(len(cli.conn.events), 1)

    def test_invalid_count_uses_random(self):
        cli = FakeBehaviorClient()
        obj = behavior.BehaviorPlayArbitraryAnim(
            cli, {"behaviorID": "B", "animTriggers": ["A", "B", "C"],
                  "numAnimations": "many"}, rng=random.Random(1))
        obj.activate()
        self.assertGreaterEqual(len(obj.anim_triggers), 1)
        self.assertLessEqual(len(obj.anim_triggers), 3)

    def test_count_is_clamped(self):
        cli = FakeBehaviorClient()
        obj = behavior.BehaviorPlayArbitraryAnim(
            cli, {"behaviorID": "B", "animTriggers": ["A", "B"], "numAnimations": 10},
            rng=random.Random(1))
        obj.activate()
        self.assertEqual(len(obj.anim_triggers), 2)

    def test_reactivation_samples_from_full_pool(self):
        class RotatingRng(random.Random):
            def __init__(self):
                super().__init__(1)
                self.pools = []

            def shuffle(self, seq):
                self.pools.append(list(seq))
                shift = len(self.pools) % len(seq)
                seq[:] = seq[shift:] + seq[:shift]

        cli = FakeBehaviorClient()
        rng = RotatingRng()
        obj = behavior.BehaviorPlayArbitraryAnim(
            cli, {"behaviorID": "B", "animTriggers": ["A", "B", "C", "D"],
                  "numAnimations": 1}, rng=rng)
        obj.activate()
        first = list(obj.anim_triggers)
        obj.activate()
        second = list(obj.anim_triggers)
        self.assertEqual(len(second), 1)
        self.assertEqual(obj._all_anim_triggers, ["A", "B", "C", "D"])
        self.assertEqual(rng.pools, [["A", "B", "C", "D"], ["A", "B", "C", "D"]])
        self.assertNotEqual(first, second)


class TestBehaviorCliffAndCharger(unittest.TestCase):

    def test_react_to_cliff(self):
        cli = FakeBehaviorClient()
        obj = behavior.BehaviorReactToCliff(cli, {"behaviorID": "Cliff"})
        obj.activate()
        self.assertEqual(cli.stopped, 1)
        self.assertEqual(cli.animations, ["ReactToCliff"])

    def test_drive_off_charger(self):
        cli = FakeBehaviorClient()
        obj = behavior.BehaviorDriveOffCharger(cli, {
            "behaviorID": "Charger",
            "driveSpeed_mmps": 50,
            "extraDistanceToDrive_mm": 100,
            "timeToDriveOffContacts_sec": 0.25,
            "wakeUpAnimTrigger": "Wake",
            "playWakeUpAnim": True,
        })
        obj.activate()
        self.assertEqual(cli.animations, ["Wake"])
        self.assertEqual(cli.contacts, [(-50.0, 0.25)])
        self.assertEqual(cli.straight, [(-50.0, 2.0)])
        self.assertEqual(len(cli.conn.events), 1)

    def test_drive_off_charger_fallback(self):
        class FallbackClient:
            def __init__(self):
                self.conn = FakeConnection()
                self.straight = []
                self.stopped = 0

            def drive_straight(self, speed, duration=0):
                self.straight.append((speed, duration))

            def stop_all_motors(self):
                self.stopped += 1

        cli = FallbackClient()
        obj = behavior.BehaviorDriveOffCharger(cli, {
            "behaviorID": "Charger", "extraDistanceToDrive_mm": 0,
            "timeToDriveOffContacts_sec": 1.0})
        obj.activate()
        self.assertEqual(cli.straight, [(-50.0, 1.0)])
        self.assertEqual(len(cli.conn.events), 1)

    def test_drive_off_charger_cancel(self):
        cli = FakeBehaviorClient()
        obj = behavior.BehaviorDriveOffCharger(cli, {
            "behaviorID": "Charger", "extraDistanceToDrive_mm": 100,
            "timeToDriveOffContacts_sec": 0.5})
        cli.on_contact = obj.deactivate
        obj.activate()
        self.assertEqual(cli.contacts, [(-50.0, 0.5)])
        self.assertEqual(cli.straight, [])
        self.assertEqual(cli.stopped, 1)
        self.assertEqual(cli.conn.events, [])

    def test_drive_off_charger_error_still_finishes(self):
        class ErrorClient(FakeBehaviorClient):
            def drive_off_charger_contacts(self, speed, duration=0):
                raise RuntimeError("motor fault")

        cli = ErrorClient()
        obj = behavior.BehaviorDriveOffCharger(cli, {"behaviorID": "Charger"})
        obj.activate()
        self.assertEqual(len(cli.conn.events), 1)


class TestBehaviorLookup(unittest.TestCase):

    def test_get_behavior_class(self):
        self.assertIs(behavior.get_behavior_class_from_dict(
            {"behaviorClass": "PlayAnim"}), behavior.BehaviorPlayAnim)
        self.assertIs(behavior.get_behavior_class_from_dict(
            {"behaviorClass": "playarbitraryanim"}),
            behavior.BehaviorPlayArbitraryAnim)
        self.assertIs(behavior.get_behavior_class_from_dict(
            {"behaviorClass": "some.module.DriveOffCharger"}),
            behavior.BehaviorDriveOffCharger)
        self.assertIs(behavior.get_behavior_class_from_dict(
            {"behaviorClass": "Unknown"}), behavior.Behavior)
        self.assertIs(behavior.get_behavior_class_from_dict({}), behavior.Behavior)

    def test_custom_behavior_class(self):
        class Custom(behavior.Behavior):
            pass

        behavior.BEHAVIOR_CLASSES["Custom"] = Custom
        try:
            self.assertIs(behavior.get_behavior_class_from_dict(
                {"behaviorClass": "Custom"}), Custom)
        finally:
            del behavior.BEHAVIOR_CLASSES["Custom"]

    def test_base_behavior_posts_done(self):
        cli = FakeBehaviorClient()
        obj = behavior.Behavior(cli, {"behaviorID": "Base"})
        obj.activate()
        self.assertEqual(len(cli.conn.events), 1)
        obj.deactivate()

    def test_load_behaviors_and_reaction_map(self):
        root = tempfile.mkdtemp()
        behaviors_dir = os.path.join(root, "cozmo_resources", "config", "engine",
                                     "behaviorSystem", "behaviors")
        os.makedirs(behaviors_dir)
        try:
            with open(os.path.join(behaviors_dir, "play.json"), "w") as stream:
                json.dump({"behaviorID": "Play", "behaviorClass": "PlayAnim",
                           "animTriggers": ["A"]}, stream)
            with open(os.path.join(behaviors_dir, "unknown.json"), "w") as stream:
                json.dump({"behaviorID": "Other", "behaviorClass": "Nope"}, stream)
            loaded = behavior.load_behaviors(root, FakeBehaviorClient())
            self.assertEqual(set(loaded), {"Play", "Other"})
            self.assertIsInstance(loaded["Play"], behavior.BehaviorPlayAnim)
            self.assertIs(type(loaded["Other"]), behavior.Behavior)

            map_file = os.path.join(root, "cozmo_resources", "config", "engine",
                                    "behaviorSystem", "reactionTrigger_behavior_map.json")
            with open(map_file, "w") as stream:
                json.dump({"reactionTriggerBehaviorMap": [{
                    "reactionTrigger": "CliffDetected",
                    "behaviorID": "Play",
                    "genericStrategyParams": {"shouldResumeLast": True},
                }]}, stream)
            triggers = behavior.load_reaction_trigger_behavior_map(root)
            self.assertEqual(set(triggers), {"CliffDetected"})
            trigger = triggers["CliffDetected"]
            self.assertIsInstance(trigger, behavior.ReactionTrigger)
            self.assertEqual(trigger.behavior_id, "Play")
            self.assertTrue(trigger.should_resume_last)
        finally:
            for dirpath, _, filenames in os.walk(root, topdown=False):
                for filename in filenames:
                    os.unlink(os.path.join(dirpath, filename))
                os.rmdir(dirpath)


def chooser_json(choice_type="Selection", names=("B1", "B2"), scoring=None):
    behaviors = []
    for name in names:
        item = {"behaviorID": name}
        if scoring and name in scoring:
            item["scoring"] = scoring[name]
        behaviors.append(item)
    return {"type": choice_type, "behaviors": behaviors}


def activity_json(activity_id, activity_type="BehaviorsOnly", choice_type="Selection",
                  names=("B1", "B2"), **extra):
    data = {
        "activityID": activity_id,
        "activityType": activity_type,
        "activityStrategy": {"type": choice_type},
        "behaviorChooser": chooser_json(choice_type, names),
    }
    data.update(extra)
    return data


class TestBehaviorChooser(unittest.TestCase):

    def test_selection(self):
        chooser = activity.BehaviorChooser("Selection", [
            {"behaviorID": "A"}, "B"], rng=random.Random(0))
        self.assertEqual(chooser.get_sorted_choices(), ["A", "B"])
        self.assertEqual(chooser.choose(), "A")
        self.assertEqual(chooser.choose({"B"}), "B")
        self.assertIsNone(chooser.choose(set()))
        self.assertEqual(chooser.iteration, 2)
        chooser.reset()
        self.assertEqual(chooser.iteration, 0)

    def test_strict_priority(self):
        chooser = activity.BehaviorChooser("StrictPriority", ["A", "B", "C"])
        self.assertEqual(chooser.get_sorted_choices(), ["A", "B", "C"])
        self.assertEqual(chooser.choose(), "A")
        self.assertEqual(chooser.get_sorted_choices(), ["B", "C"])
        self.assertEqual(chooser.choose(), "B")
        self.assertEqual(chooser.choose(), "C")
        self.assertIsNone(chooser.choose())
        self.assertIsNone(chooser.get_sorted_choices())
        chooser.reset()
        self.assertEqual(chooser.choose(), "A")

    def test_availability_callable(self):
        chooser = activity.BehaviorChooser("Selection", ["A", "B"])
        self.assertEqual(chooser.choose(lambda behavior_id: behavior_id == "B"), "B")
        self.assertEqual(chooser.choose(lambda: False), None)
        self.assertEqual(chooser.choose(lambda: True), "A")
        self.assertTrue(chooser.is_available("A", None))
        self.assertTrue(chooser.is_available("A", object()))

    def test_scoring_penalty(self):
        scoring = {
            "A": {"flatScore": 10, "repetitionPenalty": {
                "nodes": [{"x": 1, "y": 1}, {"x": 2, "y": 2}, {"x": 3, "y": 3}]}},
            "B": {"flatScore": 5},
        }
        chooser = activity.BehaviorChooser("Scoring", [
            {"behaviorID": "A", "scoring": scoring["A"]},
            {"behaviorID": "B", "scoring": scoring["B"]},
        ], rng=random.Random(2))
        ordered = chooser.get_sorted_choices()
        self.assertEqual(sorted(ordered), ["A", "B"])
        chooser.apply_repetition_penalty("A")
        self.assertEqual(chooser.behavior_repetitions[0], 1)
        self.assertAlmostEqual(chooser.behavior_scores[0], 9)
        self.assertAlmostEqual(chooser.total_score, 14)
        chooser.apply_repetition_penalty("A")
        self.assertAlmostEqual(chooser.behavior_scores[0], 7)
        chooser.apply_repetition_penalty("missing")
        chooser.reset()
        self.assertEqual(chooser.behavior_scores, [10.0, 5.0])
        self.assertEqual(chooser.choose({"B"}), "B")

    def test_scoring_empty_and_unknown(self):
        self.assertIsNone(activity.BehaviorChooser("Scoring", []).get_sorted_choices())
        chooser = activity.BehaviorChooser("Nope", ["A"])
        with self.assertRaises(ValueError):
            chooser.get_sorted_choices()
        with self.assertRaises(ValueError):
            chooser.choose()

    def test_from_json(self):
        chooser = activity.BehaviorChooser.from_json(
            chooser_json("Selection", ("A", "B")))
        self.assertEqual(chooser.choice_type, "Selection")
        self.assertEqual(chooser.get_sorted_choices(), ["A", "B"])


class TestActivities(unittest.TestCase):

    def test_objective_defaults(self):
        obj = activity.Objective("o", "b", "", 0.5)
        self.assertEqual(obj.random_completions_needed_min, 0)
        self.assertEqual(obj.random_completions_needed_max, 0)
        obj = activity.Objective.from_json({
            "objective": "o", "behaviorID": "b", "ignoreIfLocked": "x",
            "probabilityToRequireObjective": 0.25,
        })
        self.assertEqual(obj.probability_to_require_objective, 0.25)
        self.assertEqual(obj.random_completions_needed_min, 0)

    def test_base_activity(self):
        obj = activity.Activity("A", "Custom", "Selection")
        self.assertTrue(obj.available())
        self.assertIsNone(obj.choose())
        self.assertEqual(obj.to_dict(), {
            "activityID": "A", "activityType": "Custom",
            "activityStrategy": {"type": "Selection"}})

    def test_behaviors_activity(self):
        obj = activity.BehaviorsActivity.from_json(activity_json("A"))
        self.assertTrue(obj.available())
        self.assertEqual(obj.choose(), "B1")
        self.assertEqual(obj.choose_behaviors(), ["B1", "B2"])
        self.assertEqual(obj.choose({"available_behaviors": {"B2"}}), "B2")
        empty = activity.BehaviorsActivity(None, "E", "BehaviorsOnly", "Selection")
        self.assertFalse(empty.available())
        self.assertIsNone(empty.choose())
        self.assertEqual(empty.choose_behaviors(), [])

    def test_voice_command_activity(self):
        obj = activity.from_dict(activity_json("V", "VoiceCommand"))
        self.assertIsInstance(obj, activity.VoiceCommandActivity)
        self.assertEqual(obj.id, "V")

    def test_feeding_activity(self):
        data = activity_json("F", "Feeding")
        data.pop("behaviorChooser")
        data["universalChooser"] = {"behaviors": ["B1", "B2"]}
        obj = activity.FeedingActivity.from_json(data)
        self.assertTrue(obj.available())
        self.assertIn(obj.choose(rng=random.Random(0)), ("B1", "B2"))
        self.assertEqual(obj.choose({"available_behaviors": {"B2"}}, random.Random(0)), "B2")
        self.assertIsNone(obj.choose({"available_behaviors": set()}))
        empty = activity.FeedingActivity(None, "E", "Feeding", "Selection")
        self.assertFalse(empty.available())
        self.assertIsNone(empty.choose())

    def test_freeplay_activity(self):
        data = activity_json("P", "Freeplay")
        data.pop("behaviorChooser")
        data["desiredActivityNames"] = {
            "cubeOnlyActivityName": "Cube",
            "faceOnlyActivityName": "Face",
            "faceAndCubeActivityName": "Both",
            "noFaceNoCubeActivityName": "Neither",
        }
        data["subActivities"] = ["Cube", "Face", "Both", "Neither"]
        obj = activity.FreeplayActivity.from_json(data)
        self.assertEqual(obj.choose_sub_activity({}), "Neither")
        self.assertEqual(obj.choose_sub_activity({"face_visible": True}), "Face")
        self.assertEqual(obj.choose_sub_activity({"cube_visible": True}), "Cube")
        self.assertEqual(obj.choose_sub_activity({"faces": True, "cubes": True}), "Both")
        self.assertEqual(obj.choose(), "Neither")
        self.assertTrue(obj.available())

    def test_sparked_activity(self):
        data = activity_json("S", "Sparked", "StrictPriority", ("B1", "B2"))
        data.update({
            "requireSpark": "true", "minTimeSecs": 1, "maxTimeSecs": 2,
            "numberOfRepetitions": 2, "behaviorObjective": "obj",
            "softSparkTrigger": "soft",
        })
        obj = activity.SparkedActivity.from_json(data)
        self.assertFalse(obj.available({"spark": False}))
        self.assertIsNone(obj.choose({"spark": False}))
        self.assertTrue(obj.available({"spark": True}))
        self.assertEqual(obj.choose({"spark": True}), "B1")
        none_spark = activity.SparkedActivity.from_json(dict(data, requireSpark="none"))
        self.assertTrue(none_spark.available())

    def test_sparked_activity_delegate(self):
        data = activity_json("S", "Sparked", "Selection", ("B1",))
        data.update({
            "requireSpark": "false", "minTimeSecs": 1, "maxTimeSecs": 2,
            "numberOfRepetitions": 1, "behaviorObjective": "obj",
            "softSparkTrigger": "soft",
            "subActivityDelegate": activity_json("D", names=("D1",)),
        })
        obj = activity.SparkedActivity.from_json(data)
        self.assertIsNotNone(obj.sub_activity_delegate)
        self.assertEqual(obj.choose({"use_sub_activity": True}), "D1")
        self.assertEqual(obj.choose(), "B1")

    def test_pyramid_activity(self):
        data = activity_json("Y", "BuildPyramid", "Selection", ("setup",))
        data.pop("behaviorChooser")
        data["setupChooser"] = chooser_json("Selection", ("setup",))
        data["buildChooser"] = chooser_json("Selection", ("build",))
        data["interludeBehaviorChooser"] = chooser_json("Selection", ("interlude",))
        data["needsActionID"] = "Needs"
        obj = activity.PyramidActivity.from_json(data)
        self.assertEqual(obj.choose({}), "setup")
        self.assertEqual(obj.choose({"pyramid_started": True}), "build")
        self.assertEqual(obj.choose({"pyramid_built": True}), "interlude")
        self.assertEqual(obj.needs_action_id, "Needs")

    def test_socialize_activity(self):
        data = activity_json("So", "Socialize", "Selection", ("B1",))
        data.pop("behaviorChooser")
        data["behaviorChooser"] = chooser_json("Selection", ("face",))
        data["interludeBehaviorChooser"] = chooser_json("Selection", ("interlude",))
        data["maxNumFindFacesSearchIterations"] = 1
        data["requiredObjectives"] = [{
            "objective": "o", "behaviorID": "b", "ignoreIfLocked": "",
            "probabilityToRequireObjective": 0.5,
        }]
        obj = activity.SocializeActivity.from_json(data)
        self.assertEqual(obj.choose({"face_visible": True}), "face")
        self.assertEqual(obj.face_iterations, 1)
        self.assertEqual(obj.choose({}), "interlude")
        self.assertEqual(len(obj.required_objectives), 1)
        obj.reset()
        self.assertEqual(obj.face_iterations, 0)
        self.assertEqual(obj.choose({}), "face")

    def test_needs_activity(self):
        data = activity_json("N", "NeedsExpression", "Selection", ("need",))
        obj = activity.NeedsActivity.from_json(data)
        self.assertEqual(obj.choose(), "need")
        self.assertEqual(obj.choose({"available_behaviors": {"need"}}), "need")
        empty = activity.NeedsActivity(None, "E", "NeedsExpression", "Selection")
        self.assertIsNone(empty.choose())

    def test_unknown_activity_type(self):
        obj = activity.from_dict(activity_json("X", "Mystery"))
        self.assertIs(type(obj), activity.Activity)
        self.assertEqual(obj.id, "X")


class TestActivityLoader(unittest.TestCase):

    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.base = os.path.join(self.root, "cozmo_resources", "config", "engine")
        self.system = os.path.join(self.base, "behaviorSystem")
        self.activities_dir = os.path.join(self.system, "activities")
        os.makedirs(self.activities_dir)

    def tearDown(self):
        for dirpath, _, filenames in os.walk(self.root, topdown=False):
            for filename in filenames:
                os.unlink(os.path.join(dirpath, filename))
            os.rmdir(dirpath)

    def write(self, path, data):
        with open(path, "w") as stream:
            if isinstance(data, str):
                stream.write(data)
            else:
                json.dump(data, stream)

    def test_loads_all_supported_shapes(self):
        self.write(os.path.join(self.system, "activities_config.json"),
                   {"activities": [
                       activity_json("A", "BehaviorsOnly"),
                       activity_json("V", "VoiceCommand"),
                   ]})
        self.write(os.path.join(self.system, "behavior_system_config.json"),
                   {"activityList": [activity_json("F", "Freeplay", **{
                       "desiredActivityNames": {
                           "cubeOnlyActivityName": "Cube",
                           "faceOnlyActivityName": "Face",
                           "faceAndCubeActivityName": "Both",
                           "noFaceNoCubeActivityName": "Neither",
                       },
                       "subActivities": [],
                   })]})
        self.write(os.path.join(self.activities_dir, "extra.json"),
                   activity_json("S", "Sparked", **{
                       "requireSpark": "false", "minTimeSecs": 1, "maxTimeSecs": 2,
                       "numberOfRepetitions": 1, "behaviorObjective": "obj",
                       "softSparkTrigger": "soft",
                   }))
        self.write(os.path.join(self.activities_dir, "broken.json"), "{not json")
        self.write(os.path.join(self.base, "needs_action_config.json"),
                   activity_json("N", "NeedsExpression"))
        self.write(os.path.join(self.base, "do_a_trick_weights.json"),
                   {"tricks": [{"name": "ignored"}]})

        loaded = activity.load_activities(self.root)
        self.assertEqual(set(loaded), {"A", "V", "F", "S", "N"})
        self.assertIsInstance(loaded["A"], activity.BehaviorsActivity)
        self.assertIsInstance(loaded["V"], activity.VoiceCommandActivity)
        self.assertIsInstance(loaded["F"], activity.FreeplayActivity)
        self.assertIsInstance(loaded["S"], activity.SparkedActivity)
        self.assertIsInstance(loaded["N"], activity.NeedsActivity)

    def test_skips_malformed_activities(self):
        self.write(os.path.join(self.system, "activities_config.json"), {
            "activities": [
                {"activityID": "broken", "activityType": "BehaviorsOnly"},
                activity_json("ok"),
            ],
        })
        loaded = activity.load_activities(self.root)
        self.assertEqual(set(loaded), {"ok"})


if __name__ == "__main__":
    unittest.main()
