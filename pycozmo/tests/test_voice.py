
import unittest

from pycozmo.voice import (Command, CommandExecutor, CommandParser, CommandRule,
                           execute_command, parse_command, parse_number)


class FakeClient:
    def __init__(self):
        self.calls = []
        self.anim_groups = []
        self.animations = []
        self.lights = []
        self.volume = None
        self.audio = []
        self.battery_voltage = 4.1
        self.latest_image = None
        self.camera_enabled = None
        self.charger_result = True

    def set_head_angle(self, angle, *args, **kwargs):
        self.calls.append(("head", angle))

    def set_lift_height(self, height, *args, **kwargs):
        self.calls.append(("lift", height))

    def drive_straight(self, speed, duration=None, *args, **kwargs):
        self.calls.append(("drive", speed, duration))

    def drive_wheels(self, left, right, duration=None, *args, **kwargs):
        self.calls.append(("wheels", left, right, duration))

    def turn_in_place_at_speed(self, speed, direction=1, *args, **kwargs):
        self.calls.append(("turn", speed, direction))

    def stop_all_motors(self):
        self.calls.append(("stop",))

    def cancel_anim(self):
        self.calls.append(("cancel_anim",))

    def play_anim_group(self, name):
        self.anim_groups.append(name)

    def play_anim(self, name):
        self.animations.append(name)

    def say(self, text, *args, **kwargs):
        self.audio.append(text)

    def set_all_backpack_lights(self, light):
        self.lights.append(light)

    def set_center_backpack_lights(self, light):
        self.lights.append(light)

    def set_volume(self, level):
        self.volume = level

    def enable_camera(self, enable=True, *args, **kwargs):
        self.camera_enabled = enable

    def drive_onto_charger(self):
        self.calls.append(("charger",))
        return self.charger_result


class TestNumberParsing(unittest.TestCase):

    def test_numbers(self):
        self.assertEqual(parse_number(42), 42.0)
        self.assertEqual(parse_number("42"), 42.0)
        self.assertEqual(parse_number("12.5 degrees"), 12.5)
        self.assertEqual(parse_number("ninety"), 90.0)
        self.assertEqual(parse_number("twenty five"), 25.0)
        self.assertEqual(parse_number(None, 7), 7.0)
        self.assertEqual(parse_number("not a number", 3), 3.0)


class TestCommandParser(unittest.TestCase):

    def setUp(self):
        self.parser = CommandParser()

    def assertCommand(self, utterance, expected_name, **parameters):
        command = self.parser.parse(utterance)
        self.assertIsNotNone(command, utterance)
        self.assertEqual(command.name, expected_name)
        for key, value in parameters.items():
            self.assertEqual(command.get(key), value)

    def test_help(self):
        self.assertCommand("help", "help")
        self.assertCommand("what can you do?", "help")

    def test_stop(self):
        self.assertCommand("stop", "stop")
        self.assertCommand("stop all motors", "stop")

    def test_head(self):
        self.assertCommand("look up", "head", angle_deg=34.0)
        self.assertCommand("look down", "head", angle_deg=-17.0)
        self.assertCommand("raise your head", "head", angle_deg=34.0)
        self.assertCommand("look straight", "head", angle_deg=0.0)
        self.assertCommand("look at me", "head", angle_deg=17.0)
        self.assertCommand("head angle to -10", "head", angle_deg=-10.0)

    def test_lift(self):
        self.assertCommand("lift up", "lift", height_mm=80.0)
        self.assertCommand("lower your lift", "lift", height_mm=0.0)
        self.assertCommand("set lift height to 25", "lift", height_mm=25.0)

    def test_drive_and_turn(self):
        self.assertCommand("drive forward", "drive", direction="forward")
        self.assertCommand("drive forward for 2 seconds at 60", "drive",
                           direction="forward", duration=2.0, speed=60.0)
        self.assertCommand("move backward for 1.5 seconds", "drive",
                           direction="backward", duration=1.5)
        self.assertCommand("turn left", "turn", direction="left", angle_deg=90.0)
        self.assertCommand("turn right 45 degrees", "turn",
                           direction="right", angle_deg=45.0)
        self.assertCommand("spin", "spin", direction="right", angle_deg=360.0)

    def test_lights_and_volume(self):
        self.assertCommand("lights red", "lights", color="red")
        self.assertCommand("set backpack lights to blue", "lights",
                           scope="backpack", color="blue")
        self.assertCommand("turn the lights off", "lights_off", color="off")
        self.assertCommand("set volume 75", "volume", percent=75.0)
        self.assertCommand("volume 50 percent", "volume", percent=50.0)
        self.assertCommand("mute", "mute")
        self.assertCommand("unmute", "unmute")

    def test_animations(self):
        self.assertCommand("play animation happy", "play_anim", name="happy")
        self.assertCommand("play happy", "play_anim", name="happy")
        self.assertCommand("dance", "dance")
        self.assertCommand("celebrate", "celebrate")
        self.assertCommand("do a trick", "trick")

    def test_say(self):
        self.assertCommand("say hello there", "say", text="hello there")
        self.assertCommand("tell me a joke", "say", text="a joke")

    def test_misc(self):
        self.assertCommand("take a picture", "camera")
        self.assertCommand("go to the charger", "charge")
        self.assertCommand("go to sleep", "sleep")
        self.assertCommand("wake up", "wake")
        self.assertCommand("follow me", "follow")
        self.assertCommand("come here", "come")
        self.assertCommand("what is your name", "name")
        self.assertCommand("how are you", "status")
        self.assertCommand("yes", "yes")
        self.assertCommand("nope", "no")

    def test_unknown(self):
        self.assertIsNone(self.parser.parse(""))
        self.assertIsNone(self.parser.parse("defenestrate the robot"))

    def test_custom_rule(self):
        parser = CommandParser(rules=[])
        parser.add("echo", r"echo (?P<text>.+)",
                   lambda match: {"text": match.group("text").upper()},
                   confidence=0.5)
        command = parser.parse("Echo hello", confidence=0.8)
        self.assertEqual(command.name, "echo")
        self.assertEqual(command["text"], "HELLO")
        self.assertAlmostEqual(command.confidence, 0.4)
        self.assertIsNone(parser.parse("something else"))

    def test_rule_create(self):
        rule = CommandRule.create("test", r"test (?P<value>\d+)")
        self.assertEqual(rule.pattern.fullmatch("test 12").group("value"), "12")
        self.assertEqual(rule.confidence, 1.0)

    def test_parse_all(self):
        parser = CommandParser(rules=[])
        parser.add("one", r"same")
        parser.add("two", r"same")
        commands = parser.parse_all("same")
        self.assertEqual([command.name for command in commands], ["one", "two"])

    def test_parse_command_helper(self):
        command = parse_command("dance")
        self.assertEqual(command.name, "dance")
        self.assertIsInstance(command, Command)


class TestCommandExecutor(unittest.TestCase):

    def setUp(self):
        self.client = FakeClient()
        self.slept = []
        self.executor = CommandExecutor(self.client, sleep=self.slept.append)

    def execute(self, text):
        command = parse_command(text)
        self.assertIsNotNone(command, text)
        return self.executor.execute(command)

    def test_stop(self):
        result = self.execute("stop")
        self.assertTrue(result.success)
        self.assertIn(("stop",), self.client.calls)
        self.assertIn(("cancel_anim",), self.client.calls)

    def test_head_and_lift(self):
        self.assertTrue(self.execute("look up").success)
        self.assertTrue(self.execute("lift up").success)
        self.assertAlmostEqual(self.client.calls[0][1], 0.5934, places=3)
        self.assertEqual(self.client.calls[1][1], 80.0)

    def test_drive(self):
        result = self.execute("drive forward for 2 seconds at 60")
        self.assertTrue(result.success)
        self.assertIn(("drive", 60.0, 2.0), self.client.calls)
        result = self.execute("drive backward for 1 second at 40")
        self.assertTrue(result.success)
        self.assertIn(("drive", -40.0, 1.0), self.client.calls)

    def test_turn(self):
        result = self.execute("turn left 90 degrees")
        self.assertTrue(result.success)
        self.assertIn(("turn", 50.0, 1), self.client.calls)
        self.assertIn(("stop",), self.client.calls)
        self.assertAlmostEqual(self.slept[0], 1.0)

    def test_turn_without_turn_method(self):
        class NoTurnClient:
            def __init__(self):
                self.calls = []

            def drive_wheels(self, left, right, duration=None):
                self.calls.append((left, right, duration))

        client = NoTurnClient()
        result = CommandExecutor(client, sleep=self.slept.append).execute(
            parse_command("turn right 90 degrees"))
        self.assertTrue(result.success)
        self.assertIn((50.0, -50.0, 1.0), client.calls)

    def test_say_and_animation(self):
        self.assertTrue(self.execute("say hello").success)
        self.assertEqual(self.client.audio, ["hello"])
        self.assertTrue(self.execute("play animation happy").success)
        self.assertEqual(self.client.anim_groups, ["happy"])
        self.assertTrue(self.execute("dance").success)
        self.assertEqual(self.client.anim_groups[-1], "Dance")
        self.assertTrue(self.execute("celebrate").success)
        self.assertTrue(self.execute("do a trick").success)
        self.assertEqual(self.client.anim_groups[-1], "DoATrick")

    def test_lights(self):
        from pycozmo import lights
        self.assertTrue(self.execute("lights red").success)
        self.assertTrue(self.execute("lights off").success)
        self.assertEqual(len(self.client.lights), 2)
        self.assertEqual(self.client.lights[0].on_color,
                         lights.red_light.on_color)
        self.assertNotEqual(self.client.lights[0].on_color,
                            self.client.lights[1].on_color)

    def test_volume(self):
        self.assertTrue(self.execute("set volume 50").success)
        self.assertEqual(self.client.volume, 32768)
        self.assertTrue(self.execute("mute").success)
        self.assertEqual(self.client.volume, 0)
        self.assertTrue(self.execute("unmute").success)
        self.assertEqual(self.client.volume, 65535)

    def test_camera(self):
        result = self.execute("take a picture")
        self.assertTrue(result.success)
        self.assertEqual(self.client.camera_enabled, True)

    def test_charge(self):
        result = self.execute("go to the charger")
        self.assertTrue(result.success)
        self.assertIn("On charger", result.message)
        self.client.charger_result = False
        result = self.execute("charge")
        self.assertIn("Could not reach", result.message)

    def test_status_and_name(self):
        result = self.execute("status")
        self.assertIn("4.10", result.message)
        result = self.execute("what is your name")
        self.assertEqual(result.message, "I am Cozmo.")

    def test_errors_are_reported(self):
        class NoSpeechClient:
            pass

        executor = CommandExecutor(NoSpeechClient(), sleep=self.slept.append)
        result = executor.execute(parse_command("say hello"))
        self.assertFalse(result.success)
        self.assertIn("does not support text-to-speech", result.error)
        result = self.execute("follow me")
        self.assertFalse(result.success)

    def test_unknown_command(self):
        result = self.executor.execute(Command("unknown"))
        self.assertFalse(result.success)
        self.assertIn("Unknown command", result.error)

    def test_execute_command_helper(self):
        result = execute_command(self.client, parse_command("dance"),
                                 sleep=self.slept.append)
        self.assertTrue(result.success)
        self.assertIn("Dance", self.client.anim_groups)

    def test_command_result_serialization(self):
        result = self.execute("dance")
        data = result.to_dict()
        self.assertTrue(data["success"])
        self.assertEqual(data["command"]["name"], "dance")


if __name__ == "__main__":
    unittest.main()
