"""

Voice command parsing and execution.

The parser is independent of any speech-recognition backend.  It turns text
into a :class:`Command`, which can be executed with :func:`execute_command`
against a Cozmo :class:`~pycozmo.client.Client` (or a fake object with the
same methods).  This makes voice control testable without a microphone.

"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
import time
from typing import Any, Callable, Dict, List, Optional, Pattern, Sequence


__all__ = [
    "Command",
    "CommandResult",
    "CommandRule",
    "CommandParser",
    "CommandExecutor",
    "VoiceCommandError",
    "execute_command",
    "parse_command",
]


class VoiceCommandError(ValueError):
    """ Raised when a voice command cannot be parsed or executed. """


@dataclass
class Command:
    """ A parsed voice command. """

    name: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    text: str = ""
    confidence: float = 1.0

    def get(self, key: str, default: Any = None) -> Any:
        return self.parameters.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self.parameters[key]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "parameters": dict(self.parameters),
            "text": self.text,
            "confidence": self.confidence,
        }

    def __repr__(self) -> str:
        return "Command(name={!r}, parameters={!r})".format(self.name, self.parameters)


@dataclass
class CommandResult:
    """ Result of executing a command. """

    command: Command
    success: bool = True
    message: str = ""
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "command": self.command.to_dict(),
            "success": self.success,
            "message": self.message,
            "error": self.error,
        }


@dataclass
class CommandRule:
    """ A named regular expression and its parameter extractor. """

    name: str
    pattern: Pattern
    extractor: Optional[Callable[[re.Match], Dict[str, Any]]] = None
    confidence: float = 1.0

    @classmethod
    def create(cls,
               name: str,
               pattern: str,
               extractor: Optional[Callable[[re.Match], Dict[str, Any]]] = None,
               confidence: float = 1.0) -> "CommandRule":
        return cls(name=name, pattern=re.compile(pattern, re.IGNORECASE),
                   extractor=extractor, confidence=confidence)


_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100,
    "half": 0.5,
}


def parse_number(value: Any, default: float = 0.0) -> float:
    """ Parse an integer/float or a common English number word. """
    if value is None:
        return float(default)
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().lower()
    if not text:
        return float(default)
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if match:
        try:
            return float(match.group(0))
        except ValueError:
            pass
    if text in _NUMBER_WORDS:
        return float(_NUMBER_WORDS[text])
    parts = text.split()
    if len(parts) == 2 and parts[0] in _NUMBER_WORDS and parts[1] in _NUMBER_WORDS:
        return float(_NUMBER_WORDS[parts[0]] + _NUMBER_WORDS[parts[1]])
    return float(default)


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9%.\- ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


class CommandParser:
    """ Ordered regular-expression command parser.

    Rules added later have lower priority.  Add specific rules before generic
    ones (for example ``play animation`` before ``play``).
    """

    def __init__(self, rules: Optional[Sequence[CommandRule]] = None) -> None:
        self.rules: List[CommandRule] = list(rules or ())
        if rules is None:
            self._add_default_rules()

    def add_rule(self, rule: CommandRule) -> "CommandParser":
        self.rules.append(rule)
        return self

    def add(self, name: str, pattern: str,
            extractor: Optional[Callable[[re.Match], Dict[str, Any]]] = None,
            confidence: float = 1.0) -> "CommandParser":
        return self.add_rule(CommandRule.create(name, pattern, extractor, confidence))

    def parse(self, text: str, confidence: float = 1.0) -> Optional[Command]:
        normalized = _normalize(text)
        if not normalized:
            return None
        for rule in self.rules:
            match = rule.pattern.fullmatch(normalized)
            if not match:
                continue
            parameters = {}
            if rule.extractor is not None:
                parameters = dict(rule.extractor(match) or {})
            else:
                parameters = {key: value for key, value in match.groupdict().items()
                              if value is not None}
            return Command(name=rule.name, parameters=parameters, text=text.strip(),
                           confidence=min(1.0, max(0.0, confidence * rule.confidence)))
        return None

    def parse_all(self, text: str) -> List[Command]:
        """ Return every rule that matches, in priority order. """
        normalized = _normalize(text)
        commands = []
        for rule in self.rules:
            match = rule.pattern.fullmatch(normalized)
            if not match:
                continue
            parameters = dict(rule.extractor(match) or {}) if rule.extractor else \
                {key: value for key, value in match.groupdict().items() if value is not None}
            commands.append(Command(name=rule.name, parameters=parameters, text=text.strip()))
        return commands

    # ------------------------------------------------------------------
    # Built-in rules
    # ------------------------------------------------------------------

    def _add_default_rules(self) -> None:
        self.add("help", r"(?:help|what can you do|commands|list commands)")

        self.add("stop", r"(?:stop|halt|freeze|stop moving|stop all motors)")

        self.add("volume", r"(?:set )?volume(?: to)? (?P<percent>\d+|\w+)(?: percent|%)?",
                 lambda m: {"percent": parse_number(m.group("percent"), 50)})
        self.add("mute", r"(?:mute|silence)")
        self.add("unmute", r"(?:unmute|sound on)")

        self.add("lights_off", r"(?:(?:turn|switch) )?(?:the )?(?:backpack )?(?:lights?|leds?) off",
                 lambda m: {"color": "off"})
        self.add("lights",
                 r"(?:(?:set|turn) )?(?:the )?(?:(?P<scope>backpack|center|all) )?"
                 r"(?:lights?|leds?)(?: to)? (?P<color>red|green|blue|white|off|orange|yellow|purple|pink|cyan)")

        self.add("head", r"(?:look|head|tilt) (?P<direction>up|down)",
                 lambda m: {"angle_deg": 34.0 if m.group("direction") == "up" else -17.0})
        self.add("head", r"(?P<direction>raise|lower) (?:your )?head",
                 lambda m: {"angle_deg": 34.0 if m.group("direction") == "raise" else -17.0})
        self.add("head", r"(?:look|head) (?:straight|forward|center)",
                 lambda m: {"angle_deg": 0.0})
        self.add("head", r"look at me", lambda m: {"angle_deg": 17.0})
        self.add("head", r"(?:set )?head (?:angle )?(?:to )?(?P<angle>-?\d+(?:\.\d+)?)",
                 lambda m: {"angle_deg": parse_number(m.group("angle"))})

        self.add("lift", r"(?:lift|raise) up", lambda m: {"height_mm": 80.0})
        self.add("lift", r"(?:lift|lower) down", lambda m: {"height_mm": 0.0})
        self.add("lift", r"(?P<direction>raise|lower) (?:your )?lift",
                 lambda m: {"height_mm": 80.0 if m.group("direction") == "raise" else 0.0})
        self.add("lift", r"(?:set )?lift (?:height )?(?:to )?(?P<height>-?\d+(?:\.\d+)?)",
                 lambda m: {"height_mm": parse_number(m.group("height"))})

        self.add("turn",
                 r"turn (?P<direction>left|right)(?: (?:by |for )?(?P<angle>\d+(?:\.\d+)?|\w+)"
                 r"(?: degrees?| deg)?)?",
                 lambda m: {"direction": m.group("direction"),
                            "angle_deg": parse_number(m.group("angle"), 90.0)})
        self.add("spin", r"(?:spin|spin around|turn around)",
                 lambda m: {"direction": "right", "angle_deg": 360.0})

        self.add("drive",
                 r"(?:drive|move|go) (?P<direction>forward|forwards|ahead|back|backward|backwards)"
                 r"(?: for (?P<duration>\d+(?:\.\d+)?|\w+) (?:seconds?|secs?))?"
                 r"(?: at (?P<speed>\d+(?:\.\d+)?|\w+) ?(?:mm/?s|millimeters per second)?)?",
                 lambda m: {
                     "direction": "forward" if m.group("direction") in
                     ("forward", "forwards", "ahead") else "backward",
                     "duration": parse_number(m.group("duration"), 1.0),
                     "speed": parse_number(m.group("speed"), 50.0),
                 })

        self.add("say", r"(?:say|tell me|repeat) (?P<text>.+)",
                 lambda m: {"text": m.group("text")})

        self.add("play_anim", r"play (?:the )?animation (?P<name>.+)",
                 lambda m: {"name": m.group("name")})
        self.add("dance", r"(?:dance|party|let s dance)")
        self.add("celebrate", r"(?:celebrate|good job|well done|yay)")
        self.add("trick", r"(?:do a trick|show me a trick|trick)")
        self.add("play_anim", r"play (?P<name>.+)", lambda m: {"name": m.group("name")})

        self.add("camera", r"(?:take a (?:picture|photo)|camera|snapshot|take a snapshot)")
        self.add("charge", r"(?:go to (?:the )?charger|charge|dock|go charge)")
        self.add("sleep", r"(?:go to sleep|sleep|good night)")
        self.add("wake", r"(?:wake up|good morning|wake)")
        self.add("follow", r"(?:follow me|follow)")
        self.add("come", r"(?:come here|come to me|come)")

        self.add("name", r"(?:what is your name|what s your name|who are you)")
        self.add("status", r"(?:how are you|battery|battery level|status|how do you feel)")
        self.add("yes", r"(?:yes|yeah|yep|sure|okay|ok)")
        self.add("no", r"(?:no|nope|nah)")


class CommandExecutor:
    """ Execute parsed commands against a Cozmo-like client. """

    def __init__(self, client: Any, sleep: Callable[[float], None] = time.sleep,
                 animation_group: str = "Dance") -> None:
        self.client = client
        self.sleep = sleep
        self.animation_group = animation_group

    def execute(self, command: Command) -> CommandResult:
        try:
            message = self._execute(command)
        except Exception as exc:
            return CommandResult(command=command, success=False,
                                 error="{}: {}".format(type(exc).__name__, exc))
        return CommandResult(command=command, success=True, message=message or "")

    def _execute(self, command: Command) -> str:
        name = command.name
        if name == "stop":
            self.client.stop_all_motors()
            if hasattr(self.client, "cancel_anim"):
                self.client.cancel_anim()
            return "Stopped."
        if name == "head":
            angle = math.radians(float(command.get("angle_deg", 0.0)))
            self.client.set_head_angle(angle)
            return "Head moved."
        if name == "lift":
            self.client.set_lift_height(float(command.get("height_mm", 0.0)))
            return "Lift moved."
        if name == "drive":
            speed = float(command.get("speed", 50.0))
            if command.get("direction") == "backward":
                speed = -speed
            duration = max(0.0, float(command.get("duration", 1.0)))
            self.client.drive_straight(speed, duration=duration)
            return "Drove {} for {:.1f} s.".format(
                command.get("direction", "forward"), duration)
        if name == "turn":
            angle = abs(float(command.get("angle_deg", 90.0)))
            direction = 1 if command.get("direction") == "left" else -1
            # Cozmo turns at roughly 90 degrees per second at wheel speed 50.
            duration = angle / 90.0
            if hasattr(self.client, "turn_in_place_at_speed"):
                self.client.turn_in_place_at_speed(50.0, direction=direction)
                self.sleep(duration)
                self.client.stop_all_motors()
            else:
                self.client.drive_wheels(-50.0 * direction, 50.0 * direction,
                                         duration=duration)
            return "Turned {} degrees.".format(angle)
        if name == "say":
            text = str(command.get("text", "")).strip()
            if hasattr(self.client, "say"):
                self.client.say(text)
            else:
                raise VoiceCommandError("The client does not support text-to-speech.")
            return "Said: {}".format(text)
        if name == "play_anim":
            anim_name = str(command.get("name", "")).strip()
            if hasattr(self.client, "play_anim_group"):
                self.client.play_anim_group(anim_name)
            elif hasattr(self.client, "play_anim"):
                self.client.play_anim(anim_name)
            else:
                raise VoiceCommandError("The client does not support animations.")
            return "Playing {}.".format(anim_name)
        if name == "dance":
            self._play_group("Dance")
            return "Dancing."
        if name == "celebrate":
            self._play_group("Celebrate")
            return "Celebrating."
        if name == "trick":
            self._play_group("DoATrick")
            return "Doing a trick."
        if name in ("lights", "lights_off"):
            color = str(command.get("color", "off")).lower()
            self._set_lights(color, str(command.get("scope", "all")).lower())
            return "Lights {}.".format(color)
        if name == "volume":
            percent = max(0.0, min(100.0, float(command.get("percent", 50.0))))
            self.client.set_volume(int(round(percent * 65535.0 / 100.0)))
            return "Volume set to {:.0f}%.".format(percent)
        if name == "mute":
            self.client.set_volume(0)
            return "Muted."
        if name == "unmute":
            self.client.set_volume(65535)
            return "Unmuted."
        if name == "camera":
            image = getattr(self.client, "latest_image", None)
            if image is None:
                if hasattr(self.client, "enable_camera"):
                    self.client.enable_camera(True)
                return "Camera enabled."
            return "Latest image is {}x{}.".format(*image.size)
        if name == "charge":
            if hasattr(self.client, "drive_onto_charger"):
                reached = self.client.drive_onto_charger()
                return "On charger." if reached else "Could not reach the charger."
            raise VoiceCommandError("The client does not support charger driving.")
        if name == "sleep":
            self._play_group("Sleep")
            return "Going to sleep."
        if name == "wake":
            self._play_group("WakeUp")
            return "Waking up."
        if name == "status":
            battery = getattr(self.client, "battery_voltage", None)
            if battery is None:
                return "I am ready."
            return "Battery voltage is {:.2f} V.".format(float(battery))
        if name == "name":
            return "I am Cozmo."
        if name == "help":
            return "Try: stop, drive forward, turn left, look up, raise lift, say hello, dance, lights red."
        if name in ("yes", "no"):
            return name.capitalize() + "."
        if name == "follow":
            raise VoiceCommandError("Follow mode requires the navigation module.")
        if name == "come":
            raise VoiceCommandError("Come-here requires the navigation module.")
        raise VoiceCommandError("Unknown command '{}'.".format(name))

    def _play_group(self, name: str) -> None:
        if hasattr(self.client, "play_anim_group"):
            self.client.play_anim_group(name)
        elif hasattr(self.client, "play_anim"):
            self.client.play_anim(name)
        else:
            raise VoiceCommandError("The client does not support animations.")

    def _set_lights(self, color: str, scope: str = "all") -> None:
        from .. import lights
        palette = {
            "red": lights.red_light,
            "green": lights.green_light,
            "blue": lights.blue_light,
            "white": lights.white_light,
            "off": lights.off_light,
            "orange": lights.protocol_encoder.LightState(
                on_color=lights.Color(rgb=(255, 120, 0)).to_int16(),
                off_color=lights.Color(rgb=(255, 120, 0)).to_int16()),
            "yellow": lights.protocol_encoder.LightState(
                on_color=lights.Color(rgb=(255, 220, 0)).to_int16(),
                off_color=lights.Color(rgb=(255, 220, 0)).to_int16()),
            "purple": lights.protocol_encoder.LightState(
                on_color=lights.Color(rgb=(160, 0, 255)).to_int16(),
                off_color=lights.Color(rgb=(160, 0, 255)).to_int16()),
            "pink": lights.protocol_encoder.LightState(
                on_color=lights.Color(rgb=(255, 80, 160)).to_int16(),
                off_color=lights.Color(rgb=(255, 80, 160)).to_int16()),
            "cyan": lights.protocol_encoder.LightState(
                on_color=lights.Color(rgb=(0, 220, 220)).to_int16(),
                off_color=lights.Color(rgb=(0, 220, 220)).to_int16()),
        }
        light = palette.get(color, lights.off_light)
        if scope == "center" and hasattr(self.client, "set_center_backpack_lights"):
            self.client.set_center_backpack_lights(light)
        elif hasattr(self.client, "set_all_backpack_lights"):
            self.client.set_all_backpack_lights(light)
        elif hasattr(self.client, "set_backpack_lights"):
            self.client.set_backpack_lights(light, light, light, light, light)
        else:
            raise VoiceCommandError("The client does not support backpack lights.")


def parse_command(text: str,
                  parser: Optional[CommandParser] = None) -> Optional[Command]:
    """ Parse ``text`` with a default or supplied parser. """
    return (parser or CommandParser()).parse(text)


def execute_command(client: Any, command: Command,
                    sleep: Callable[[float], None] = time.sleep) -> CommandResult:
    """ Execute a command against ``client``. """
    return CommandExecutor(client, sleep=sleep).execute(command)
