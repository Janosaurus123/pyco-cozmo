
PyCozmo (pyco-cozmo)
====================

[![CI](https://github.com/Janosaus123/pyco-cozmo/actions/workflows/ci.yml/badge.svg)](https://github.com/Janosaus123/pyco-cozmo/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/pyco-cozmo.svg)](https://pypi.org/project/pyco-cozmo/)
[![Python versions](https://img.shields.io/pypi/pyversions/pyco-cozmo.svg)](https://pypi.org/project/pyco-cozmo/)

`PyCozmo` is a pure-Python communication library, alternative SDK, and application for the
[Cozmo robot](https://www.digitaldreamlabs.com/pages/cozmo). It allows controlling a Cozmo robot directly, without
having to go through a mobile device, running the Cozmo app.  It also includes off-board vision, navigation,
personality, voice, music and animation tools, plus readers for Anki Vector animation files.

This repository is the `pyco-cozmo` distribution, an enhanced fork of the original
[PyCozmo project](https://github.com/zayfod/pycozmo). Install it with `pip install pyco-cozmo`; the Python
import name remains `pycozmo`.

The library is loosely based on the [Anki Cozmo Python SDK](https://github.com/anki/cozmo-python-sdk) and the
[cozmoclad](https://pypi.org/project/cozmoclad/) ("C-Like Abstract Data") library.

This project is a tool for exploring the hardware and software of the Digital Dream Labs (originally Anki) Cozmo robot.
The low-level protocol implementation is stable; the higher-level application modules continue to evolve.


Usage
-----

Basic:

```python
import time
import pycozmo

with pycozmo.connect() as cli:
    cli.set_head_angle(angle=0.6)
    time.sleep(1)
```

Advanced:

```python
import pycozmo

cli = pycozmo.Client()
cli.start()
cli.connect()
cli.wait_for_robot()

cli.drive_wheels(lwheel_speed=50.0, rwheel_speed=50.0, duration=2.0)

cli.disconnect()
cli.stop()
```


Feature overview
----------------

- **Protocol and robot control** - Cozmo UDP protocol, animations, audio, camera,
  cubes, charger, firmware updates, and the original tools.
- **Personality engine** - traits, drives, emotions, decay, event reactions, and
  activity/behavior scoring (`pycozmo.personality`).
- **Behaviors and activities** - sequential and random animation behaviors, cliff
  reactions, drive-off-charger, behavior choosers, and resource loaders
  (`pycozmo.behavior`, `pycozmo.activity`).
- **Computer vision** - motion, faces, expressions, recognition, pets, robots,
  cubes, platform/charger detection, custom markers, ArUco markers, calibration,
  drivable-area estimation, and tracking (`pycozmo.vision`).  OpenCV is optional.
- **Navigation** - occupancy grids, map building, A* planning, path
  simplification, and path following (`pycozmo.navigation`).
- **Text-to-speech and songs** - espeak/SAPI/macOS/pyttsx3 backends and a small
  pure-Python song notation and synthesizer (`pycozmo.tts`, `pycozmo.songs`).
- **Voice commands** - a dependency-free command parser/executor and an optional
  `SpeechRecognition` wrapper (`pycozmo.voice`).
- **Vector animations and behaviors** - read Vector `.bin` FlatBuffers files,
  convert common tracks to Cozmo clips, play them, and drive animation-based
  behaviors (`pycozmo.vector`).
- **Discovery and Wi-Fi** - cross-platform network scanning, connection helpers,
  and robot discovery (`pycozmo.wifi`, `pycozmo.discovery`).

Quick examples:

```python
from pycozmo import vision

pipeline = vision.VisionPipeline(vision.VisionConfig(faces=True, cubes=True))
result = pipeline.process(image)
annotated = result.annotate(image)
```

```python
from pycozmo import personality

mood = personality.Personality.playful()
mood.update(dt=10.0)
mood.react("EvtFaceObserved")
activity_id = mood.choose_activity(activities)
```

```python
from pycozmo import songs

song = songs.parse("C4/4 D4/4 E4/4 G4/4 C5/2")
songs.play(cli, song)
```

See [docs/offboard_functions.md](docs/offboard_functions.md) for details and the
[examples](examples) directory for runnable programs.


Documentation
-------------

- [Off-board functions and new modules](docs/offboard_functions.md)
- [Protocol and architecture](docs/architecture.md)
- [Upstream API documentation](https://pycozmo.readthedocs.io/) (does not cover fork-only modules)


Robot Support
-------------

Sensors:
- [x] Camera
- [x] Cliff sensor
- [x] Accelerometers
- [x] Gyro
- [x] Battery voltage
- [x] Cube battery voltage
- [x] Cube accelerometers
- [x] Backpack button (v1.5 hardware and newer)

Actuators:
- [x] Wheel motors
- [x] Head motor
- [x] Lift motor
- [x] OLED display
- [x] Speaker
- [x] Backpack LEDs
- [x] IR LED
- [x] Cube LEDs
- [x] Platform LEDs (when available)

On-board functions (see [docs/functions.md](docs/functions.md) for details):
- [x] Wi-Fi AP
- [x] Bluetooth LE
- [x] Localization
- [x] Path tracking
- [x] NV RAM storage
- [x] Over-the-air (OTA) firmware updates

Off-board functions (see [docs/offboard_functions.md](docs/offboard_functions.md) for details):
- [x] Procedural face generation
- [x] Cozmo animations from FlatBuffers .bin files
- [x] Personality engine
- [x] Cozmo behaviors
- [x] Motion detection
- [x] Object (cube and platform) detection
- [x] Cube marker recognition
- [x] Face detection
- [x] Face recognition
- [x] Facial expression estimation
- [x] Pet detection
- [x] Camera calibration
- [x] Navigation map building
- [x] Text-to-speech
- [x] Songs

Extra off-board functions:
- [x] Vector animations from FlatBuffers .bin files
- [x] Vector behaviors
- [x] ArUco marker recognition
- [x] Cozmo and Vector robot detection
- [x] Drivable area estimation
- [x] Voice commands

If you have ideas for other functionality [share them via GitHub](https://github.com/Janosaus123/pyco-cozmo/issues).


Tools
-----

- [pycozmo_app.py](tools/pycozmo_app.py) - an alternative Cozmo application, implementing off-board functions
    ([video](https://youtu.be/gMEc6RzIm-E)).
- [pycozmo_dump.py](tools/pycozmo_dump.py) - a command-line application that can read and annotate Cozmo communication
    from [pcap files](https://en.wikipedia.org/wiki/Pcap) or capture it live using
    [pypcap](https://github.com/pynetwork/pypcap).
- [pycozmo_replay.py](tools/pycozmo_replay.py) - a basic command-line application that can replay .pcap files back to
    Cozmo.
- [pycozmo_anim.py](tools/pycozmo_anim.py) - a tool for examining and manipulating animation files.
- [pycozmo_update.py](tools/pycozmo_update.py) - a tool for over-the-air (OTA) updates of Cozmo's firmware.
- [pycozmo_discover.py](tools/pycozmo_discover.py) - scans Wi-Fi and probes robot
    addresses (`pycozmo-discover` after installation).
- [pycozmo_protocol_generator.py](tools/pycozmo_protocol_generator.py) - a tool for generating Cozmo protocol encoder
    code.

**Note**: PyCozmo and `pycozmo_protocol_generator.py` in particular could be used as a base for creating a Cozmo protocol
encoder code generator for languages other than Python (C/C++, Java, etc.).
 

Examples
--------

Basic:
- [minimal.py](examples/minimal.py) - minimal code to communicate with Cozmo, using PyCozmo
- [extremes.py](examples/extremes.py) - demonstrates Cozmo lift and head control
- [backpack_lights.py](examples/backpack_lights.py) - demonstrates Cozmo backpack LED control
- [display_image.py](examples/display_image.py) - demonstrates visualization of image files on Cozmo's display
- [events.py](examples/events.py) - demonstrates event handling
- [camera.py](examples/camera.py) - demonstrates capturing a camera image 
- [go_to_pose.py](examples/go_to_pose.py) - demonstrates moving to a specific pose (position and orientation) 
- [path.py](examples/path.py) - demonstrates following a predefined path

Advanced:
- [display_lines.py](examples/display_lines.py) - demonstrates 2D graphics, using
    [PIL.ImageDraw](https://pillow.readthedocs.io/en/stable/reference/ImageDraw.html) on Cozmo's display
    ([video](https://youtu.be/ZT81PlmItrU))
- [rc.py](examples/rc.py) - turns Cozmo into an RC tank that can be driven with an XBox 360 Wireless controller or 
    Logitech Gamepad F310
- [video.py](examples/video.py) - demonstrates visualizing video captured from the camera back on display
- [cube_lights.py](examples/cube_lights.py) - demonstrates cube connection and LED control
- [cube_light_animation.py](examples/cube_light_animation.py) - demonstrates cube LED animation control
- [charger_lights.py](examples/charger_lights.py) - demonstrates Cozmo charging platform LED control
- [audio.py](examples/audio.py) - demonstrates 22 kHz, 16-bit, mono WAVE file playback through Cozmo's speaker 
- [nvram.py](examples/nvram.py) - demonstrates reading data from Cozmo's NVRAM (non-volatile memory)
- [procedural_face.py](examples/procedural_face.py) - demonstrates drawing a procedural face on Cozmo's display
- [procedural_face_show.py](examples/procedural_face_show.py) - demonstrates generating a procedural face
- [procedural_face_expressions.py](examples/procedural_face_expressions.py) - demonstrates displaying various facial
    expressions ([video](https://youtu.be/UyMTEOG2FyU))
- [anim.py](examples/anim.py) - demonstrates animating Cozmo

Off-board and Vector:
- [discovery.py](examples/discovery.py) - scans for Cozmo/Vector robots
- [vision.py](examples/vision.py) - runs the vision pipeline on camera frames and saves an annotated image
- [navigation.py](examples/navigation.py) - builds an occupancy grid, plans a path and follows it
- [personality.py](examples/personality.py) - updates a personality and chooses an activity
- [tts_song.py](examples/tts_song.py) - speaks text and plays a generated song
- [voice_commands.py](examples/voice_commands.py) - parses and executes typed or spoken commands
- [vector_anim.py](examples/vector_anim.py) - loads and plays a Vector animation file


PyCozmo In The Wild
-------------------

- [Expressive Eyes](https://git.brl.ac.uk/ca2-chambers/expressive-eyes) - rendering various facial expressions using
    PyCozmo's procedural_face module
- an [application](https://github.com/paulbaumgarten/paulbaumgarten/blob/6d9025ec715e0b5f133ce9b7e39e74a5389c5334/myp/cozmo/assets/02-camera.py)
    that recognizes [ArUco markers](https://docs.opencv.org/master/d5/dae/tutorial_aruco_detection.html), using OpenCV
- a [ROS2 driver](https://github.com/solosito/cozmo_ros2_ws/)
- another [ROS2 driver](https://github.com/brean/cozmo_ros)


Connecting to Cozmo over Wi-Fi
------------------------------

A Wi-Fi connection needs to be established with Cozmo before using PyCozmo applications.

1. Wake up Cozmo by placing it on the charging platform
2. Make Cozmo display it's Wi-Fi PSK by rising and lowering its lift
3. Scan for Cozmo's Wi-Fi SSID (depends on the OS)
4. Connect using Cozmo's Wi-Fi PSK (depends on the OS)

[This video](https://www.youtube.com/watch?v=-k_oiQhBa5o) summarizes the connection process.

The same steps can be automated:

```python
from pycozmo import discovery, wifi

networks = wifi.find_cozmo_networks()
if networks:
    connection = wifi.connect(networks[0].ssid, password="123456")
    print(connection.to_dict())

robot = discovery.find_robot()
if robot:
    print(robot)
```

`pycozmo-discover --networks` prints visible networks and robots, and
`pycozmo-discover --json` emits machine-readable output.


PyCozmo vs. the Cozmo SDK
-------------------------

A Cozmo SDK application (aka "game") acts as a client to the Cozmo app (aka "engine") that runs on a mobile device.
The low-level communication happens over USB and is handled by the `cozmoclad` library.

In contrast, an application using PyCozmo basically replaces the Cozmo app and acts as the "engine". PyCozmo handles
the low-level UDP communication with Cozmo.
   
```
+------------------+                   +------------------+                   +------------------+
|     SDK app      |     Cozmo SDK     |    Cozmo app     |       Cozmo       |      Cozmo       |
|      "game"      |     cozmoclad     |     "engine"     |      protocol     |     "robot"      |
|                  | ----------------> |   Wi-Fi client   | ----------------> |     Wi-Fi AP     |
|                  |        USB        |    UDP client    |     UDP/Wi-Fi     |    UDP Server    |
+------------------+                   +------------------+                   +------------------+
```


Requirements
------------

- [Python](https://www.python.org/downloads/) 3.9 or newer (tested through 3.14)
- [NumPy](https://numpy.org/) - array and image processing
- [Pillow](https://github.com/python-pillow/Pillow) 6.0.0 or newer - Python image library
- [FlatBuffers](https://github.com/google/flatbuffers) - serialization library
- [dpkt](https://github.com/kbandla/dpkt) - TCP/IP packet parsing library

Optional extras:

- `vision` - OpenCV for template matching, custom markers and ArUco detection
- `tts` - `pyttsx3` on platforms without a built-in speech engine
- `voice` - `SpeechRecognition` for microphone input
- `all` - all of the above


Installation
------------

Using pip:

```
pip install pyco-cozmo
pycozmo-resources download
```

With optional features:

```
pip install "pyco-cozmo[all]"
```

From source:

```
git clone https://github.com/Janosaus123/pyco-cozmo.git
cd pyco-cozmo
pip install .
pycozmo-resources download
```

From source, for development:

```
git clone https://github.com/Janosaus123/pyco-cozmo.git
cd pyco-cozmo
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev,all]"
pycozmo-resources download
```

The command-line tools are also installed as console scripts:
`pycozmo-app`, `pycozmo-anim`, `pycozmo-discover`, `pycozmo-dump`,
`pycozmo-replay`, `pycozmo-resources`, and `pycozmo-update`.

 
Support
-------

Bug reports and changes should be sent via GitHub:

[https://github.com/Janosaus123/pyco-cozmo](https://github.com/Janosaus123/pyco-cozmo)

Upstream project and original documentation:

[https://github.com/zayfod/pycozmo](https://github.com/zayfod/pycozmo)

DDL Robot Discord server, channel #development-cozmo:

[https://discord.gg/ew92haS](https://discord.gg/ew92haS)


Fork and attribution
--------------------

`pyco-cozmo` is an independent fork of [zayfod/pycozmo](https://github.com/zayfod/pycozmo).
The original PyCozmo protocol implementation, tools, and documentation were created by
Kaloyan Tenchov and the PyCozmo contributors. This fork adds the vision, navigation,
personality, voice, music, discovery, and Vector modules documented above.

The original project is available at [https://github.com/zayfod/pycozmo](https://github.com/zayfod/pycozmo)
and on PyPI as [`pycozmo`](https://pypi.org/project/pycozmo/).


Disclaimer
----------

This project is not affiliated with [Digital Dream Labs](https://www.digitaldreamlabs.com/) or
[Anki](https://anki.com/).
#   p y c o - c o z m o  
 