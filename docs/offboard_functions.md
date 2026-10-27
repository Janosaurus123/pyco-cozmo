
Cozmo Off-Board Functions
=========================

Cozmo mobile application resources consist of:
- audio files
- animations
- animation group descriptions
- behaviors
- reaction triggers
- emotions
- activities
- text-to-speech models

Robot firmware images are also distributed as part of the app resources.


Directory structure
-------------------

```
cozmo_resources/
    assets/
        animationGroupMaps/
        animationGroups/
        animations/
        cubeAnimationGroupMaps/
        faceAnimations/
        RewardedActions/
    config/
        engine/
            animations/
            behaviorSystem/
                activities/
                behaviors/
            emotionevents/
            firmware/
            lights/
                backpackLights/
                cubeLights/
    sound/
        English(US)
    tts/
```


Audio files
-----------

### WEM files
### BNK files

`pycozmo.audio` can play WAV files and raw PCM through Cozmo's speaker.
`pycozmo.songs` synthesizes simple melodies without external audio assets.


Animations
----------

Cozmo "animations" allow animating the following aspects of the robot:

- body movement
- lift movement
- head movement
- face images
- backpack LED animations
- audio

Cozmo animations are series of keyframes, stored in binary files in [FlatBuffers](https://google.github.io/flatbuffers/)
format. Animation data structures are declared in FlatBuffers format in
`files/cozmo/cozmo_resources/config/cozmo_anim.fbs` . The animation files are available in the following directory of
the Android mobile application:

`files/cozmo/cozmo_resources/assets/animations`

Face images are generated procedurally. They are described by 43 parameters - 5 for the face and 19 for each eye.
The face as a whole can be translated, scaled, and rotated. Each individual eye can be translated, scaled, and rotated.
The 4 corners of each eye can be controlled and each eye has a lower and upper lid.

The following presentation from Anki provides some background information on Cozmo animations:

[Cozmo: Animation pipeline for a physical robot](https://www.gdcvault.com/play/1024488/Cozmo-Animation-Pipeline-for-a)


Animation groups
----------------

Animation groups are sets of animations with the same purpose.  `pycozmo.anim`
reads the resource files and `Client.play_anim_group()` chooses a member.


Behaviors
---------

Behaviors can be thought of as small applications that perform a specific function using the robot client API.

`pycozmo.behavior` provides:

- `Behavior` - base class which posts `EvtBehaviorDone`.
- `BehaviorPlayAnim` - play a sequence of animation triggers.
- `BehaviorPlayArbitraryAnim` - play a random selection and count.
- `BehaviorReactToCliff` - stop motors and play `ReactToCliff`.
- `BehaviorDriveOffCharger` - back off the charger contacts, then drive an
  optional extra distance.
- `load_behaviors()` - load the behavior resource directory.
- `BEHAVIOR_CLASSES` - extension point for application-defined behavior classes.

Behaviors are event dispatchers.  `BehaviorPlayAnim` listens for
`EvtAnimationCompleted` and advances to the next trigger:

```python
from pycozmo import behavior

behaviors = behavior.load_behaviors(resource_dir, cli)
behaviors["PlayAnim_Example"].activate()
```


Reactions
---------

Reactions map robot events to behaviors.  `load_reaction_trigger_behavior_map()`
reads `reactionTrigger_behavior_map.json`; `ReactionTrigger` keeps the behavior
ID and the `shouldResumeLast` flag.

The `Brain` applies reactions automatically.  `Personality.react()` can also be
used directly when building a custom engine.


Emotions
--------

Emotions are modeled as value functions that change in one of the following ways:
- over time, driven by a decay function
- as a result of reactions
- as a result of behaviors

`pycozmo.personality` implements traits, drives, emotions, decay toward
baselines, event responses, and weighted activity/behavior selection:

```python
from pycozmo import personality

p = personality.Personality.default()
p.update(dt=30.0)
p.feel("happiness", 0.4)
p.satisfy("social", 0.5)
p.react("EvtFaceObserved")
p.save("personality.json")
```

Presets: `default`, `playful`, `social`, `curious`, `shy`, `calm`, `brave`, and
`tired`.


Activities
----------

Activities are sets of behaviors with a rule how to choose between them.

`pycozmo.activity` supports:

- `BehaviorChooser` strategies `Selection`, `StrictPriority`, and `Scoring`
  (weighted sampling without replacement plus repetition penalties).
- `Objective` for activity requirements.
- `BehaviorsActivity`, `VoiceCommandActivity`, `FeedingActivity`,
  `FreeplayActivity`, `SparkedActivity`, `PyramidActivity`,
  `SocializeActivity`, and `NeedsActivity`.
- `from_dict()` and `load_activities()` for the resource file shapes.

`Brain` uses the activity chooser and the personality scores to select the next
behavior.  `Client.set_activity()` is available for applications which manage
activities themselves.


Computer vision
---------------

`pycozmo.vision` works with NumPy arrays and Pillow images.  OpenCV is optional
and enables the more advanced detectors.

Available detectors:

- motion detection and tracking
- face detection, expression estimation, recognition and tracking
- pet detection (species and fur color)
- Cozmo/Vector robot detection (screen color)
- Power Cube face and marker detection
- platform/charger detection
- custom marker rendering and recognition
- ArUco marker recognition (`cv2.aruco`)
- camera calibration from chessboard images and homography helpers
- drivable-area and ground-contact estimation
- a composable `VisionPipeline`

```python
from pycozmo import vision

pipeline = vision.VisionPipeline(
    vision.VisionConfig(faces=True, cubes=True, platforms=True, ground=True))
result = pipeline.process(image)
for detection in result.detections:
    print(detection.kind_name, detection.bbox, detection.confidence)
result.annotate(image).save("annotated.png")
```

`Brain` can run the pipeline on camera frames and react to faces and objects.


Navigation
----------

`pycozmo.navigation` provides:

- `OccupancyGrid` - log-odds grid with world/cell conversion.
- `MapBuilder` - integrate camera detections and robot poses.
- `AStarPlanner`, `plan_path()`, `simplify_path()` - grid planning.
- `PathFollower` - pure-pursuit style drive commands with wheel speeds.

```python
from pycozmo import navigation

grid = navigation.OccupancyGrid(width=40, height=40, resolution=25.0,
                                origin=(0.0, 0.0))
grid.mark_occupied(300, 400)
path = navigation.plan_path(grid, start=(100, 100), goal=(800, 800),
                            allow_unknown=True)
follower = navigation.PathFollower()
command = follower.compute(pose, path)
left_speed, right_speed = command.wheel_speeds()
```


Text-to-speech
--------------

`pycozmo.tts` uses a platform backend (espeak, Windows SAPI, macOS `say`) or
`pyttsx3`, normalizes the result to Cozmo's 22 kHz mono format, and returns WAV
bytes:

```python
from pycozmo import tts

speech = tts.TextToSpeech()
speech.say(cli, "Hello from PyCozmo")
# or synthesize without a robot
wav = speech.synthesize("Hello")
```


Songs
-----

`pycozmo.songs` parses a tiny notation and synthesizes WAV data:

```python
from pycozmo import songs

song = songs.parse("C4/4 D4/4 E4/4 G4/4 C5/2")
songs.play(cli, song)
```

`NOTE/LENGTH` uses scientific pitch names; `R` is a rest and `~` ties notes.
Waveforms: `sine`, `square`, `triangle`, and `saw`.


Voice commands
--------------

`pycozmo.voice` includes a dependency-free command parser and executor plus an
optional microphone wrapper:

```python
from pycozmo.voice import VoiceController, parse_command

command = parse_command("turn left 90 degrees")
print(command.name, command.parameters)

controller = VoiceController(cli)  # requires SpeechRecognition
controller.run_forever()
```


Vector animations and behaviors
-------------------------------

`pycozmo.vector` reads Anki Vector `.bin` animation files without the Vector SDK.
It parses the FlatBuffers schema directly and exposes:

- all common Vector tracks (lift, head, body, lights, faces, audio, events,
  sprites, heading recording/turning)
- JSON conversion and round-tripping
- conversion of common tracks to `pycozmo.anim_encoder.AnimClip`
- `VectorAnimator` for timed frames and playback through a Cozmo-like client or
  handler callbacks
- `VectorBehavior` / `VectorBehaviorSystem` for priority-based, looping,
  cooldown-aware animation behaviors

```python
from pycozmo.vector import load_vector_animations, VectorAnimator

animations = load_vector_animations("animation.bin")
clip = animations["NameOfClip"]
VectorAnimator(clip).play(client=cli)
```

`is_vector_animation()` distinguishes Vector files from Cozmo animation files.


Wi-Fi and discovery
-------------------

`pycozmo.wifi` wraps the platform network tools:

- `scan_networks()` and `find_cozmo_networks()`
- `current_connection()`
- `connect()`, `disconnect()`, `wait_for_network()`

`pycozmo.discovery` combines Wi-Fi scanning with an optional ICMP check:

```python
from pycozmo import discovery

robot = discovery.find_robot()
if robot:
    print(robot.name, robot.kind, robot.address, robot.reachable)
```

The `pycozmo-discover` console script exposes the same functionality:

```
pycozmo-discover --networks
pycozmo-discover --json
pycozmo-discover --host 172.31.1.1 --wait 30
```


Optional dependencies
---------------------

The base package only needs NumPy, Pillow, FlatBuffers, and dpkt.  Install
extras as needed:

```
pip install "pyco-cozmo[vision]"   # OpenCV
pip install "pyco-cozmo[tts]"      # pyttsx3
pip install "pyco-cozmo[voice]"    # SpeechRecognition
pip install "pyco-cozmo[all]"
```
