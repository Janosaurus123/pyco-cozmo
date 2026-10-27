
Overview
========

https://github.com/Janosaus123/pyco-cozmo

`PyCozmo` is a pure-Python communication library, alternative SDK, and application for the
[Cozmo robot](https://www.digitaldreamlabs.com/pages/cozmo). It allows controlling a Cozmo robot directly, without
having to go through a mobile device, running the Cozmo app.

This documentation covers the `pyco-cozmo` fork. The Python import name is `pycozmo`.

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

 
Support
-------

Bug reports and changes should be sent via GitHub:

[https://github.com/Janosaus123/pyco-cozmo](https://github.com/Janosaus123/pyco-cozmo)

Upstream project:

[https://github.com/zayfod/pycozmo](https://github.com/zayfod/pycozmo)

DDL Robot Discord server, channel #development-cozmo:

[https://discord.gg/ew92haS](https://discord.gg/ew92haS)


Disclaimer
----------

This project is not affiliated with [Digital Dream Labs](https://www.digitaldreamlabs.com/) or
[Anki](https://anki.com/).
