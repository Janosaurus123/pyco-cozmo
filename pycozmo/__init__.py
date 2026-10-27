"""

PyCozmo - a pure-Python Cozmo robot communication library.

"""

import sys

if sys.version_info < (3, 9, 0):
    sys.exit("ERROR: PyCozmo requires Python 3.9.0 or newer.")

from .logger import *
from .run import *

from .frame import Frame
from .conn import ROBOT_ADDR
from .client import Client
from .robot import *
from .event import *

from . import exception
from . import util
from . import window
from . import protocol_base
from . import protocol_declaration
from . import protocol_generator
from . import protocol_encoder
from . import protocol_utils
from . import lights
from . import camera
from . import object
from . import filter
from . import anim
from . import anim_encoder
from . import image_encoder
from . import procedural_face
from . import activity
from . import behavior
from . import emotions
from . import brain
from . import audiokinetic
from . import expressions
from . import personality
from . import tts
from . import songs
from . import vision
from . import navigation
from . import vector
from . import voice
from . import wifi
from . import discovery

from .personality import Personality
from .discovery import RobotInfo
from .wifi import WifiNetwork


__version__ = "0.9.0"

__all__ = [
    "logger",
    "logger_protocol",
    "logger_robot",

    "Frame",
    "ROBOT_ADDR",
    "Client",

    "setup_basic_logging",
    "connect",

    "Personality",
    "RobotInfo",
    "WifiNetwork",
]

del sys
