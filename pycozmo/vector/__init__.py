"""

Vector animation support.

This package reads Vector ``.bin`` animation files (FlatBuffers), converts
the common tracks to Cozmo animation clips, and plays clips through a
Cozmo-like client or user-supplied callbacks.  No Vector SDK is required.

"""

from .animation import (
    KEYFRAME_TYPES,
    VectorAnimClips,
    VectorAnimationError,
    VectorAnimations,
    VectorAnimator,
    VectorAudioEventGroup,
    VectorAudioParameter,
    VectorAudioState,
    VectorAudioSwitch,
    VectorBackpackLights,
    VectorBodyMotion,
    VectorClip,
    VectorEvent,
    VectorFaceAnimation,
    VectorFrame,
    VectorHeadAngle,
    VectorKeyframe,
    VectorLiftHeight,
    VectorProceduralFace,
    VectorRecordHeading,
    VectorRobotAudio,
    VectorSpriteBox,
    VectorTurnToRecordedHeading,
    cozmo_clips_to_vector,
    is_vector_animation,
    load_and_play,
    load_vector_animations,
)
from .behavior import (VectorBehavior, VectorBehaviorError,
                       VectorBehaviorSystem, load_vector_behaviors)


__all__ = [
    "KEYFRAME_TYPES",
    "VectorAnimClips",
    "VectorAnimationError",
    "VectorAnimations",
    "VectorAnimator",
    "VectorAudioEventGroup",
    "VectorAudioParameter",
    "VectorAudioState",
    "VectorAudioSwitch",
    "VectorBackpackLights",
    "VectorBodyMotion",
    "VectorClip",
    "VectorEvent",
    "VectorFaceAnimation",
    "VectorFrame",
    "VectorHeadAngle",
    "VectorKeyframe",
    "VectorLiftHeight",
    "VectorProceduralFace",
    "VectorRecordHeading",
    "VectorRobotAudio",
    "VectorSpriteBox",
    "VectorTurnToRecordedHeading",
    "VectorBehavior",
    "VectorBehaviorError",
    "VectorBehaviorSystem",
    "cozmo_clips_to_vector",
    "is_vector_animation",
    "load_and_play",
    "load_vector_animations",
    "load_vector_behaviors",
]
