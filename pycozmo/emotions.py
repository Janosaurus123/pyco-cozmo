"""

Emotion representation and reading.

"""

import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import logger
from .json_loader import get_json_files, load_json_file


__all__ = [
    "EmotionType",
    "EmotionEvent",

    "load_emotion_types",
    "load_emotion_events",
]


class Node:
    def __init__(self, x: float, y: float):
        self.x = float(x)
        self.y = float(y)


class DecayGraph:
    __slots__ = [
        "nodes_x",
        "nodes_y",
        "ext_line_params",
    ]

    def __init__(self, nodes: List[Node]) -> None:
        self.nodes_x = [node.x for node in nodes]
        self.nodes_y = [node.y for node in nodes]
        self.ext_line_params = self.get_line_parameters(nodes[-2], nodes[-1]) if len(nodes) > 1 else None

    def get_increment(self, val) -> float:
        if self.ext_line_params is None:
            f_out = self.nodes_y[0]
        elif val <= self.nodes_x[-1]:
            f_out = np.interp(val, self.nodes_x, self.nodes_y)
        else:
            f_out = self.ext_line_params[0] * val + self.ext_line_params[1]
        return f_out

    @staticmethod
    def get_line_parameters(p1: Node, p2: Node) -> Tuple[float]:
        try:
            m = (p1.y - p2.y) / (p1.x - p2.x)
            b = p1.y - m * p1.x
        except ZeroDivisionError:
            m, b = 0, p1.y
        return m, b


class EmotionType:
    """ A decaying emotion value.

    Args:
        name: Emotion name.
        decay_graph: Resource decay graph, kept for inspection and tuning.
        repetition_penaly: Resource repetition penalty graph.
        value: Initial intensity in the range 0..1.
        baseline: Value the emotion decays towards.
        decay_rate: Fraction of the distance to ``baseline`` removed per
            second.  The default removes about 5% per second.
    """

    __slots__ = [
        "name",
        "decay_graph",
        "repetition_penalty",
        "value",
        "baseline",
        "decay_rate",
        "last_update",
    ]

    def __init__(self,
                 name: str,
                 decay_graph: DecayGraph,
                 repetition_penaly: DecayGraph,
                 value: float = 0.0,
                 baseline: float = 0.0,
                 decay_rate: float = 0.05) -> None:
        self.name = str(name)
        self.decay_graph = decay_graph
        self.repetition_penalty = repetition_penaly
        self.value = max(0.0, min(1.0, float(value)))
        self.baseline = max(0.0, min(1.0, float(baseline)))
        self.decay_rate = max(0.0, float(decay_rate))
        self.last_update = time.monotonic()

    def _clamp(self) -> None:
        self.value = max(0.0, min(1.0, self.value))

    def add(self, amount: float) -> float:
        """ Add to the emotion intensity and return the new value. """
        self.value += float(amount)
        self._clamp()
        return self.value

    def set(self, value: float) -> float:
        """ Set the emotion intensity and return it. """
        self.value = float(value)
        self._clamp()
        return self.value

    def update(self, dt: Optional[float] = None) -> float:
        """ Decay towards the baseline and return the new value.

        With no ``dt`` argument the elapsed wall-clock time since the last
        update is used.  The decay is exponential, so repeated calls are
        stable at any frame rate.
        """
        now = time.monotonic()
        if dt is None:
            dt = now - self.last_update
        self.last_update = now
        dt = float(dt)
        if dt <= 0:
            return self.value
        factor = min(1.0, self.decay_rate * dt)
        self.value += (self.baseline - self.value) * factor
        self._clamp()
        return self.value


class EmotionEvent:
    """ EmotionEvent representation class. """

    __slots__ = [
        "name",
        "affectors",
    ]

    def __init__(self, name: str, affectors: Dict[str, float]) -> None:
        self.name = str(name)
        self.affectors = dict(affectors)

    @classmethod
    def from_json(cls, data: Dict):
        affectors = {}
        for affector in data['emotionAffectors']:
            affectors[affector['emotionType']] = affector['value']
        return cls(name=data['name'], affectors=affectors)


def load_emotion_types(resource_dir: str) -> Dict[str, EmotionType]:

    start_time = time.perf_counter()

    json_data = load_json_file(
        os.path.join(resource_dir, 'cozmo_resources', 'config', 'engine', 'mood_config.json'))

    decay_graphs = {}
    for graph in json_data['decayGraphs']:
        nodes = [Node(x=n['x'], y=n['y']) for n in graph['nodes']]
        decay_graphs[graph['emotionType']] = DecayGraph(nodes)

    # Note: the repetition penalty might be linked not only to emotion events but also any activities or behaviors.
    default_rp = DecayGraph([Node(x=n['x'], y=n['y']) for n in json_data['defaultRepetitionPenalty']['nodes']])

    emotion_types = {
        "WantToPlay": EmotionType("WantToPlay", decay_graphs.get('WantToPlay', decay_graphs['default']), default_rp),
        "Social": EmotionType("Social", decay_graphs.get('Social', decay_graphs['default']), default_rp),
        "Confident": EmotionType("Confident", decay_graphs.get('Confident', decay_graphs['default']), default_rp),
        "Excited": EmotionType("Excited", decay_graphs.get('Excited', decay_graphs['default']), default_rp),
        "Happy": EmotionType("Happy", decay_graphs.get('Happy', decay_graphs['default']), default_rp),
        "Calm": EmotionType("Calm", decay_graphs.get('Calm', decay_graphs['default']), default_rp),
        "Brave": EmotionType("Brave", decay_graphs.get('Brave', decay_graphs['default']), default_rp),
    }

    logger.debug("Loaded emotion types in {:.02f} s.".format(time.perf_counter() - start_time))

    return emotion_types


def load_emotion_events(resource_dir: str) -> Dict[str, EmotionEvent]:

    start_time = time.perf_counter()

    emotion_files = get_json_files(resource_dir,
                                   [os.path.join('cozmo_resources', 'config', 'engine', 'emotionevents/')])
    emotion_events = {}

    for ef in emotion_files:
        json_data = load_json_file(ef)
        if 'emotionEvents' not in json_data:
            emotion_events[json_data['name']] = EmotionEvent.from_json(json_data)
        else:
            for event in json_data['emotionEvents']:
                emotion_events[event['name']] = EmotionEvent.from_json(event)

    logger.debug("Loaded {} emotion events in {:.02f} s.".format(
        len(emotion_events), time.perf_counter() - start_time))

    return emotion_events
