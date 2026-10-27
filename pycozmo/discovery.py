"""

Robot discovery.

Cozmo and Vector expose a Wi-Fi access point and use a fixed gateway address
(``172.31.1.1``) for the protocol.  This module combines Wi-Fi scanning with a
small ICMP reachability check.  It is deliberately independent of the protocol
connection code so discovery can run before a :class:`~pycozmo.client.Client`
is created.

"""

from __future__ import annotations

from dataclasses import dataclass
import platform
import re
import subprocess
import time
from typing import Iterable, List, Optional, Sequence

from . import wifi


__all__ = [
    "RobotInfo",
    "DEFAULT_ROBOT_ADDRESS",
    "DEFAULT_ROBOT_PORT",
    "is_reachable",
    "discover_robots",
    "find_robot",
    "wait_for_robot",
]


DEFAULT_ROBOT_ADDRESS = "172.31.1.1"
DEFAULT_ROBOT_PORT = 5551


@dataclass
class RobotInfo:
    """ A discovered Cozmo or Vector robot. """

    name: str
    kind: str = "cozmo"
    address: str = DEFAULT_ROBOT_ADDRESS
    port: int = DEFAULT_ROBOT_PORT
    ssid: str = ""
    signal: Optional[int] = None
    reachable: bool = False

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "address": self.address,
            "port": self.port,
            "ssid": self.ssid,
            "signal": self.signal,
            "reachable": self.reachable,
        }

    def __repr__(self) -> str:
        return "RobotInfo(name={!r}, kind={!r}, address={!r}, reachable={})".format(
            self.name, self.kind, self.address, self.reachable)


def _kind_from_ssid(ssid: str) -> str:
    return "vector" if str(ssid).lower().startswith("vector") else "cozmo"


def _name_from_ssid(ssid: str) -> str:
    match = re.match(r"^(Cozmo|Vector)[-_ ]?(.+)$", str(ssid or ""), re.IGNORECASE)
    if match:
        return "{}_{}".format(match.group(1).capitalize(), match.group(2))
    return str(ssid or "robot")


def is_reachable(host: str = DEFAULT_ROBOT_ADDRESS, timeout: float = 1.0) -> bool:
    """ Return whether ``host`` answers ICMP echo requests.

    Some networks block ICMP; callers should treat ``False`` as "unknown"
    rather than definitive proof that a robot is absent.
    """
    timeout = max(0.1, float(timeout))
    system = platform.system().lower()
    if system == "windows":
        command = ["ping", "-n", "1", "-w", str(int(timeout * 1000)), host]
    else:
        command = ["ping", "-c", "1", "-W", str(max(1, int(round(timeout)))), host]
    try:
        completed = subprocess.run(command, capture_output=True, text=True,
                                   timeout=timeout + 2.0, check=False)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0


def discover_robots(scan_wifi: bool = True,
                    hosts: Optional[Iterable[str]] = None,
                    timeout: float = 1.0,
                    interface: Optional[str] = None,
                    check_reachable: bool = True) -> List[RobotInfo]:
    """ Discover robots visible on Wi-Fi or at explicit ``hosts``.

    Args:
        scan_wifi: Scan for Cozmo/Vector access points.
        hosts: Extra addresses to probe, for example a manually configured IP.
        timeout: Per-host ICMP timeout.
        interface: Optional Wi-Fi interface for scanning.
        check_reachable: When false, report access points without pinging.
    """
    robots: List[RobotInfo] = []
    seen = set()

    def add(robot: RobotInfo) -> None:
        key = (robot.address, robot.ssid)
        if key not in seen:
            seen.add(key)
            robots.append(robot)

    if hosts:
        for host in hosts:
            host = str(host)
            reachable = is_reachable(host, timeout) if check_reachable else True
            add(RobotInfo(name=host, address=host, reachable=reachable))

    if scan_wifi:
        try:
            networks = wifi.find_cozmo_networks(interface=interface)
        except wifi.WifiError:
            networks = []
        connection = None
        try:
            connection = wifi.current_connection(interface=interface)
        except wifi.WifiError:
            pass
        for network in networks:
            connected = bool(connection and connection.ssid == network.ssid)
            reachable = is_reachable(DEFAULT_ROBOT_ADDRESS, timeout) if (
                check_reachable and connected) else False
            add(RobotInfo(
                name=_name_from_ssid(network.ssid),
                kind=_kind_from_ssid(network.ssid),
                address=DEFAULT_ROBOT_ADDRESS,
                ssid=network.ssid,
                signal=network.signal,
                reachable=reachable,
            ))
    return robots


def find_robot(scan_wifi: bool = True,
               hosts: Optional[Iterable[str]] = None,
               timeout: float = 1.0,
               interface: Optional[str] = None) -> Optional[RobotInfo]:
    """ Return the first reachable robot, or the first visible one. """
    robots = discover_robots(scan_wifi=scan_wifi, hosts=hosts, timeout=timeout,
                             interface=interface)
    for robot in robots:
        if robot.reachable:
            return robot
    return robots[0] if robots else None


def wait_for_robot(timeout: float = 60.0,
                   interval: float = 2.0,
                   interface: Optional[str] = None,
                   hosts: Optional[Sequence[str]] = None) -> Optional[RobotInfo]:
    """ Poll until a robot is discovered or ``timeout`` expires. """
    deadline = time.monotonic() + max(0.0, timeout)
    while True:
        robot = find_robot(interface=interface, hosts=hosts)
        if robot is not None:
            return robot
        if time.monotonic() >= deadline:
            return None
        time.sleep(max(0.1, interval))
