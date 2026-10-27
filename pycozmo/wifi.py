"""

Wi-Fi helpers for connecting to Cozmo and Vector robots.

Cozmo creates a Wi-Fi access point named ``Cozmo_XXXXXX`` (Vector uses a
similar ``Vector_XXXX`` name).  The robot's Wi-Fi password is displayed by
raising and lowering its lift; this module cannot read that password, but it
can scan for the network, connect using a password supplied by the caller,
and report the current connection.

The implementation shells out to the operating system's network tools, so it
has no extra Python dependencies.  Commands are best-effort and raise
:class:`WifiError` with the tool output when they fail.

"""

from __future__ import annotations

from dataclasses import dataclass
import os
import platform
import re
import subprocess
import tempfile
import time
from typing import Iterable, List, Optional, Sequence
from xml.sax.saxutils import escape as _xml_escape


__all__ = [
    "WifiNetwork",
    "WifiConnection",
    "WifiError",
    "is_cozmo_ssid",
    "is_robot_ssid",
    "scan_networks",
    "find_cozmo_networks",
    "current_connection",
    "connect",
    "disconnect",
    "wait_for_network",
]


class WifiError(RuntimeError):
    """ Raised when a Wi-Fi operation fails. """


@dataclass
class WifiNetwork:
    """ One Wi-Fi network visible to the operating system. """

    ssid: str
    signal: Optional[int] = None
    security: str = ""
    bssid: str = ""

    @property
    def is_cozmo(self) -> bool:
        return is_cozmo_ssid(self.ssid)

    @property
    def is_vector(self) -> bool:
        return is_robot_ssid(self.ssid, "Vector")

    def to_dict(self) -> dict:
        return {
            "ssid": self.ssid,
            "signal": self.signal,
            "security": self.security,
            "bssid": self.bssid,
            "is_cozmo": self.is_cozmo,
            "is_vector": self.is_vector,
        }

    def __repr__(self) -> str:
        return "WifiNetwork(ssid={!r}, signal={!r})".format(self.ssid, self.signal)


@dataclass
class WifiConnection:
    """ The active Wi-Fi connection. """

    ssid: str = ""
    signal: Optional[int] = None
    state: str = ""
    interface: str = ""

    @property
    def connected(self) -> bool:
        return bool(self.ssid) and self.state.lower() in ("connected", "completed", "")

    def to_dict(self) -> dict:
        return {
            "ssid": self.ssid,
            "signal": self.signal,
            "state": self.state,
            "interface": self.interface,
            "connected": self.connected,
        }


_COZMO_SSID = re.compile(r"^(?:cozmo|vector)[-_ ]?[0-9a-z]+$", re.IGNORECASE)


def is_robot_ssid(ssid: str, prefix: str = "Cozmo") -> bool:
    """ Return whether ``ssid`` looks like a robot access point. """
    pattern = re.compile(r"^{}[-_ ]?[0-9a-z]+$".format(re.escape(prefix)), re.IGNORECASE)
    return bool(pattern.match(str(ssid or "").strip()))


def is_cozmo_ssid(ssid: str) -> bool:
    """ Return whether ``ssid`` is a Cozmo or Vector access point. """
    return bool(_COZMO_SSID.match(str(ssid or "").strip()))


def _run(command: Sequence[str], timeout: float = 15.0) -> str:
    try:
        completed = subprocess.run(
            list(command), capture_output=True, text=True, timeout=timeout,
            check=False)
    except FileNotFoundError as exc:
        raise WifiError("Network tool '{}' was not found.".format(command[0])) from exc
    except subprocess.TimeoutExpired as exc:
        raise WifiError("Network tool '{}' timed out.".format(command[0])) from exc
    if completed.returncode != 0:
        message = (completed.stderr or completed.stdout or "").strip()
        raise WifiError("'{}' failed: {}".format(" ".join(command), message))
    return completed.stdout


def _parse_percent(value: str) -> Optional[int]:
    match = re.search(r"(\d+)", value or "")
    return int(match.group(1)) if match else None


def _unescape_nmcli(value: str) -> str:
    return value.replace("\\:", ":").replace("\\\\", "\\")


def _scan_windows(interface: Optional[str] = None) -> List[WifiNetwork]:
    del interface
    output = _run(["netsh", "wlan", "show", "networks", "mode=bssid"])
    networks: List[WifiNetwork] = []
    current: Optional[WifiNetwork] = None
    for raw_line in output.splitlines():
        line = raw_line.strip()
        match = re.match(r"SSID\s+\d+\s*:\s*(.*)$", line)
        if match:
            current = WifiNetwork(ssid=match.group(1).strip())
            networks.append(current)
            continue
        if current is None:
            continue
        match = re.match(r"BSSID\s+\d+\s*:\s*(\S+)", line, re.IGNORECASE)
        if match:
            current.bssid = match.group(1)
            continue
        match = re.match(r"Signal\s*:\s*(.*)$", line, re.IGNORECASE)
        if match:
            signal = _parse_percent(match.group(1))
            if signal is not None:
                current.signal = signal if current.signal is None else max(current.signal, signal)
            continue
        match = re.match(r"Authentication\s*:\s*(.*)$", line, re.IGNORECASE)
        if match:
            current.security = match.group(1).strip()
    return networks


def _scan_linux(interface: Optional[str] = None) -> List[WifiNetwork]:
    command = ["nmcli", "-t", "-f", "SSID,BSSID,SIGNAL,SECURITY", "device", "wifi", "list"]
    if interface:
        command.extend(["ifname", interface])
    output = _run(command)
    networks = []
    for line in output.splitlines():
        parts = line.split(":")
        if len(parts) < 4:
            continue
        # nmcli escapes colons inside SSIDs and BSSIDs, so the BSSID is the
        # six split parts immediately before signal and security.
        if len(parts) >= 8:
            security = _unescape_nmcli(parts[-1])
            signal = _parse_percent(parts[-2])
            bssid = _unescape_nmcli(":".join(parts[-8:-2]))
            ssid = _unescape_nmcli(":".join(parts[:-8]))
        else:
            security = _unescape_nmcli(parts[-1])
            signal = _parse_percent(parts[-2])
            bssid = ""
            ssid = _unescape_nmcli(":".join(parts[:-2]))
        networks.append(WifiNetwork(ssid=ssid.strip(), signal=signal,
                                    security=security, bssid=bssid))
    return networks


def _scan_macos(interface: Optional[str] = None) -> List[WifiNetwork]:
    airport = ("/System/Library/PrivateFrameworks/Apple80211.framework/"
               "Versions/Current/Resources/airport")
    if not os.path.exists(airport):
        raise WifiError("The macOS 'airport' tool is not available.")
    command = [airport, "-s"]
    del interface
    output = _run(command)
    networks = []
    for line in output.splitlines()[1:]:
        line = line.strip()
        if not line:
            continue
        match = re.match(r"(?P<ssid>.+?)\s+(?P<bssid>[0-9a-f:]{17})\s+"
                         r"(?P<rssi>-?\d+)\s+.*?(?P<channel>\d+)\s+.*$", line, re.IGNORECASE)
        if match:
            networks.append(WifiNetwork(ssid=match.group("ssid").strip(),
                                        signal=None, bssid=match.group("bssid")))
            continue
        parts = line.split()
        if len(parts) >= 3:
            networks.append(WifiNetwork(ssid=" ".join(parts[:-2]).strip()))
    return networks


def scan_networks(interface: Optional[str] = None) -> List[WifiNetwork]:
    """ Return visible Wi-Fi networks using the platform's network tool. """
    system = platform.system().lower()
    if system == "windows":
        return _scan_windows(interface)
    if system == "linux":
        return _scan_linux(interface)
    if system == "darwin":
        return _scan_macos(interface)
    raise WifiError("Unsupported platform: {}".format(platform.system()))


def find_cozmo_networks(networks: Optional[Iterable[WifiNetwork]] = None,
                        interface: Optional[str] = None) -> List[WifiNetwork]:
    """ Return Cozmo/Vector networks, strongest first. """
    if networks is None:
        networks = scan_networks(interface=interface)
    found = [network for network in networks if is_cozmo_ssid(network.ssid)]
    found.sort(key=lambda network: network.signal or 0, reverse=True)
    return found


def _current_windows(interface: Optional[str] = None) -> WifiConnection:
    del interface
    output = _run(["netsh", "wlan", "show", "interfaces"])
    connection = WifiConnection()
    for raw_line in output.splitlines():
        line = raw_line.strip()
        match = re.match(r"Name\s*:\s*(.*)$", line, re.IGNORECASE)
        if match:
            connection.interface = match.group(1).strip()
        match = re.match(r"State\s*:\s*(.*)$", line, re.IGNORECASE)
        if match:
            connection.state = match.group(1).strip()
        match = re.match(r"SSID\s*:\s*(.*)$", line, re.IGNORECASE)
        if match:
            connection.ssid = match.group(1).strip()
        match = re.match(r"Signal\s*:\s*(.*)$", line, re.IGNORECASE)
        if match:
            connection.signal = _parse_percent(match.group(1))
    return connection


def _current_linux(interface: Optional[str] = None) -> WifiConnection:
    command = ["nmcli", "-t", "-f", "ACTIVE,SSID,SIGNAL,STATE", "device", "wifi"]
    if interface:
        command.extend(["ifname", interface])
    output = _run(command)
    connection = WifiConnection(interface=interface or "")
    for line in output.splitlines():
        parts = line.split(":")
        if len(parts) >= 4 and parts[0] == "yes":
            connection.ssid = _unescape_nmcli(":".join(parts[1:-2]))
            connection.signal = _parse_percent(parts[-2])
            connection.state = parts[-1]
            break
    return connection


def _current_macos(interface: Optional[str] = None) -> WifiConnection:
    airport = ("/System/Library/PrivateFrameworks/Apple80211.framework/"
               "Versions/Current/Resources/airport")
    if not os.path.exists(airport):
        raise WifiError("The macOS 'airport' tool is not available.")
    del interface
    output = _run([airport, "-I"])
    connection = WifiConnection()
    for raw_line in output.splitlines():
        key, _, value = raw_line.partition(":")
        key = key.strip().lower()
        value = value.strip()
        if key == "ssid":
            connection.ssid = value
        elif key == "state":
            connection.state = value
        elif key == "agrctlrssi":
            try:
                connection.signal = max(0, min(100, int((int(value) + 100) * 2)))
            except ValueError:
                pass
    return connection


def current_connection(interface: Optional[str] = None) -> WifiConnection:
    """ Return the active Wi-Fi connection. """
    system = platform.system().lower()
    if system == "windows":
        return _current_windows(interface)
    if system == "linux":
        return _current_linux(interface)
    if system == "darwin":
        return _current_macos(interface)
    raise WifiError("Unsupported platform: {}".format(platform.system()))


def _connect_windows(ssid: str, password: Optional[str], interface: Optional[str]) -> None:
    if password:
        profile = """<?xml version="1.0"?>
<WLANProfile xmlns="http://www.microsoft.com/networking/WLAN/profile/v1">
  <name>{ssid}</name>
  <SSIDConfig><SSID><name>{ssid}</name></SSID></SSIDConfig>
  <connectionType>ESS</connectionType>
  <connectionMode>manual</connectionMode>
  <MSM><security><authEncryption>
    <authentication>WPA2PSK</authentication>
    <encryption>AES</encryption>
    <useOneX>false</useOneX>
  </authEncryption><sharedKey>
    <keyType>passPhrase</keyType>
    <protected>false</protected>
    <keyMaterial>{password}</keyMaterial>
  </sharedKey></security></MSM>
</WLANProfile>
""".format(ssid=_xml_escape(ssid), password=_xml_escape(password))
        filename = None
        try:
            with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False,
                                             encoding="utf-8") as stream:
                stream.write(profile)
                filename = stream.name
            _run(["netsh", "wlan", "add", "profile", "filename={}".format(filename),
                  "user=current"])
        finally:
            if filename:
                try:
                    os.unlink(filename)
                except OSError:
                    pass
    command = ["netsh", "wlan", "connect", "name={}".format(ssid), "ssid={}".format(ssid)]
    if interface:
        command.append("interface={}".format(interface))
    _run(command)


def _connect_linux(ssid: str, password: Optional[str], interface: Optional[str]) -> None:
    command = ["nmcli", "device", "wifi", "connect", ssid]
    if password:
        command.extend(["password", password])
    if interface:
        command.extend(["ifname", interface])
    _run(command)


def _connect_macos(ssid: str, password: Optional[str], interface: Optional[str]) -> None:
    interface = interface or "en0"
    command = ["networksetup", "-setairportnetwork", interface, ssid]
    if password:
        command.append(password)
    _run(command)


def connect(ssid: str, password: Optional[str] = None,
            interface: Optional[str] = None, timeout: float = 30.0) -> WifiConnection:
    """ Connect to ``ssid`` and wait until the connection is active.

    The robot's password is shown on its display; pass it explicitly.  If
    ``password`` is ``None``, an existing OS profile is used (Windows) or the
    network must be open.
    """
    system = platform.system().lower()
    if system == "windows":
        _connect_windows(ssid, password, interface)
    elif system == "linux":
        _connect_linux(ssid, password, interface)
    elif system == "darwin":
        _connect_macos(ssid, password, interface)
    else:
        raise WifiError("Unsupported platform: {}".format(platform.system()))

    deadline = time.monotonic() + max(0.0, timeout)
    while True:
        connection = current_connection(interface)
        if connection.ssid == ssid and connection.connected:
            return connection
        if time.monotonic() >= deadline:
            raise WifiError("Timed out connecting to '{}'.".format(ssid))
        time.sleep(0.5)


def disconnect(interface: Optional[str] = None) -> None:
    """ Disconnect from the current Wi-Fi network. """
    system = platform.system().lower()
    if system == "windows":
        command = ["netsh", "wlan", "disconnect"]
        if interface:
            command.append("interface={}".format(interface))
        _run(command)
    elif system == "linux":
        command = ["nmcli", "device", "disconnect"]
        if interface:
            command.append(interface)
        _run(command)
    elif system == "darwin":
        interface = interface or "en0"
        _run(["networksetup", "-setairportpower", interface, "off"])
        _run(["networksetup", "-setairportpower", interface, "on"])
    else:
        raise WifiError("Unsupported platform: {}".format(platform.system()))


def wait_for_network(prefix: str = "Cozmo", timeout: float = 60.0,
                     interval: float = 2.0,
                     interface: Optional[str] = None) -> Optional[WifiNetwork]:
    """ Wait until a network whose SSID starts with ``prefix`` is visible. """
    deadline = time.monotonic() + max(0.0, timeout)
    while True:
        for network in scan_networks(interface=interface):
            if network.ssid.lower().startswith(prefix.lower()):
                return network
        if time.monotonic() >= deadline:
            return None
        time.sleep(max(0.1, interval))
