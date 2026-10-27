"""Tests for Wi-Fi helpers and robot discovery using mocked OS tools."""

import subprocess
import unittest
from unittest import mock

from pycozmo import discovery, wifi


class TestWifiTypes(unittest.TestCase):

    def test_ssid_predicates(self):
        self.assertTrue(wifi.is_cozmo_ssid("Cozmo_ABC123"))
        self.assertTrue(wifi.is_cozmo_ssid("Vector-1a2b"))
        self.assertTrue(wifi.is_cozmo_ssid("cozmo 42"))
        self.assertFalse(wifi.is_cozmo_ssid("HomeWifi"))
        self.assertFalse(wifi.is_cozmo_ssid(""))
        self.assertTrue(wifi.is_robot_ssid("Cozmo_ABC", "Cozmo"))
        self.assertFalse(wifi.is_robot_ssid("Vector_ABC", "Cozmo"))
        self.assertFalse(wifi.is_robot_ssid("Cozmo", "Cozmo"))

    def test_network(self):
        network = wifi.WifiNetwork("Cozmo_ABC", signal=80, security="WPA2",
                                   bssid="aa:bb:cc:dd:ee:ff")
        self.assertTrue(network.is_cozmo)
        self.assertFalse(network.is_vector)
        self.assertEqual(network.to_dict()["ssid"], "Cozmo_ABC")
        self.assertEqual(network.to_dict()["signal"], 80)
        self.assertIn("Cozmo_ABC", repr(network))
        self.assertTrue(wifi.WifiNetwork("Vector_1").is_vector)

    def test_connection(self):
        self.assertFalse(wifi.WifiConnection().connected)
        self.assertTrue(wifi.WifiConnection("Cozmo_ABC").connected)
        self.assertTrue(wifi.WifiConnection("Cozmo_ABC", state="connected").connected)
        self.assertTrue(wifi.WifiConnection("Cozmo_ABC", state="Completed").connected)
        self.assertFalse(wifi.WifiConnection("Cozmo_ABC", state="disconnected").connected)
        self.assertEqual(wifi.WifiConnection("Cozmo_ABC", 90, "connected", "wlan0")
                         .to_dict()["connected"], True)

    def test_parse_percent(self):
        self.assertEqual(wifi._parse_percent("85%"), 85)
        self.assertEqual(wifi._parse_percent("Signal: 42"), 42)
        self.assertIsNone(wifi._parse_percent("unknown"))


class TestWifiScanning(unittest.TestCase):

    def test_scan_windows(self):
        output = """
    Interface name : Wi-Fi
    There are 2 networks currently visible.

    SSID 1 : Cozmo_ABC123
        Network type            : Infrastructure
        Authentication          : WPA2-Personal
        Encryption              : CCMP
        BSSID 1                 : aa:bb:cc:dd:ee:01
             Signal             : 70%
        BSSID 2                 : aa:bb:cc:dd:ee:02
             Signal             : 90%

    SSID 2 : HomeWifi
        Authentication          : WPA2-Personal
        BSSID 1                 : 11:22:33:44:55:66
             Signal             : 55%
"""
        with mock.patch.object(wifi, "_run", return_value=output):
            networks = wifi._scan_windows()
        self.assertEqual(len(networks), 2)
        self.assertEqual(networks[0].ssid, "Cozmo_ABC123")
        self.assertEqual(networks[0].signal, 90)
        self.assertEqual(networks[0].security, "WPA2-Personal")
        self.assertEqual(networks[0].bssid, "aa:bb:cc:dd:ee:02")
        self.assertEqual(networks[1].ssid, "HomeWifi")
        self.assertEqual(networks[1].signal, 55)

    def test_scan_linux(self):
        output = ("Cozmo_ABC:AA\\:BB\\:CC\\:DD\\:EE\\:FF:80:WPA2\n"
                  "My\\:Network:11\\:22\\:33\\:44\\:55\\:66:55:WPA1\n"
                  "short:line\n")
        with mock.patch.object(wifi, "_run", return_value=output) as run:
            networks = wifi._scan_linux("wlan0")
        run.assert_called_once()
        self.assertIn("ifname", run.call_args[0][0])
        self.assertEqual(len(networks), 2)
        self.assertEqual(networks[0].ssid, "Cozmo_ABC")
        self.assertEqual(networks[0].bssid, "AA:BB:CC:DD:EE:FF")
        self.assertEqual(networks[0].signal, 80)
        self.assertEqual(networks[0].security, "WPA2")
        self.assertEqual(networks[1].ssid, "My:Network")
        self.assertEqual(networks[1].signal, 55)

    def test_scan_macos(self):
        output = """                            SSID BSSID             RSSI CHANNEL HT CC SECURITY (auth/unicast/group)
                         Cozmo ABC aa:bb:cc:dd:ee:ff -50  6       Y  US WPA2(PSK/AES/AES)
                         HomeWifi 11:22:33:44:55:66 -70  11      Y  US WPA2(PSK/AES/AES)
"""
        with mock.patch.object(wifi.os.path, "exists", return_value=True), \
                mock.patch.object(wifi, "_run", return_value=output):
            networks = wifi._scan_macos()
        self.assertEqual(len(networks), 2)
        self.assertEqual(networks[0].ssid, "Cozmo ABC")
        self.assertEqual(networks[0].bssid, "aa:bb:cc:dd:ee:ff")
        self.assertEqual(networks[1].ssid, "HomeWifi")

    def test_scan_macos_missing_airport(self):
        with mock.patch.object(wifi.os.path, "exists", return_value=False):
            with self.assertRaises(wifi.WifiError):
                wifi._scan_macos()

    def test_scan_dispatch(self):
        with mock.patch.object(wifi.platform, "system", return_value="Windows"), \
                mock.patch.object(wifi, "_scan_windows", return_value=["win"]) as scan:
            self.assertEqual(wifi.scan_networks("iface"), ["win"])
            scan.assert_called_once_with("iface")
        with mock.patch.object(wifi.platform, "system", return_value="Linux"), \
                mock.patch.object(wifi, "_scan_linux", return_value=["linux"]) as scan:
            self.assertEqual(wifi.scan_networks(), ["linux"])
            scan.assert_called_once_with(None)
        with mock.patch.object(wifi.platform, "system", return_value="Darwin"), \
                mock.patch.object(wifi, "_scan_macos", return_value=["mac"]) as scan:
            self.assertEqual(wifi.scan_networks(), ["mac"])
            scan.assert_called_once_with(None)
        with mock.patch.object(wifi.platform, "system", return_value="Plan9"):
            with self.assertRaises(wifi.WifiError):
                wifi.scan_networks()

    def test_find_cozmo_networks(self):
        networks = [
            wifi.WifiNetwork("Home", signal=99),
            wifi.WifiNetwork("Cozmo_ABC", signal=40),
            wifi.WifiNetwork("Vector_XYZ", signal=80),
        ]
        found = wifi.find_cozmo_networks(networks)
        self.assertEqual([network.ssid for network in found], ["Vector_XYZ", "Cozmo_ABC"])
        with mock.patch.object(wifi, "scan_networks", return_value=networks) as scan:
            self.assertEqual(len(wifi.find_cozmo_networks(interface="wlan0")), 2)
            scan.assert_called_once_with(interface="wlan0")


class TestWifiCurrentConnection(unittest.TestCase):

    def test_windows(self):
        output = """
    Name                   : Wi-Fi
    State                  : connected
    SSID                   : Cozmo_ABC
    BSSID                  : aa:bb:cc:dd:ee:ff
    Signal                 : 88%
"""
        with mock.patch.object(wifi, "_run", return_value=output):
            connection = wifi._current_windows()
        self.assertEqual(connection.interface, "Wi-Fi")
        self.assertEqual(connection.state, "connected")
        self.assertEqual(connection.ssid, "Cozmo_ABC")
        self.assertEqual(connection.signal, 88)
        self.assertTrue(connection.connected)

    def test_linux(self):
        output = "no:Home:70:disconnected\nyes:Cozmo\\:ABC:90:connected\n"
        with mock.patch.object(wifi, "_run", return_value=output) as run:
            connection = wifi._current_linux("wlan0")
        run.assert_called_once()
        self.assertEqual(connection.ssid, "Cozmo:ABC")
        self.assertEqual(connection.signal, 90)
        self.assertEqual(connection.state, "connected")
        self.assertEqual(connection.interface, "wlan0")
        self.assertTrue(connection.connected)

    def test_macos(self):
        output = "     agrCtlRSSI: -55\n     state: running\n     SSID: Cozmo_ABC\n"
        with mock.patch.object(wifi.os.path, "exists", return_value=True), \
                mock.patch.object(wifi, "_run", return_value=output):
            connection = wifi._current_macos()
        self.assertEqual(connection.ssid, "Cozmo_ABC")
        self.assertEqual(connection.state, "running")
        self.assertEqual(connection.signal, 90)

    def test_dispatch(self):
        with mock.patch.object(wifi.platform, "system", return_value="Windows"), \
                mock.patch.object(wifi, "_current_windows", return_value="win") as current:
            self.assertEqual(wifi.current_connection("iface"), "win")
            current.assert_called_once_with("iface")
        with mock.patch.object(wifi.platform, "system", return_value="Plan9"):
            with self.assertRaises(wifi.WifiError):
                wifi.current_connection()


class TestWifiCommands(unittest.TestCase):

    def test_run_success_and_errors(self):
        completed = mock.Mock(returncode=0, stdout="ok", stderr="")
        with mock.patch.object(wifi.subprocess, "run", return_value=completed):
            self.assertEqual(wifi._run(["tool"]), "ok")
        failed = mock.Mock(returncode=1, stdout="", stderr="bad")
        with mock.patch.object(wifi.subprocess, "run", return_value=failed):
            with self.assertRaises(wifi.WifiError) as raised:
                wifi._run(["tool"])
        self.assertIn("bad", str(raised.exception))
        with mock.patch.object(wifi.subprocess, "run", side_effect=FileNotFoundError):
            with self.assertRaises(wifi.WifiError):
                wifi._run(["missing"])
        with mock.patch.object(wifi.subprocess, "run",
                               side_effect=subprocess.TimeoutExpired("tool", 1)):
            with self.assertRaises(wifi.WifiError):
                wifi._run(["tool"])

    def test_connect_windows_profile(self):
        calls = []
        profiles = []

        def run(command, timeout=15.0):
            calls.append(command)
            if "add" in command:
                filename = command[-2].split("=", 1)[1]
                with open(filename, "r", encoding="utf-8") as stream:
                    profiles.append(stream.read())
            return ""

        with mock.patch.object(wifi, "_run", side_effect=run):
            wifi._connect_windows("Cozmo & Co", "p<ss", "Wi-Fi")
        self.assertEqual(len(profiles), 1)
        self.assertIn("Cozmo &amp; Co", profiles[0])
        self.assertIn("p&lt;ss", profiles[0])
        self.assertEqual(calls[-1], ["netsh", "wlan", "connect",
                                     "name=Cozmo & Co", "ssid=Cozmo & Co",
                                     "interface=Wi-Fi"])
        with mock.patch.object(wifi, "_run") as run:
            wifi._connect_windows("OpenNet", None, None)
        self.assertEqual(run.call_args[0][0], ["netsh", "wlan", "connect",
                                               "name=OpenNet", "ssid=OpenNet"])

    def test_connect_linux_and_macos(self):
        with mock.patch.object(wifi, "_run") as run:
            wifi._connect_linux("Cozmo", "secret", "wlan0")
        self.assertEqual(run.call_args[0][0],
                         ["nmcli", "device", "wifi", "connect", "Cozmo",
                          "password", "secret", "ifname", "wlan0"])
        with mock.patch.object(wifi, "_run") as run:
            wifi._connect_macos("Cozmo", "secret", None)
        self.assertEqual(run.call_args[0][0],
                         ["networksetup", "-setairportnetwork", "en0", "Cozmo", "secret"])

    def test_connect_success_and_timeout(self):
        connection = wifi.WifiConnection("Cozmo", state="connected")
        with mock.patch.object(wifi.platform, "system", return_value="Linux"), \
                mock.patch.object(wifi, "_connect_linux") as connect_linux, \
                mock.patch.object(wifi, "current_connection", return_value=connection):
            result = wifi.connect("Cozmo", "pw", timeout=0)
        self.assertIs(result, connection)
        connect_linux.assert_called_once_with("Cozmo", "pw", None)

        with mock.patch.object(wifi.platform, "system", return_value="Linux"), \
                mock.patch.object(wifi, "_connect_linux"), \
                mock.patch.object(wifi, "current_connection",
                                  return_value=wifi.WifiConnection("Other", state="connected")), \
                mock.patch.object(wifi.time, "monotonic", side_effect=[0.0, 1.0]):
            with self.assertRaises(wifi.WifiError):
                wifi.connect("Cozmo", timeout=0)
        with mock.patch.object(wifi.platform, "system", return_value="Plan9"):
            with self.assertRaises(wifi.WifiError):
                wifi.connect("Cozmo")

    def test_disconnect(self):
        with mock.patch.object(wifi.platform, "system", return_value="Windows"), \
                mock.patch.object(wifi, "_run") as run:
            wifi.disconnect("Wi-Fi")
        self.assertEqual(run.call_args[0][0],
                         ["netsh", "wlan", "disconnect", "interface=Wi-Fi"])
        with mock.patch.object(wifi.platform, "system", return_value="Linux"), \
                mock.patch.object(wifi, "_run") as run:
            wifi.disconnect("wlan0")
        self.assertEqual(run.call_args[0][0], ["nmcli", "device", "disconnect", "wlan0"])
        with mock.patch.object(wifi.platform, "system", return_value="Darwin"), \
                mock.patch.object(wifi, "_run") as run:
            wifi.disconnect()
        self.assertEqual(run.call_args_list[0][0][0],
                         ["networksetup", "-setairportpower", "en0", "off"])
        self.assertEqual(run.call_args_list[1][0][0],
                         ["networksetup", "-setairportpower", "en0", "on"])
        with mock.patch.object(wifi.platform, "system", return_value="Plan9"):
            with self.assertRaises(wifi.WifiError):
                wifi.disconnect()

    def test_wait_for_network(self):
        network = wifi.WifiNetwork("Cozmo_ABC", signal=50)
        with mock.patch.object(wifi, "scan_networks", return_value=[network]) as scan:
            self.assertIs(wifi.wait_for_network("Cozmo", timeout=0), network)
            scan.assert_called_once_with(interface=None)
        with mock.patch.object(wifi, "scan_networks", return_value=[]), \
                mock.patch.object(wifi.time, "monotonic", side_effect=[0.0, 1.0]):
            self.assertIsNone(wifi.wait_for_network(timeout=0))
        with mock.patch.object(wifi, "scan_networks",
                               return_value=[wifi.WifiNetwork("Other")]), \
                mock.patch.object(wifi.time, "monotonic", side_effect=[0.0, 1.0]):
            self.assertIsNone(wifi.wait_for_network(timeout=0))


class TestDiscovery(unittest.TestCase):

    def test_robot_info(self):
        robot = discovery.RobotInfo("Cozmo_ABC", "cozmo", "172.31.1.1", 5551,
                                    "Cozmo_ABC", 80, True)
        self.assertEqual(robot.to_dict()["reachable"], True)
        self.assertIn("Cozmo_ABC", repr(robot))
        self.assertEqual(discovery.DEFAULT_ROBOT_ADDRESS, "172.31.1.1")
        self.assertEqual(discovery.DEFAULT_ROBOT_PORT, 5551)

    def test_ssid_helpers(self):
        self.assertEqual(discovery._kind_from_ssid("Vector_ABC"), "vector")
        self.assertEqual(discovery._kind_from_ssid("Cozmo_ABC"), "cozmo")
        self.assertEqual(discovery._name_from_ssid("Cozmo_ABC"), "Cozmo_ABC")
        self.assertEqual(discovery._name_from_ssid("Vector-1a2b"), "Vector_1a2b")
        self.assertEqual(discovery._name_from_ssid("Other"), "Other")

    def test_is_reachable(self):
        completed = mock.Mock(returncode=0)
        with mock.patch.object(discovery.platform, "system", return_value="Windows"), \
                mock.patch.object(discovery.subprocess, "run",
                                  return_value=completed) as run:
            self.assertTrue(discovery.is_reachable("1.2.3.4", timeout=0.5))
        self.assertEqual(run.call_args[0][0],
                         ["ping", "-n", "1", "-w", "500", "1.2.3.4"])
        completed.returncode = 1
        with mock.patch.object(discovery.platform, "system", return_value="Linux"), \
                mock.patch.object(discovery.subprocess, "run",
                                  return_value=completed) as run:
            self.assertFalse(discovery.is_reachable("1.2.3.4", timeout=2))
        self.assertEqual(run.call_args[0][0],
                         ["ping", "-c", "1", "-W", "2", "1.2.3.4"])
        with mock.patch.object(discovery.subprocess, "run", side_effect=FileNotFoundError):
            self.assertFalse(discovery.is_reachable())
        with mock.patch.object(discovery.subprocess, "run",
                               side_effect=subprocess.TimeoutExpired("ping", 1)):
            self.assertFalse(discovery.is_reachable())

    def test_discover_explicit_hosts(self):
        with mock.patch.object(discovery, "is_reachable", return_value=True) as ping:
            robots = discovery.discover_robots(scan_wifi=False, hosts=["1.1.1.1", "2.2.2.2"],
                                               timeout=0.2)
        self.assertEqual([robot.address for robot in robots], ["1.1.1.1", "2.2.2.2"])
        self.assertTrue(all(robot.reachable for robot in robots))
        self.assertEqual(ping.call_count, 2)
        with mock.patch.object(discovery, "is_reachable") as ping:
            robots = discovery.discover_robots(scan_wifi=False, hosts=["1.1.1.1"],
                                               check_reachable=False)
        self.assertTrue(robots[0].reachable)
        ping.assert_not_called()

    def test_discover_scans_wifi(self):
        networks = [
            wifi.WifiNetwork("Cozmo_ABC", signal=80),
            wifi.WifiNetwork("Vector_XYZ", signal=60),
        ]
        connection = wifi.WifiConnection("Cozmo_ABC", state="connected")
        with mock.patch.object(wifi, "find_cozmo_networks", return_value=networks), \
                mock.patch.object(wifi, "current_connection", return_value=connection), \
                mock.patch.object(discovery, "is_reachable", return_value=True) as ping:
            robots = discovery.discover_robots(timeout=0.5)
        self.assertEqual([robot.ssid for robot in robots], ["Cozmo_ABC", "Vector_XYZ"])
        self.assertEqual(robots[0].kind, "cozmo")
        self.assertEqual(robots[1].kind, "vector")
        self.assertTrue(robots[0].reachable)
        self.assertFalse(robots[1].reachable)
        self.assertEqual(ping.call_count, 1)

    def test_discover_tolerates_wifi_errors(self):
        with mock.patch.object(wifi, "find_cozmo_networks",
                               side_effect=wifi.WifiError("no wifi")), \
                mock.patch.object(wifi, "current_connection",
                                  side_effect=wifi.WifiError("no wifi")):
            self.assertEqual(discovery.discover_robots(), [])
        with mock.patch.object(wifi, "find_cozmo_networks", return_value=[]), \
                mock.patch.object(wifi, "current_connection", return_value=None):
            self.assertEqual(discovery.discover_robots(), [])

    def test_discover_deduplicates(self):
        with mock.patch.object(discovery, "is_reachable", return_value=True):
            robots = discovery.discover_robots(scan_wifi=False,
                                               hosts=["1.1.1.1", "1.1.1.1"])
        self.assertEqual(len(robots), 1)

    def test_find_robot(self):
        unreachable = discovery.RobotInfo("A", reachable=False)
        reachable = discovery.RobotInfo("B", reachable=True)
        with mock.patch.object(discovery, "discover_robots",
                               return_value=[unreachable, reachable]):
            self.assertIs(discovery.find_robot(), reachable)
        with mock.patch.object(discovery, "discover_robots", return_value=[unreachable]):
            self.assertIs(discovery.find_robot(), unreachable)
        with mock.patch.object(discovery, "discover_robots", return_value=[]):
            self.assertIsNone(discovery.find_robot())

    def test_wait_for_robot(self):
        robot = discovery.RobotInfo("A", reachable=True)
        with mock.patch.object(discovery, "find_robot", return_value=robot):
            self.assertIs(discovery.wait_for_robot(timeout=0), robot)
        with mock.patch.object(discovery, "find_robot", return_value=None), \
                mock.patch.object(discovery.time, "monotonic", side_effect=[0.0, 1.0]):
            self.assertIsNone(discovery.wait_for_robot(timeout=0))
        with mock.patch.object(discovery, "find_robot", side_effect=[None, robot]), \
                mock.patch.object(discovery.time, "monotonic", side_effect=[0.0, 0.5]), \
                mock.patch.object(discovery.time, "sleep") as sleep:
            self.assertIs(discovery.wait_for_robot(timeout=5), robot)
        sleep.assert_called_once()


if __name__ == "__main__":
    unittest.main()
