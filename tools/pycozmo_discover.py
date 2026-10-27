#!/usr/bin/env python3
"""

Discover Cozmo and Vector robots on Wi-Fi.

Examples::

    pycozmo-discover
    pycozmo-discover --json
    pycozmo-discover --host 172.31.1.1 --timeout 2
    pycozmo-discover --wait 60

"""

import argparse
import json
import sys

from pycozmo import discovery, wifi


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Discover Cozmo and Vector robots on Wi-Fi.")
    parser.add_argument("--json", action="store_true",
                        help="Print machine-readable JSON.")
    parser.add_argument("--host", action="append", default=[],
                        help="Additional robot address to probe (repeatable).")
    parser.add_argument("--interface", default=None,
                        help="Wi-Fi interface name.")
    parser.add_argument("--timeout", type=float, default=1.0,
                        help="Per-host reachability timeout in seconds.")
    parser.add_argument("--wait", type=float, default=0.0,
                        help="Keep scanning for this many seconds.")
    parser.add_argument("--no-wifi-scan", action="store_true",
                        help="Only probe explicit hosts.")
    parser.add_argument("--networks", action="store_true",
                        help="Also print all visible Wi-Fi networks.")
    return parser


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)

    try:
        networks = wifi.scan_networks(interface=args.interface) if (
            args.networks or not args.no_wifi_scan) else []
    except wifi.WifiError as exc:
        if not args.host:
            print("Wi-Fi scan failed: {}".format(exc), file=sys.stderr)
            return 1
        networks = []

    robots = discovery.discover_robots(
        scan_wifi=not args.no_wifi_scan,
        hosts=args.host or None,
        timeout=args.timeout,
        interface=args.interface,
    )
    if args.wait > 0 and not robots:
        robot = discovery.wait_for_robot(timeout=args.wait, interface=args.interface,
                                         hosts=args.host or None)
        robots = [robot] if robot else []

    if args.json:
        payload = {
            "robots": [robot.to_dict() for robot in robots],
            "networks": [network.to_dict() for network in networks],
        }
        print(json.dumps(payload, indent=2))
        return 0

    if args.networks:
        print("Wi-Fi networks:")
        for network in networks:
            marker = " *" if network.is_cozmo else ""
            signal = "{}%".format(network.signal) if network.signal is not None else "?"
            print("  {:<32} {:>4}  {}{}".format(network.ssid, signal,
                                                network.security, marker))
        print()

    if robots:
        print("Robots:")
        for robot in robots:
            status = "reachable" if robot.reachable else "visible"
            print("  {} ({}) at {}:{} - {}{}".format(
                robot.name, robot.kind, robot.address, robot.port, status,
                " via {}".format(robot.ssid) if robot.ssid else ""))
    else:
        print("No robots found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
