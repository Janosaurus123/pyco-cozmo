#!/usr/bin/env python

"""Discover Cozmo and Vector robots visible on Wi-Fi."""

import argparse
import json

from pycozmo import discovery, wifi


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print JSON.")
    parser.add_argument("--host", action="append", default=[],
                        help="Additional robot address to probe.")
    parser.add_argument("--interface", default=None, help="Wi-Fi interface.")
    parser.add_argument("--timeout", type=float, default=1.0)
    args = parser.parse_args()

    networks = []
    try:
        networks = wifi.find_cozmo_networks(interface=args.interface)
    except wifi.WifiError as exc:
        print("Wi-Fi scan failed: {}".format(exc))

    robots = discovery.discover_robots(
        hosts=args.host or None, interface=args.interface, timeout=args.timeout)

    if args.json:
        print(json.dumps({
            "networks": [network.to_dict() for network in networks],
            "robots": [robot.to_dict() for robot in robots],
        }, indent=2))
        return 0

    for network in networks:
        print("Network: {} ({}%)".format(network.ssid, network.signal))
    for robot in robots:
        print("Robot: {} ({}) at {}:{} reachable={}".format(
            robot.name, robot.kind, robot.address, robot.port, robot.reachable))
    if not networks and not robots:
        print("No Cozmo or Vector networks found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
