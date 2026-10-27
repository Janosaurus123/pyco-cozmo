
import unittest
from unittest import mock

from pycozmo.client import Client
from pycozmo.protocol_base import UnknownCommand


class ClientStartTest(unittest.TestCase):
    def test_start_registers_unknown_command_handler(self):
        client = Client()
        with mock.patch.object(client.conn, "start") as start:
            client.start()
            start.assert_called_once_with()
        self.assertIn(UnknownCommand, client.dispatch_handlers)
        self.assertTrue(client.dispatch_handlers[UnknownCommand])


if __name__ == "__main__":
    unittest.main()
