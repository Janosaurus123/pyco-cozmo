Contributing to pyco-cozmo
==========================

Any contribution to `pyco-cozmo` is welcome and appreciated.

This project is a fork of [PyCozmo](https://github.com/zayfod/pycozmo). Please report
fork-specific bugs and feature requests in this repository, and report issues that also
affect the original project upstream.


Bug Reports and Feature Requests
--------------------------------

- Bug reports and feature requests should be made using [GitHub issues](https://github.com/Janosaus123/pyco-cozmo/issues).
- Include your operating system, Python version, and the output of `pip show pyco-cozmo` when relevant.
- For robot problems, include the Cozmo hardware version and firmware version if available.


Pull Requests
-------------

- Pull requests should be made against the `main` branch.
- All [tests](https://github.com/Janosaus123/pyco-cozmo/actions) should be passing before a PR is merged.
- Code should adhere to the [PEP 8 style guide](https://peps.python.org/pep-0008/).
- Add tests for new behavior and update the documentation when public APIs change.
- Run `python -m pytest -q` and `python -m flake8 pycozmo examples tools setup.py` before opening a PR.
