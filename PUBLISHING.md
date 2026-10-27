# Publishing pyco-cozmo

This folder is a self-contained Git repository ready to push to GitHub and publish to
PyPI. The distribution name is **`pyco-cozmo`**; the Python import name remains
**`pycozmo`**.

`pco` and `pyco` are already taken on PyPI, so they cannot be used for a new project.
`pyco-cozmo` was available when this folder was prepared. The GitHub repository is
expected to be named `pyco-cozmo` as well, but the repository name and the PyPI
distribution name do not have to match.


## 1. Replace the GitHub placeholder

The metadata contains a GitHub username placeholder. Replace it with your username or
organization:

```powershell
python tools/configure_repo.py --github-user your-github-username
```

Review the result with:

```powershell
git diff
```

The prepared links use `pyco-cozmo`. If you choose a different repository name, replace
`/pyco-cozmo` with your repository name in `pyproject.toml`, `README.md`,
`CONTRIBUTING.md`, and `docs/source/overview.md`.


## 2. Configure your Git identity

The prepared commit uses a neutral placeholder author. Set your own identity and amend it
before pushing:

```powershell
git config user.name "Your Name"
git config user.email "you@example.com"
git add -A
git commit --amend --reset-author --no-edit
```

Check the result:

```powershell
git log -1 --format="%an <%ae>"
git status
```


## 3. Create the GitHub repository and push

Create an empty public repository on GitHub named `pyco-cozmo` (do **not** initialize it
with a README, license, or `.gitignore`). Then:

```powershell
git remote add origin https://github.com/Janosaus123/pyco-cozmo.git
git push -u origin main
```

The `CI` workflow runs on every push and pull request. It tests Python 3.9, 3.12, and
3.14 on Linux, Windows, and macOS, exercises the optional OpenCV vision extras, and
builds/checks the sdist and wheel.

The original upstream repository is already configured as `upstream`:

```powershell
git remote -v
```


## 4. Create the PyPI project with trusted publishing

1. Create an account at <https://pypi.org/account/register/> and verify your email.
2. Open <https://pypi.org/manage/account/publishing/>.
3. Add a pending trusted publisher with these values:

   | Field | Value |
   | --- | --- |
   | PyPI project name | `pyco-cozmo` |
   | Owner | your GitHub username |
   | Repository name | `pyco-cozmo` |
   | Workflow name | `publish.yml` |
   | Environment name | `pypi` |

4. In the GitHub repository, open **Settings → Environments → New environment**, create an
   environment named `pypi`, and optionally add protection rules. No PyPI token or secret
   is needed; GitHub's OIDC identity is used by the publish workflow.

The `publish.yml` workflow builds the distributions and uploads them through PyPI's
trusted-publishing action when a GitHub release is published.


## 5. Optional: test on TestPyPI first

TestPyPI is a separate service and needs its own pending trusted publisher:

1. Create an account at <https://test.pypi.org/account/register/>.
2. Open <https://test.pypi.org/manage/account/publishing/>.
3. Add a pending publisher for project `pyco-cozmo`, owner/repository as above, workflow
   `publish-testpypi.yml`, environment `testpypi`.
4. In GitHub, create an environment named `testpypi`.
5. Run **Actions → Publish to TestPyPI → Run workflow**.

Install the TestPyPI build in a clean virtual environment with:

```powershell
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ pyco-cozmo
```


## 6. Publish to PyPI

1. Make sure the version in `pyproject.toml` is the version you want to release.
2. Commit and push any changes.
3. On GitHub, open **Releases → Draft a new release**.
4. Create a tag such as `v0.9.0`, add release notes, and click **Publish release**.
5. The **Publish to PyPI** workflow builds the sdist and wheel and uploads them.

You can also run **Actions → Publish to PyPI → Run workflow** from `main` to retry a
failed publish. A version can only be uploaded to PyPI once; if an upload partially
succeeds, bump the version before retrying.


## 7. Verify the published package

In a clean environment:

```powershell
python -m venv .venv-pypi-test
.venv-pypi-test\Scripts\activate
python -m pip install --upgrade pip
pip install pyco-cozmo
python -c "import pycozmo; print(pycozmo.__version__)"
pycozmo-discover --help
```

The package installs as `pyco-cozmo`, imports as `pycozmo`, and keeps the existing
console-script names such as `pycozmo-discover`.


## Manual upload fallback

If you prefer not to use GitHub Actions, build and upload from this folder:

```powershell
python -m pip install --upgrade build twine
python -m build
python -m twine check dist/*
python -m twine upload dist/*
```

`twine` asks for a PyPI API token. Create one at
<https://pypi.org/manage/account/token/>. Never commit the token or put it in the
repository.

For TestPyPI:

```powershell
python -m twine upload --repository testpypi dist/*
```


## Releasing a new version

1. Update `version` in `pyproject.toml`.
2. Add an entry to `CHANGES.md`.
3. Run the tests and linter:
   ```powershell
   python -m pytest -q
   python -m flake8 pycozmo examples tools setup.py
   ```
4. Commit and push:
   ```powershell
   git push origin main
   ```
5. Create a GitHub release with a new tag such as `vX.Y.Z`. The release UI creates the
   tag, and the workflow uploads the new version to PyPI. If you created the tag locally
   instead, push only that tag with `git push origin vX.Y.Z`.
