"""Install tests/fake_tesseract.py as a real executable at a chosen path."""

from __future__ import annotations

import shutil
import stat
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def install_fake_tesseract(target: Path) -> Path:
    """Create a program at `target` (parent folders are made) that runs the fake."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if sys.platform == "win32":
        _install_windows(target)
    else:
        target.write_text(
            f"#!{sys.executable}\nimport runpy\nrunpy.run_path({str(HERE / 'fake_tesseract.py')!r},"
            ' run_name="__main__")\n',
            encoding="utf-8",
        )
        target.chmod(target.stat().st_mode | stat.S_IRWXU | stat.S_IRGRP | stat.S_IROTH)
    return target


def _install_windows(target: Path) -> None:
    from pip._vendor.distlib.scripts import ScriptMaker

    with tempfile.TemporaryDirectory() as work:
        maker = ScriptMaker(None, work)
        maker.executable = sys.executable
        maker.variants = {""}
        maker.clobber = True
        maker.script_template = (  # the launched program must find the fake next to this file
            f"import sys\nsys.path.insert(0, {str(HERE)!r})\n"
            "from %(module)s import %(import_name)s\nsys.exit(%(func)s())\n"
        )
        made = maker.make_multiple(["fake = fake_tesseract:main"], {"gui": False})
        exe = next(Path(m) for m in made if m.lower().endswith(".exe"))
        shutil.copyfile(exe, target)
