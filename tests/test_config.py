"""Settings read from environment variables: defaults, validation and secret hygiene."""

import os
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from folio_ocr import config
from folio_ocr.config import Settings, default_weights_dir, find_tesseract, is_loopback
from folio_ocr.errors import ConfigError

S = "SENTINEL-4242"
WEIGHTS = "/w"
ABS = Path(os.path.abspath(os.sep)) / "cfg" / "tesseract"


def load(env=None, **kw):
    return Settings.from_env({"FOLIO_WEIGHTS_DIR": WEIGHTS, **(env or {})}, **kw)


def fails(env, name, **kw):
    with pytest.raises(ConfigError) as ei:
        load(env, **kw)
    msg = str(ei.value)
    assert name in msg
    assert S not in msg
    assert msg.splitlines() == [msg] and msg
    return msg


def test_defaults_from_empty_environment():
    s = Settings.from_env({}, windows=False)
    assert (s.host, s.port, s.line_timeout_s, s.page_deadline_s) == ("127.0.0.1", 8000, 60.0, 900.0)
    assert s.two_read is False
    assert s.model_url is s.model_name is s.model_api_key is s.tesseract_path is None
    assert s.model_host is None


def test_empty_strings_count_as_unset():
    names = [
        "FOLIO_MODEL_URL",
        "FOLIO_MODEL_NAME",
        "FOLIO_MODEL_API_KEY",
        "FOLIO_TESSERACT_PATH",
        "FOLIO_WEIGHTS_DIR",
        "FOLIO_HOST",
        "FOLIO_PORT",
        "FOLIO_LINE_TIMEOUT_S",
        "FOLIO_PAGE_DEADLINE_S",
        "FOLIO_TWO_READ",
    ]
    s = Settings.from_env(dict.fromkeys(names, ""), windows=False)
    assert s == Settings.from_env({}, windows=False)


def test_good_values_parsed():
    s = load(
        {
            "FOLIO_PORT": "8080",
            "FOLIO_LINE_TIMEOUT_S": "30",
            "FOLIO_PAGE_DEADLINE_S": "2.5",
            "FOLIO_TWO_READ": "1",
            "FOLIO_MODEL_NAME": " Odd Name ",
            "FOLIO_TESSERACT_PATH": str(ABS),
        }
    )
    assert (s.port, s.line_timeout_s, s.page_deadline_s, s.two_read) == (8080, 30.0, 2.5, True)
    assert s.model_name == " Odd Name "
    assert s.tesseract_path == ABS and s.weights_dir == Path(WEIGHTS)
    assert load({"FOLIO_TWO_READ": "0"}).two_read is False
    assert load({"FOLIO_PORT": "65535"}).port == 65535
    assert load({"FOLIO_PORT": "1"}).port == 1


@pytest.mark.parametrize("host", ["::1", "localhost", "127.0.0.2", "LOCALHOST"])
def test_loopback_hosts_accepted(host):
    assert load({"FOLIO_HOST": host}).host == host


def test_remote_host_needs_allow_remote():
    msg = fails({"FOLIO_HOST": "0.0.0.0"}, "FOLIO_HOST")
    assert "--allow-remote" in msg
    assert load({"FOLIO_HOST": "0.0.0.0"}, allow_remote=True).host == "0.0.0.0"
    assert load({"FOLIO_HOST": "example.com"}, allow_remote=True).host == "example.com"


@pytest.mark.parametrize("host", ["bad host!", f"{S}!", "-a", "a-", "é.com", "[::1]", "a b"])
def test_bad_host_refused_even_with_allow_remote(host):
    fails({"FOLIO_HOST": host}, "FOLIO_HOST", allow_remote=True)


@pytest.mark.parametrize(
    "v", ["0", "65536", " 8000", "8000 ", "+8000", "8_000", "٨٠٠٠", "abc", "-1", "", S]
)
def test_bad_port(v):
    if v == "":
        assert load({"FOLIO_PORT": v}).port == 8000
        return
    fails({"FOLIO_PORT": v}, "FOLIO_PORT")


@pytest.mark.parametrize("name", ["FOLIO_LINE_TIMEOUT_S", "FOLIO_PAGE_DEADLINE_S"])
@pytest.mark.parametrize(
    "v", ["0", "0.0", "nan", "inf", "-1", "1e3", " 5", "5 ", "5.", ".5", "٥", "1_0", S]
)
def test_bad_timeouts(name, v):
    fails({name: v}, name)


@pytest.mark.parametrize("v", ["2", "true", " 1", "01", "1 ", "yes", S])
def test_bad_two_read(v):
    fails({"FOLIO_TWO_READ": v}, "FOLIO_TWO_READ")


KEY = "FOLIO_MODEL_API_KEY"
URL = "FOLIO_MODEL_URL"


@pytest.mark.parametrize(
    "url",
    [
        "http://remote.example/v1",
        "http://127.0.0.1.example.com",
        "http://localhost.example.com",
        "http://10.0.0.5:11434",
    ],
)
def test_key_refused_over_http_to_remote(url):
    msg = fails({URL: url, KEY: "sk-" + S}, KEY)
    assert "https://" in msg


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:11434/v1",
        "http://localhost:11434",
        "http://[::1]:1",
        "http://[::ffff:127.0.0.1]/",
        "HTTP://127.0.0.1:11434",
        "https://remote.example/v1",
    ],
)
def test_key_allowed(url):
    s = load({URL: url, KEY: "sk-1"})
    assert s.model_url == url and s.model_api_key == "sk-1"


def test_http_remote_without_key_allowed():
    assert load({URL: "http://remote.example/v1"}).model_url == "http://remote.example/v1"


@pytest.mark.parametrize(
    "url",
    [
        "ftp://x",
        "http://",
        "not a url",
        "http://host:99999",
        "http://exa mple.com",
        "http://a.example/\n",
        "http://a.example/\x00",
        f"ftp://u:{S}@x",
        f"http://u:{S}@:80",
        f"http://u:{S}@h:99999",
        f"http://u:{S}@h /",
        "",
        "http://[::1",
    ],
)
def test_bad_url(url):
    if url == "":
        assert load({URL: url}).model_url is None
        return
    fails({URL: url}, URL)


def test_url_kept_exactly():
    url = "HTTP://User:pw@Example.COM:8080/x?y=1#z"
    assert load({URL: url}).model_url == url


def test_bad_key_control_character():
    fails({KEY: "sk-" + S + "\n"}, KEY)
    fails({KEY: "sk-" + S + "\x07"}, KEY)


def test_secrets_never_shown():
    url = f"http://user:{S}@remote.example/"
    fails({URL: url, KEY: "sk-" + S}, KEY)
    s = load({URL: f"https://user:{S}@remote.example/", KEY: "sk-" + S})
    assert S not in repr(s) and S not in str(s)


def test_key_without_url_is_kept():
    assert load({KEY: "sk-" + S}).model_api_key == "sk-" + S


def test_huge_numbers_are_config_errors():
    fails({"FOLIO_PORT": "1" * 5000}, "FOLIO_PORT")
    fails({"FOLIO_PORT": "000080"}, "FOLIO_PORT")
    fails({"FOLIO_LINE_TIMEOUT_S": "9" * 400}, "FOLIO_LINE_TIMEOUT_S")
    fails({"FOLIO_PAGE_DEADLINE_S": "9" * 400 + ".5"}, "FOLIO_PAGE_DEADLINE_S")


def test_model_host():
    s = load({URL: "http://user:pw@Example.COM:8080/x"})
    assert s.model_host == "example.com"
    assert load().model_host is None


def test_first_failure_wins_in_fixed_order():
    msg = fails({URL: "ftp://x", "FOLIO_PORT": "0"}, URL)
    assert "FOLIO_PORT" not in msg


def test_frozen():
    with pytest.raises(FrozenInstanceError):
        load().port = 1  # type: ignore[misc]


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("localhost", True),
        ("LocalHost", True),
        ("127.0.0.1", True),
        ("127.9.9.9", True),
        ("::1", True),
        ("::ffff:127.0.0.1", True),
        ("::ffff:10.0.0.1", False),
        ("0.0.0.0", False),
        ("10.0.0.5", False),
        ("::", False),
        ("127.0.0.1.example.com", False),
        ("localhost.example.com", False),
        ("localhost.", False),
        ("127.1", False),
        ("", False),
        ("xlocalhost", False),
        ("localhostx", False),
    ],
)
def test_is_loopback(host, expected):
    assert is_loopback(host) is expected


def test_default_weights_dir_windows(tmp_path, monkeypatch):
    env = {"LOCALAPPDATA": "C:/L"}
    assert default_weights_dir(env, windows=True) == Path("C:/L") / "folio-ocr" / "assets"
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    expected = tmp_path / "AppData" / "Local" / "folio-ocr" / "assets"
    assert default_weights_dir({}, windows=True) == expected
    assert default_weights_dir({"LOCALAPPDATA": ""}, windows=True) == expected


def test_default_weights_dir_posix(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    assert default_weights_dir({"XDG_DATA_HOME": "/x"}, windows=False) == (
        Path("/x") / "folio-ocr" / "assets"
    )
    fallback = tmp_path / ".local" / "share" / "folio-ocr" / "assets"
    assert default_weights_dir({"XDG_DATA_HOME": "rel/x"}, windows=False) == fallback
    assert default_weights_dir({"XDG_DATA_HOME": ""}, windows=False) == fallback
    assert default_weights_dir({}, windows=False) == fallback


@pytest.mark.parametrize("windows", [True, False])
def test_default_weights_dir_no_home(monkeypatch, windows):
    def boom():
        raise RuntimeError("no home")

    monkeypatch.setattr(Path, "home", boom)
    with pytest.raises(ConfigError, match="FOLIO_WEIGHTS_DIR"):
        default_weights_dir({}, windows=windows)


def test_from_env_uses_default_weights_dir_and_explicit_one(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    s = Settings.from_env({}, windows=False)
    assert s.weights_dir == tmp_path / ".local" / "share" / "folio-ocr" / "assets"
    assert load().weights_dir == Path(WEIGHTS)


POSIX_ONLY = pytest.mark.skipif(os.name == "nt", reason="execute bits are a POSIX notion")


def prog(folder: Path, windows: bool) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / ("tesseract.exe" if windows else "tesseract")
    path.write_bytes(b"x")
    path.chmod(0o755)
    return path


def settings_with(path=None):
    return Settings(weights_dir=Path(WEIGHTS), tesseract_path=path)


@pytest.mark.parametrize("windows", [True, False])
def test_find_tesseract_configured_returned_as_given_without_existence_check(windows):
    assert find_tesseract(settings_with(ABS), {"PATH": ""}, windows=windows) == ABS


def test_find_tesseract_configured_beats_path_beats_windows_default(tmp_path):
    on_path = prog(tmp_path / "bin", True)
    default = prog(tmp_path / "pf" / "Tesseract-OCR", True)
    env = {"PATH": str(tmp_path / "bin"), "ProgramFiles": str(tmp_path / "pf")}
    assert find_tesseract(settings_with(ABS), env, windows=True) == ABS
    assert find_tesseract(settings_with(), env, windows=True) == on_path
    env["PATH"] = ""
    assert find_tesseract(settings_with(), env, windows=True) == default


def test_relative_configured_path_is_refused_by_from_env():
    msg = fails({"FOLIO_TESSERACT_PATH": "tesseract-" + S + ".exe"}, "FOLIO_TESSERACT_PATH")
    assert "absolute" in msg


def test_relative_configured_path_is_refused_by_find_tesseract():
    with pytest.raises(ConfigError, match="FOLIO_TESSERACT_PATH must be an absolute path"):
        find_tesseract(settings_with(Path("tesseract.exe")), {})


@pytest.mark.parametrize("windows", [True, False])
def test_find_tesseract_skips_empty_and_relative_path_entries(tmp_path, windows, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "relative" / "dir").mkdir(parents=True)
    prog(tmp_path / "relative" / "dir", windows)
    good = prog(tmp_path / "good", windows)
    entries = ["", ".", "relative/dir", str(tmp_path / "good")]
    env = {"PATH": os.pathsep.join(entries)}
    assert find_tesseract(settings_with(), env, windows=windows) == good


@pytest.mark.parametrize("windows", [True, False])
def test_find_tesseract_ignores_a_folder_with_the_program_name(tmp_path, windows):
    (tmp_path / ("tesseract.exe" if windows else "tesseract")).mkdir()
    assert find_tesseract(settings_with(), {"PATH": str(tmp_path)}, windows=windows) is None


@POSIX_ONLY
def test_find_tesseract_skips_a_file_that_cannot_be_executed(tmp_path):
    prog(tmp_path / "a", False).chmod(0o644)
    good = prog(tmp_path / "b", False)
    env = {"PATH": os.pathsep.join([str(tmp_path / "a"), str(tmp_path / "b")])}
    assert find_tesseract(settings_with(), env, windows=False) == good


@pytest.mark.parametrize("windows", [True, False])
def test_find_tesseract_never_picks_a_program_from_the_current_folder(
    tmp_path, monkeypatch, windows
):
    (tmp_path / "sub").mkdir()
    for name in ("tesseract.exe", "tesseract"):
        (tmp_path / name).write_bytes(b"x")
    monkeypatch.chdir(tmp_path)
    env = {"PATH": os.pathsep.join(["", ".", "sub"])}
    assert find_tesseract(settings_with(), env, windows=windows) is None


def test_find_tesseract_windows_default_folder(tmp_path):
    prog(tmp_path / "Tesseract-OCR", True)
    env = {"PATH": "", "ProgramFiles": str(tmp_path)}
    assert find_tesseract(settings_with(), env, windows=True) == (
        tmp_path / "Tesseract-OCR" / "tesseract.exe"
    )
    assert find_tesseract(settings_with(), env, windows=False) is None


def test_find_tesseract_ignores_a_relative_program_files(tmp_path, monkeypatch):
    prog(tmp_path / "pf" / "Tesseract-OCR", True)
    monkeypatch.chdir(tmp_path)
    env = {"PATH": "", "ProgramFiles": "pf"}
    assert find_tesseract(settings_with(), env, windows=True) is None


def raise_for(monkeypatch, blocked):
    real = Path.is_file

    def is_file(self):
        if self == blocked:
            raise PermissionError(13, "denied")
        return real(self)

    monkeypatch.setattr(Path, "is_file", is_file)


def test_find_tesseract_skips_a_path_entry_it_may_not_search(tmp_path, monkeypatch):
    name = "tesseract.exe"
    prog(tmp_path / "locked", True)
    good = prog(tmp_path / "open", True)
    raise_for(monkeypatch, tmp_path / "locked" / name)
    env = {"PATH": os.pathsep.join([str(tmp_path / "locked"), str(tmp_path / "open")])}
    assert find_tesseract(settings_with(), env, windows=True) == good
    assert find_tesseract(settings_with(), {"PATH": str(tmp_path / "locked")}, windows=True) is None


def test_find_tesseract_windows_default_it_may_not_search_is_none(tmp_path, monkeypatch):
    prog(tmp_path / "Tesseract-OCR", True)
    raise_for(monkeypatch, tmp_path / "Tesseract-OCR" / "tesseract.exe")
    env = {"PATH": "", "ProgramFiles": str(tmp_path)}
    assert find_tesseract(settings_with(), env, windows=True) is None


def test_find_tesseract_without_path_variable_is_none():
    assert find_tesseract(settings_with(), {}, windows=True) is None


def test_module_exports():
    assert config.Settings is Settings
