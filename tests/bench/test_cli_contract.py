"""Contract tests: the CLI parser snapshot and the main() dispatch modes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from astro_denoise.bench.__main__ import build_parser, main

SNAPSHOT_PATH = Path(__file__).parent / "snapshots" / "cli_parser.json"


def _snapshot(parser):
    titles = {id(a): g.title for g in parser._action_groups for a in g._group_actions}
    return [
        {
            "option_strings": a.option_strings,
            "dest": a.dest,
            "default": repr(a.default),
            "nargs": a.nargs,
            "const": repr(a.const),
            "type": a.type.__name__ if a.type else None,
            "help": a.help,
            "group": titles.get(id(a)),
        }
        for a in parser._actions
    ]


def test_parser_matches_snapshot():
    snapshot = json.loads(SNAPSHOT_PATH.read_text())
    assert _snapshot(build_parser()) == snapshot


def _last_json_line(capsys):
    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    return json.loads(lines[-1])


def test_list_mode_returns_ok(tmp_path, capsys):
    assert main(["--list"]) == 0
    assert _last_json_line(capsys)["status"] == "ok"


def test_new_method_scaffold_and_repeat(tmp_path, capsys):
    methods_dir = tmp_path / "methods"
    methods_dir.mkdir()
    assert main(["--new-method", "wavelet", "--methods-dir", str(methods_dir)]) == 0
    assert (methods_dir / "astro-denoise-wavelet").is_dir()
    assert _last_json_line(capsys)["status"] == "ok"

    assert main(["--new-method", "wavelet", "--methods-dir", str(methods_dir)]) == 1
    assert _last_json_line(capsys)["status"] == "error"


def test_no_args_exits_with_usage_error(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main([])
    assert excinfo.value.code == 2
