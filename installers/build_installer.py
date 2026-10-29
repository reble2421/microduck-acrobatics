"""Build installers/setup_acrobatics.py (one self-checking installer) from tasks/.

    python3 installers/build_installer.py

The installer carries every task and test file verbatim plus a SHA-256 of the payload,
checks the target microduck_rl clone first and changes nothing unless every check
passes. Running it twice is safe.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "tasks"
OUT = ROOT / "installers" / "setup_acrobatics.py"

TASK_FILES = ["headstand_mdp.py", "microduck_headstand_env_cfg.py", "microduck_backroll_env_cfg.py",
              "headspin_mdp.py", "microduck_headspin_env_cfg.py", "microduck_robust_env_cfg.py",
              "microduck_headspin_fast_env_cfg.py"]
TEST_FILES = ["test_headstand_cfg.py", "test_backroll_cfg.py", "test_headspin_cfg.py",
              "test_robust_cfg.py", "test_headspin_fast_cfg.py"]

HEADER = '''#!/usr/bin/env python3
"""Add the microduck-acrobatics tasks to a fresh microduck_rl clone.

    git clone https://github.com/pollen-robotics/microduck_rl && cd microduck_rl
    git checkout 53b8971b61baf5b7f3c16d135dd7cac37623de4b
    python3 /path/to/setup_acrobatics.py

Tasks: Mjlab-HeadStand-Flat-MicroDuck, Mjlab-BackRoll-Flat-MicroDuck,
Mjlab-HeadSpin-Flat-MicroDuck, Mjlab-BackRollRobust-Flat-MicroDuck,
Mjlab-HeadSpinRobust-Flat-MicroDuck, Mjlab-HeadSpinFast5-Flat-MicroDuck,
Mjlab-HeadSpinFast6-Flat-MicroDuck.

Everything is checked first; nothing is changed unless every check passes. Running it
again is safe (it only fills in what is missing). Two existing files of the clone are
changed: tasks/__init__.py gets the registrations appended, and tasks/mdp.py gets nine
small edits that let the roulade's reset and head latch run backwards (a roll
direction and a head-latch window per task; the defaults keep the forward roulade
exactly as it was).
"""

import hashlib
import subprocess
import sys
from pathlib import Path

PINNED = "53b8971b61baf5b7f3c16d135dd7cac37623de4b"
TAG = "[acrobatics]"
'''

FOOTER = '''

FILES = [
{files_list}
]


def fail(*lines):
    print("NG: " + lines[0])
    for line in lines[1:]:
        print("    " + line)
    print("Nothing was changed.")
    sys.exit(1)


def payload_digest():
    h = hashlib.sha256()
    for rel, text in FILES:
        h.update(rel.encode("utf-8"))
        h.update(text.encode("utf-8"))
    h.update(INIT_APPEND.encode("utf-8"))
    for old, rep, count in MDP_EDITS:
        h.update(old.encode("utf-8"))
        h.update(rep.encode("utf-8"))
        h.update(str(count).encode("utf-8"))
    return h.hexdigest()


MDP_REL = "src/mjlab_microduck/tasks/mdp.py"
MDP_MARK = "[backroll] Roll direction"       # present once the edits below are in


# Things the new files rely on in the microduck_rl clone.
NEEDED = [
    ("src/mjlab_microduck/tasks/microduck_roulade_env_cfg.py",
     ("def make_microduck_roulade_env_cfg(", "TUCK_OVERRIDES", "MicroduckRouladeRlCfg =")),
    ("src/mjlab_microduck/tasks/mdp.py",
     ("def reset_roulade_state(", "class VelocityCommandCommandOnly", "def velocity_command_ranges_curriculum(",
      "def event_param_curriculum(", "def reward_weight(")),
    ("src/mjlab_microduck/robot/microduck_constants.py", ("delay_min_lag=3", "delay_max_lag=6")),
    ("src/mjlab_microduck/tasks/__init__.py", ("register_mjlab_task", "MicroduckOnPolicyRunner")),
]
# Task registrations of earlier per-task installers: they would collide with ours.
OLD_TAGS = ("[headstand]", "[backroll]", "[headspin]", "[robust]", "[headspin_fast]")


def main():
    root = Path.cwd()
    if payload_digest() != PAYLOAD_SHA256:
        fail("this script was not copied exactly (a line is missing or changed).")
    pyproject = root / "pyproject.toml"
    if not pyproject.is_file() or "mjlab-microduck" not in pyproject.read_text(encoding="utf-8"):
        fail("run this inside the microduck_rl folder:  cd microduck_rl")
    for rel, needles in NEEDED:
        p = root / rel
        if not p.is_file():
            fail(f"{{rel}} not found")
        text = p.read_text(encoding="utf-8")
        for needle in needles:
            if needle not in text:
                fail(f"{{rel}} does not look like the tested version (missing: {{needle}})")
    init = root / "src" / "mjlab_microduck" / "tasks" / "__init__.py"
    init_text = init.read_text(encoding="utf-8")
    old = [t for t in OLD_TAGS if t in init_text]
    if old:
        fail("tasks were already registered by older installers: " + ", ".join(old),
             "Use a fresh clone of microduck_rl.")

    # The backward roll needs the roulade's reset / latch helpers in mdp.py to take a
    # roll direction and a head-latch window (defaults = the forward roulade, unchanged).
    mdp_path = root / MDP_REL
    mdp = mdp_path.read_text(encoding="utf-8")
    new_mdp = None
    if MDP_MARK not in mdp:
        new_mdp = mdp
        for old, rep, count in MDP_EDITS:
            found = new_mdp.count(old)
            if found != count:
                fail(f"mdp.py: expected {{count}} match(es), found {{found}}, for:", old.strip().splitlines()[0])
            new_mdp = new_mdp.replace(old, rep)

    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                          capture_output=True, text=True).stdout.strip()
    if head != PINNED:
        print(f"WARN: repository is at {{head[:7] or '?'}}; this script was tested on {{PINNED[:7]}}.")

    changed = 0
    if new_mdp is None:
        print(f"[OK] {{MDP_REL}} (roll direction already there)")
    else:
        mdp_path.write_text(new_mdp, encoding="utf-8")
        changed += 1
        print(f"[OK] {{MDP_REL}}: roll direction and head-latch window are now per-task settings")
    for rel, text in FILES:
        p = root / rel
        if p.is_file() and p.read_text(encoding="utf-8") == text:
            print(f"[OK] {{rel}} (already there)")
            continue
        p.write_text(text, encoding="utf-8")
        changed += 1
        print(f"[OK] wrote {{rel}}")
    if TAG in init_text:
        print("[OK] tasks already registered")
    else:
        with init.open("a", encoding="utf-8") as f:
            f.write(INIT_APPEND)
        changed += 1
        print("[OK] registered 7 tasks")
    if changed == 0:
        print("Already installed. Nothing to do.")
    print("DONE")


if __name__ == "__main__":
    main()
'''


def main():
    parts = [HEADER]
    entries = []
    h = hashlib.sha256()
    for i, name in enumerate(TASK_FILES + TEST_FILES):
        is_test = name in TEST_FILES
        text = (SRC / ("tests" if is_test else "mjlab_microduck/tasks") / name).read_text(encoding="utf-8")
        assert "'''" not in text, name
        dest = f"tests/{name}" if is_test else f"src/mjlab_microduck/tasks/{name}"
        const = f"TEXT_{i:02d}"
        parts.append(f"\n{const} = r'''" + text + "'''\n")
        entries.append((dest, const))
        h.update(dest.encode("utf-8"))
        h.update(text.encode("utf-8"))
    init_text = (SRC / "init_append.txt").read_text(encoding="utf-8")
    assert "'''" not in init_text
    parts.append("\nINIT_APPEND = r'''" + init_text + "'''\n")
    h.update(init_text.encode("utf-8"))
    edits = [tuple(e) for e in json.loads((SRC / "mdp_edits_backroll.json").read_text(encoding="utf-8"))]
    parts.append("\n# (text that must be in mdp.py, replacement, how many times it must appear)\n"
                 "MDP_EDITS = [\n" + "".join(f"    {e!r},\n" for e in edits) + "]\n")
    for old, rep, count in edits:
        h.update(old.encode("utf-8"))
        h.update(rep.encode("utf-8"))
        h.update(str(count).encode("utf-8"))
    parts.insert(1, f'PAYLOAD_SHA256 = "{h.hexdigest()}"\n')
    files_list = "\n".join(f'    ("{dest}", {const}),' for dest, const in entries)
    parts.append(FOOTER.format(files_list=files_list))
    out = "".join(parts)
    bad = sorted({c for c in out if ord(c) > 127})
    assert not bad, f"non-ASCII characters: {bad}"
    OUT.write_text(out, encoding="utf-8", newline="\n")
    print(f"wrote {OUT} ({len(out.splitlines())} lines), payload sha256 {h.hexdigest()[:12]}")


if __name__ == "__main__":
    main()
