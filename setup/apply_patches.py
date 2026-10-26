"""One-time setup for the fork. Run from the repo root (idempotent: safe to run twice).

    python setup/apply_patches.py

1. Inserts two attributes into src/controller/da_mcts/controller.py so the runner can read
   the iteration count and budget of the last planning call.
2. Adds venv/, results/, __pycache__/ to .gitignore.
3. Keeps the authors' README as README_UPSTREAM.md and installs README_STUDY.md as README.md.
"""
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CTRL = REPO / "src" / "controller" / "da_mcts" / "controller.py"


def patch_controller():
    if not CTRL.exists():
        sys.exit(f"{CTRL} not found. Run this from inside your fork.")
    s = CTRL.read_text()
    if "_last_n_sims" in s:
        print("controller.py: already patched, skipping")
        return

    anchor1 = "        self._timestep = 0\n        self._local_cx = np.zeros(106)\n"
    add1 = ("        self._timestep = 0\n"
            "        self._last_n_sims = -1\n"
            "        self._last_budget_s = float(\"nan\")\n"
            "        self._local_cx = np.zeros(106)\n")
    anchor2 = ("        log.debug(\"MCTS ran %d sims (C++ MLP, budget=%.1fms)\", n_sims, "
               "budget * 1000 if self.mcts_time_budget > 0 else 0)\n")
    add2 = (anchor2 +
            "        self._last_n_sims = int(n_sims)\n"
            "        self._last_budget_s = budget if self.mcts_time_budget > 0 else float(\"nan\")\n")

    for name, anchor in (("__init__ anchor", anchor1), ("compute_action anchor", anchor2)):
        if s.count(anchor) != 1:
            sys.exit(f"controller.py: could not find the {name} exactly once "
                     f"(found {s.count(anchor)}). The file differs from what I expected; "
                     "make the two inserts by hand (see docs/methodology.md).")
    shutil.copy(CTRL, CTRL.with_suffix(".py.bak"))
    s = s.replace(anchor1, add1).replace(anchor2, add2)
    CTRL.write_text(s)
    print("controller.py: patched (backup: controller.py.bak)")


def patch_gitignore():
    gi = REPO / ".gitignore"
    text = gi.read_text() if gi.exists() else ""
    have = text.splitlines()
    want = ["venv/", "results/", "__pycache__/", "*.pyc", "*.old.csv"]
    missing = [w for w in want if w not in have]
    if missing:
        with open(gi, "a") as f:
            f.write(("\n" if text and not text.endswith("\n") else "") + "\n".join(missing) + "\n")
    print(".gitignore: added", missing if missing else "nothing (already complete)")


def patch_readme():
    readme, upstream, study = REPO / "README.md", REPO / "README_UPSTREAM.md", REPO / "README_STUDY.md"
    if not study.exists():
        print("README: README_STUDY.md not found, skipping")
        return
    if readme.exists() and "fork of DA-MCTS" in readme.read_text()[:300]:
        print("README.md: already the study README, left untouched")
        return
    if readme.exists() and not upstream.exists():
        shutil.copy(readme, upstream)
    shutil.copy(study, readme)
    print("README.md: replaced by the study README (original saved as README_UPSTREAM.md)")


if __name__ == "__main__":
    patch_controller()
    patch_gitignore()
    patch_readme()
    print("\nDone. Next: python -m pip install pandas matplotlib pyyaml")
