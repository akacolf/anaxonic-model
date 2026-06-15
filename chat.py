from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    venv_python = repo_root / ".venv" / "bin" / "python"
    if venv_python.exists() and Path(sys.executable).resolve() != venv_python.resolve():
        env = os.environ.copy()
        src_path = str(repo_root / "src")
        env["PYTHONPATH"] = (
            src_path if not env.get("PYTHONPATH") else f"{src_path}{os.pathsep}{env['PYTHONPATH']}"
        )
        os.execve(str(venv_python), [str(venv_python), "-m", "anaxonic.chat", *sys.argv[1:]], env)

    sys.path.insert(0, str(repo_root / "src"))
    from anaxonic.chat import main as chat_main

    chat_main()


if __name__ == "__main__":
    main()
