"""ggufy entry point.

Run the CLI:  python main.py <command> <file> [options]
Run the GUI:  python main.py gui
"""

import sys


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "gui":
        from ggufy.gui import run_gui
        run_gui()
        return 0
    from ggufy.cli import main as cli_main
    return cli_main()


if __name__ == "__main__":
    sys.exit(main())
