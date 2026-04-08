"""Allow running assessment subcommands via python -m assessment <command>."""
import sys


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m assessment <command> [options]")
        print("Commands: generate, run_pipeline, make_report")
        sys.exit(1)

    command = sys.argv[1]
    sys.argv = sys.argv[1:]  # shift args so subcommand sees correct argv

    if command == "generate":
        from .generate import main as gen_main
        gen_main()
    elif command == "run_pipeline":
        from .run_pipeline import main as run_main
        run_main()
    elif command == "make_report":
        from .make_report import main as report_main
        report_main()
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
