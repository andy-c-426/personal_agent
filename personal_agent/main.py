import argparse
from personal_agent.config import Config
from personal_agent.cli.app import run


def main():
    parser = argparse.ArgumentParser(description="Personal Agent — local AI assistant")
    parser.add_argument("--debug", action="store_true", help="Print full tool payloads")
    args = parser.parse_args()

    config = Config.from_env()

    if not config.deepseek_api_key:
        print("Error: DEEPSEEK_API_KEY environment variable is required.")
        return 1
    if not config.tavily_api_key:
        print("Error: TAVILY_API_KEY environment variable is required.")
        return 1

    run(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
