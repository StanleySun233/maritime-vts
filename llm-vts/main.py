import argparse
import json

from agent import run_agent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    args = parser.parse_args()
    answer, trace = run_agent([{"role": "user", "content": args.question}])
    print(json.dumps({"answer": answer, "tool_trace": trace}, ensure_ascii=False))


if __name__ == "__main__":
    main()
