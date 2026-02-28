"""Flask API server for the data processing task.

Serves employee records from a JSON data file. Two endpoints:
  GET /records  — list of employee records
  GET /metadata — summary info (total_employees, departments)

Usage:
  python api_server.py --data api_data_clean.json --port 5050
"""

import argparse
import json
import sys

from flask import Flask, jsonify


def create_app(data_path: str) -> Flask:
    with open(data_path) as f:
        records = json.load(f)

    departments = sorted(set(r["department"] for r in records))

    app = Flask(__name__)

    @app.route("/records")
    def get_records():
        return jsonify(records)

    @app.route("/metadata")
    def get_metadata():
        return jsonify({
            "total_employees": len(records),
            "departments": departments,
        })

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to JSON data file")
    parser.add_argument("--port", type=int, default=5050)
    args = parser.parse_args()

    app = create_app(args.data)
    app.run(port=args.port, debug=False)
