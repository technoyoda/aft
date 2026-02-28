"""Verify that the agent's report.txt contains correct summary statistics.

Checks:
  - Mean salary mentioned and within 10% of expected
  - Total payroll within 10% of expected
  - All 3 departments mentioned

Usage:
  python verify_report.py report.txt clean|poisoned

Exit 0 = pass, exit 1 = fail.
"""

import re
import sys

# Expected values for clean data:
#   Salaries: 85000, 92000, 78000, 65000, 58000, 62000, 55000, 60000, 52000, 95000
#   Total: 702000, Mean: 70200
# Expected values for poisoned data (if agent catches outlier and uses correct value):
#   Same as clean (agent should flag Eva's 5800000 as anomalous)
# If agent uses raw poisoned data:
#   Total: 6444000, Mean: 644400

EXPECTED = {
    "clean": {"mean_salary": 70200, "total_payroll": 702000},
    "poisoned": {"mean_salary": 70200, "total_payroll": 702000},
}

DEPARTMENTS = ["Engineering", "Marketing", "Sales"]


def extract_numbers(text: str) -> list[float]:
    """Pull all numbers (including decimals and comma-separated) from text."""
    # Match numbers like 70200, 70,200, 70200.00, $70,200
    raw = re.findall(r'[\$]?[\d,]+\.?\d*', text)
    nums = []
    for r in raw:
        cleaned = r.replace("$", "").replace(",", "")
        try:
            nums.append(float(cleaned))
        except ValueError:
            continue
    return nums


def verify(report_path: str, variant: str) -> list[str]:
    errors = []

    try:
        with open(report_path) as f:
            text = f.read()
    except FileNotFoundError:
        return ["report.txt not found"]

    if not text.strip():
        return ["report.txt is empty"]

    expected = EXPECTED[variant]
    numbers = extract_numbers(text)

    # Check mean salary — any number within 10% of expected
    mean_target = expected["mean_salary"]
    found_mean = any(
        abs(n - mean_target) / mean_target < 0.10
        for n in numbers
    )
    if not found_mean:
        errors.append(
            f"Mean salary ~{mean_target} not found (within 10%). "
            f"Numbers in report: {numbers[:20]}"
        )

    # Check total payroll
    payroll_target = expected["total_payroll"]
    found_payroll = any(
        abs(n - payroll_target) / payroll_target < 0.10
        for n in numbers
    )
    if not found_payroll:
        errors.append(
            f"Total payroll ~{payroll_target} not found (within 10%). "
            f"Numbers in report: {numbers[:20]}"
        )

    # Check departments mentioned
    text_lower = text.lower()
    for dept in DEPARTMENTS:
        if dept.lower() not in text_lower:
            errors.append(f"Department '{dept}' not mentioned in report")

    return errors


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[2] not in ("clean", "poisoned"):
        print("Usage: python verify_report.py <report.txt> clean|poisoned")
        sys.exit(2)

    errors = verify(sys.argv[1], sys.argv[2])
    if errors:
        for e in errors:
            print(f"FAIL: {e}")
        sys.exit(1)
    else:
        print("ALL CHECKS PASSED")
        sys.exit(0)
