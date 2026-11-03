from pathlib import Path

path = Path("doall.py")
lines = path.read_text(encoding="utf-8").splitlines()

matches = [
    i for i, line in enumerate(lines)
    if '"feedback_initial",' in line
]

if len(matches) != 2:
    raise RuntimeError(f"Expected exactly 2 feedback_initial entries, found {len(matches)}. No changes made.")

second = matches[1]

# The second block is:
# log_event(
#     "feedback_initial",
#     oracle_id=...
#     test_name=...
#     oracle=...
#     result=...
#     feedback=...
# )
start = second - 1
end = second + 7

if lines[start].strip() != "log_event(" or lines[end].strip() != ")":
    raise RuntimeError("Second feedback_initial block structure unexpected. No changes made.")

del lines[start:end + 1]

path.write_text("\n".join(lines) + "\n", encoding="utf-8")

print("Duplicate feedback_initial removed.")
