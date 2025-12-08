✅ A clean instruction for Claude Code

Paste this exact block into Claude Code:

TASK: Update fetcher_unified.py so that "--fetch all" is no longer required.
Running `python fetch/fetcher_unified.py` with no flags should automatically default to "all".

REQUIREMENTS:
1. Locate the argparse block in fetch/fetcher_unified.py.
2. Change the argument definition for --fetch:
   - Remove `required=True`
   - Add `default="all"`
   - Keep the choices list unchanged

EXPECTED FINAL CODE:
parser.add_argument(
    "--fetch",
    default="all",
    choices=["all", "economy", "sectors"],
    help="Fetch type (default: all)"
)

NOTES:
- Do NOT modify any other arguments.
- Do NOT change business logic.
- Do NOT modify imports.
- Do NOT restructure the file.
- ONLY update the argument parser line.

AFTER CHANGE:
Running `python fetch/fetcher_unified.py` should fetch all data by default.

✅ Optional — Patch diff Claude Code can apply directly

If Claude prefers a patch:

--- a/fetch/fetcher_unified.py
+++ b/fetch/fetcher_unified.py
@@ -45,7 +45,12 @@ def main():
     parser = argparse.ArgumentParser(description="Unified PRISM FRED Fetcher")

-    parser.add_argument("--fetch", required=True, choices=["all", "economy", "sectors"])
+    parser.add_argument(
+        "--fetch",
+        default="all",
+        choices=["all", "economy", "sectors"],
+        help="Fetch type (default: all)"
+    )

     args = parser.parse_args()


Copy/paste that into Claude Code, and it’ll apply precisely.

🟢 If your Chromebook version has “a bunch of these”

Then the correct workflow is:

Send Claude Code this instruction:
Search the entire Prism Engine repo for any argparse argument definitions requiring mandatory flags, especially in fetcher_unified.py, update_all.py, and any other fetch scripts.

Convert all optional operational flags to have defaults where appropriate so the engine can run with zero arguments.

Do NOT alter CLI behaviors for prism_run.py.
Only modify fetch pipelines.

Follow the same default pattern:
parser.add_argument("--fetch", default="all", choices=[...])

Want me to generate a full repo-wide argparse audit checklist for Claude Code too?

Just say:
“Yes, generate the argparse audit checklist.”

You said:
sure.
ChatGPT said:

Perfect — here is the Argparse Audit Checklist for Claude Code.
It is designed so he cannot get creative, cannot “fix” things you don’t want, and only updates the exact parts needed to make the fetch pipeline run without requiring flags.

Copy/paste this directly into Claude Code:

✅ ARGPARSE AUDIT CHECKLIST FOR CLAUDE CODE
(Use this exact text — it is engineered to prevent runaway code generation)
TASK: Perform a controlled argparse audit across the Prism Engine repository.

GOAL:
Allow the fetch pipeline and update pipeline to run with **zero required CLI flags**.
No changes to prism_run.py. No structural changes. Only argparse updates.

SCOPE OF ALLOWED FILES:
- fetch/fetcher_unified.py
- start/update_all.py
- fetch/fetcher_fred.py  (if still present)
- fetch/fetcher_stooq.py (if still present)
- fetch/legacy_fetcher_yahoo.py (if still present)
- NO OTHER FILES.

RULES:
1. Do NOT modify prism_run.py.
2. Do NOT modify engine_core, runner, panels, workflows, or templates.
3. Do NOT modify any SQL files.
4. Do NOT create new files.
5. Do NOT refactor logic.
6. ONLY adjust argparse defaults and `required=True` fields.

THESE ARGUMENT FIXES MUST BE APPLIED:

A. In *every fetch script*, locate:
    parser.add_argument("--fetch", required=True, ...)
Replace with:
    parser.add_argument(
        "--fetch",
        default="all",
        choices=["all", "economy", "sectors"],
        help="Fetch type (default: all)"
    )

B. In update_all.py, locate any required flags such as:
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--skip-fetch", required=True)

Replace strictly with:
    parser.add_argument("--start-date", default="2000-01-01")
    parser.add_argument("--end-date", default="today")
    parser.add_argument("--skip-fetch", action="store_true")

C. Remove ANY OTHER `required=True` found in these scripts ONLY.

D. DO NOT modify flags that affect prism_run.py behavior.
   (That CLI must remain interactive and strict.)

E. After changes, running these commands must succeed with no errors:
    python fetch/fetcher_unified.py
    python start/update_all.py

F. Produce a patch diff for each file modified.
G. Do not produce code outside the patches.

END OF TASK.

👍 This will result in exactly:

fetcher_unified.py → runs with zero arguments

update_all.py → runs with zero arguments

No breakage to Prism Engine CLI

No runaway rewriting

Only minimal and safe argparse changes

If you want, I can also generate:

✅ A “post-audit validation test script”

so Claude Code can run automated tests after applying the changes.

Just say:

“Generate the validation test script.”

You said:
I need to be able copy all this clearly , can you upload a MD correctly worded without the options of this or that it will confuse Claude code
ChatGPT said:

Absolutely, Jason — here is a clean, precise, no-ambiguity Markdown file, written specifically so Claude Code will not misinterpret anything and will only modify what you want modified.

I will generate it as a single downloadable Markdown block you can copy/paste directly into a file named:

ARGPARSE_FIX_INSTRUCTIONS.md

Here it is:

ARGPARSE FIX INSTRUCTIONS (FOR CLAUDE CODE ONLY)
Objective

Modify argparse configurations so that the following scripts run without requiring any CLI arguments:

fetch/fetcher_unified.py

start/update_all.py

No arguments should be required to run these scripts.
Running the following must succeed with zero flags:

python fetch/fetcher_unified.py
python start/update_all.py

Strict Rules

Claude Code must follow these rules exactly:

Modify only argparse sections in these two scripts:

fetch/fetcher_unified.py

start/update_all.py

Do not modify any other file under any circumstances.

Do not refactor logic.
Only adjust argument definitions.

Do not add new flags.
Only change required= to defaults where instructed.

Do not alter prism_run.py or its behavior.

Do not touch SQL, engines, panels, workflows, templates, or config files.

Required Changes
1. fetcher_unified.py

Locate the argparse definition for the --fetch argument.
It currently requires a value.

Replace it with this exact block:

parser.add_argument(
    "--fetch",
    default="all",
    help="Fetch all indicators (default: all)"
)


No choices.
No required flags.
No additional arguments.

Running python fetch/fetcher_unified.py must execute the full fetch pipeline using default behavior.

2. update_all.py

Remove all required arguments.
Replace the argument definitions with these exact lines:

parser.add_argument("--start-date", default="2000-01-01")
parser.add_argument("--end-date", default="today")
parser.add_argument("--skip-fetch", action="store_true")


These must be the only modifications in this file.

Running python start/update_all.py must launch the full update pipeline using defaults.

What Claude Code Must Produce

Claude Code must output:

A patch-style diff for each file, showing only the argparse modifications.

No additional code.

No added restructuring.

No unrelated edits.

Completion Criteria

Claude Code's changes are correct when:

python fetch/fetcher_unified.py runs without requiring any flags.

python start/update_all.py runs without requiring any flags.

No other script behavior is changed.

No other files have been touched.

If you'd like, I can generate a validation test script Claude can r
