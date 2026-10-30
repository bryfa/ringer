#!/usr/bin/env python3
"""reporting_emitter (work#1038) — both directions.

It emits: step records with measured costs/tokens, a fragment whose collections
match the work#1036 metric field names, a commit on main. It refuses: a 0 where
the meter reported nothing (null-never-0 is the whole point), a commit on another
session's branch, an exception escaping emit() into the run's result.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ringer import (  # noqa: E402
    ReportingConfig,
    build_fragment,
    emit_reporting_fragment,
    load_reporting_config,
    parse_step_records,
    write_and_commit,
)

# A real opencode step_finish line, verbatim shape from a worker log of 2026-10-10.
STEP_EVENT = {
    "type": "step_finish",
    "part": {"type": "step-finish",
             "tokens": {"total": 34433, "input": 3, "output": 121, "reasoning": 0,
                        "cache": {"write": 34309, "read": 17}},
             "cost": 0.0123},
}


def fake_runtime(key="t1", status="pass", final_verdict="PASS", attempts=1,
                 cost_usd=None, log_path=Path("/nonexistent.log"),
                 started=100.0, ended=160.0, ticket="", task_type="probe",
                 engine="opencode", model="openrouter/z-ai/glm-5.2"):
    return SimpleNamespace(
        task=SimpleNamespace(key=key, ticket=ticket, task_type=task_type,
                             engine=engine, model=model),
        status=status, final_verdict=final_verdict, attempts=attempts,
        cost_usd=cost_usd, log_path=log_path,
        started_at_monotonic=started, ended_at_monotonic=ended,
    )


class ConfigTests(unittest.TestCase):
    def test_absent_and_partial_config_is_inactive(self):
        self.assertFalse(load_reporting_config(None).active)
        self.assertFalse(load_reporting_config({"enabled": True}).active)
        self.assertFalse(load_reporting_config(
            {"enabled": True, "aios_repo": "/nonexistent-dir"}).active)
        self.assertFalse(load_reporting_config({"aios_repo": "/tmp"}).active,
                         "a repo without enabled=true must stay off")

    def test_both_switches_make_it_active(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = load_reporting_config({"enabled": True, "aios_repo": td})
            self.assertTrue(cfg.active)


class StepParsingTests(unittest.TestCase):
    def _log(self, td, lines):
        p = Path(td) / "w.log"
        p.write_text("\n".join(lines), encoding="utf-8")
        return p

    def test_measured_fields_carry_measured_provenance(self):
        with tempfile.TemporaryDirectory() as td:
            recs = parse_step_records(self._log(td, [
                "[ringer.py] attempt 1 started", json.dumps(STEP_EVENT)]))
            self.assertEqual(len(recs), 1)
            r = recs[0]
            self.assertEqual(r["cost_reported_usd"], 0.0123)
            self.assertEqual(r["tokens_in_reported"], 3)
            self.assertEqual(r["tokens_out_reported"], 121)
            self.assertEqual(r["tokens_cache_read_reported"], 17)
            self.assertEqual(r["provenance"]["cost_reported_usd"], "measured")
            self.assertIsNone(r["duration_s"], "per-step wall clock is absent, not 0")
            self.assertEqual(r["provenance"]["duration_s"], "absent")

    def test_plan_billed_log_yields_no_records_not_zeros(self):
        with tempfile.TemporaryDirectory() as td:
            recs = parse_step_records(self._log(td, [
                "[ringer.py] attempt 1 started", "plain codex text, no JSON events"]))
            self.assertEqual(recs, [])

    def test_untrusted_cost_values_become_absent_never_negative(self):
        bad = dict(STEP_EVENT)
        bad["part"] = {"tokens": {"input": 1}, "cost": -4.0}
        nan = dict(STEP_EVENT)
        nan["part"] = {"cost": float("nan")}
        with tempfile.TemporaryDirectory() as td:
            # json.dumps refuses nan by default; emulate a hostile log by hand
            p = Path(td) / "w.log"
            p.write_text(json.dumps(bad) + "\n"
                         + '{"type": "step_finish", "part": {"cost": NaN}}\n',
                         encoding="utf-8")
            recs = parse_step_records(p)
            self.assertEqual(len(recs), 2)
            self.assertIsNone(recs[0]["cost_reported_usd"],
                              "a negative cost is skipped, not refunded")
            self.assertIsNone(recs[1]["cost_reported_usd"], "NaN is skipped")

    def test_missing_log_is_empty_not_an_error(self):
        self.assertEqual(parse_step_records(Path("/no/such/file.log")), [])


class FragmentTests(unittest.TestCase):
    def build(self, runtimes, budget=None):
        return build_fragment(
            run_id="r-20261029T? test", run_name="probe", identity="claude-aios",
            started_at=datetime(2026, 10, 29, 12, 0, tzinfo=timezone.utc),
            runtimes=runtimes, repo=Path("/tmp/nexo"), budget_usd=budget)

    def test_fragment_is_a_full_bundle_with_the_1036_collections(self):
        frag = self.build([fake_runtime()])
        for key in ("report_id", "period", "projects", "quality_results",
                    "loop_results", "task_attempts", "run_tasks",
                    "run_tasks_failed", "task_retry_attempts", "run_snapshot"):
            self.assertIn(key, frag)
        self.assertEqual(frag["period"]["start"], frag["period"]["end"])
        self.assertEqual(frag["period"]["end"], "2026-10-29")

    def test_verdict_mapping_both_ways(self):
        frag = self.build([
            fake_runtime(key="ok", status="pass"),
            fake_runtime(key="bad", status="fail", final_verdict="FAIL"),
            fake_runtime(key="err", status="fail", final_verdict="ERROR"),
        ])
        verdicts = {q["subject"]: q["verdict"] for q in frag["quality_results"]}
        self.assertEqual(verdicts, {"ok": "pass", "bad": "fail", "err": "indeterminate"})
        self.assertEqual({t["task_key"] for t in frag["run_tasks_failed"]}, {"bad", "err"})
        self.assertEqual(len(frag["run_tasks"]), 3)

    def test_plan_billed_cost_is_null_never_zero(self):
        frag = self.build([fake_runtime(cost_usd=None)])
        row = frag["run_tasks"][0]
        self.assertIsNone(row["task_cost_usd"])
        self.assertEqual(row["provenance"]["task_cost_usd"], "absent")
        # and a measured zero stays a zero — the two are different facts
        frag = self.build([fake_runtime(cost_usd=0.0)])
        self.assertEqual(frag["run_tasks"][0]["task_cost_usd"], 0.0)
        self.assertEqual(frag["run_tasks"][0]["provenance"]["task_cost_usd"], "measured")

    def test_retry_spend_is_absent_not_apportioned(self):
        frag = self.build([fake_runtime(attempts=2)])
        self.assertEqual(len(frag["task_retry_attempts"]), 1)
        self.assertIsNone(frag["task_retry_attempts"][0]["cost_reported_usd"])
        frag = self.build([fake_runtime(attempts=1)])
        self.assertEqual(frag["task_retry_attempts"], [],
                         "no retries is a true empty list — a real 0, not absent")

    def test_run_span_and_budget(self):
        frag = self.build([fake_runtime(started=100.0, ended=160.0),
                           fake_runtime(key="t2", started=120.0, ended=220.0)],
                          budget=2.5)
        self.assertEqual(frag["run_snapshot"]["duration_s"], 120.0)
        self.assertEqual(frag["run_snapshot"]["budget_usd"], 2.5)
        frag = self.build([fake_runtime(started=None, ended=None)])
        self.assertIsNone(frag["run_snapshot"]["duration_s"],
                          "an unmeasured span is null, never 0")

    def test_subject_prefers_the_ticket(self):
        frag = self.build([fake_runtime(ticket="work#1038")])
        self.assertEqual(frag["quality_results"][0]["subject"], "work#1038")


class TransportTests(unittest.TestCase):
    def _repo(self, td, branch="main"):
        repo = Path(td) / "aios"
        repo.mkdir()
        subprocess.run(["git", "-C", str(repo), "init", "-q", "-b", branch], check=True)
        # CI runners have no global git identity; the emitter commits as whoever
        # runs ringer, so the fixture must carry its own identity like a real
        # operator's checkout does.
        subprocess.run(["git", "-C", str(repo), "config", "user.email", "t@t"], check=True)
        subprocess.run(["git", "-C", str(repo), "config", "user.name", "t"], check=True)
        subprocess.run(["git", "-C", str(repo), "commit", "--allow-empty", "-q",
                        "-m", "root"], check=True)
        return repo

    def _frag(self):
        return build_fragment(
            run_id="probe-1", run_name="probe", identity="t",
            started_at=datetime(2026, 10, 29, tzinfo=timezone.utc),
            runtimes=[fake_runtime()], repo=None, budget_usd=None)

    def test_on_main_it_commits(self):
        with tempfile.TemporaryDirectory() as td:
            repo = self._repo(td)
            line = write_and_commit(self._frag(), repo, "probe-1")
            self.assertIn("committed", line)
            log = subprocess.run(["git", "-C", str(repo), "log", "--oneline"],
                                 capture_output=True, text=True).stdout
            self.assertIn("work#1038", log)
            # push fails (no origin) but the commit survives and the line says so
            self.assertIn("push failed", line)

    def test_on_a_branch_it_writes_but_never_commits(self):
        with tempfile.TemporaryDirectory() as td:
            repo = self._repo(td, branch="somebody-elses-branch")
            line = write_and_commit(self._frag(), repo, "probe-1")
            self.assertIn("NOT committed", line)
            frag_file = repo / "scripts/reporting/inputs/fragments/2026-10-29-probe-1.json"
            self.assertTrue(frag_file.exists(), "the file always lands — data first")
            log = subprocess.run(["git", "-C", str(repo), "log", "--oneline"],
                                 capture_output=True, text=True).stdout
            self.assertNotIn("work#1038", log)

    def test_emit_never_raises_and_off_is_silent(self):
        self.assertEqual(emit_reporting_fragment(config=ReportingConfig(), run_id="x", run_name="x",
                              identity="t", started_at=datetime.now(timezone.utc),
                              runtimes=[], repo=None, budget_usd=None), "")
        with tempfile.TemporaryDirectory() as td:
            cfg = ReportingConfig(enabled=True, aios_repo=Path(td))
            line = emit_reporting_fragment(config=cfg, run_id="x", run_name="x", identity="t",
                        started_at=datetime.now(timezone.utc),
                        runtimes=[SimpleNamespace(task=None)],  # malformed on purpose
                        repo=None, budget_usd=None)
            self.assertIn("FAILED (run unaffected)", line)


if __name__ == "__main__":
    unittest.main(verbosity=2)
