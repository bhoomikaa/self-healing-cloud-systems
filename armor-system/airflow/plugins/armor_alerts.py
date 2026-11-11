"""Alerting for the ARMOR D1 DAG.

A red task in the Airflow UI is only an alert if someone is looking at the UI.
This turns a failure into something that leaves the machine, or at minimum
leaves a durable record that can be shown afterwards.

Three sinks, tried in order, and every one that is configured is used:

  log        always. A single clearly delimited block in the task log, so a
             failure is findable by grepping rather than by scrolling.
  file       always. One JSON object per alert under the alerts directory,
             which is what you show when asked to prove an alert fired, since
             task logs roll and the UI only shows the latest try.
  webhook    when ALERT_WEBHOOK_URL is set. Slack, Discord and Teams all accept
             a JSON POST, and a webhook URL is the only kind of notification
             endpoint that needs no account credentials in the image.
  email      when SMTP_HOST is set to something other than localhost. Airflow's
             own email_on_failure is left off because it fails silently when
             SMTP is unconfigured, which is how this DAG shipped with alerting
             that never sent anything.

Nothing here raises. An alerting path that can itself fail the task turns one
problem into two, so every sink is wrapped and failures are reported into the
log rather than propagated.

Self-test, which fires a real alert through the real code path:

    python -I armor_alerts.py --selftest
"""
from __future__ import annotations

import datetime as dt
import json
import os
import socket
import sys
import traceback
import urllib.error
import urllib.request

ALERT_DIR = os.environ.get("ARMOR_ALERT_DIR", "/opt/airflow/workspace/alerts")
WEBHOOK = os.environ.get("ALERT_WEBHOOK_URL", "").strip()
SMTP_HOST = os.environ.get("SMTP_HOST", "").strip()
SMTP_PORT = int(os.environ.get("SMTP_PORT", "25") or 25)
SMTP_FROM = os.environ.get("SMTP_FROM", "airflow@localhost")
ALERT_TO = [a for a in os.environ.get("ALERT_EMAIL_TO", "").split(",") if a.strip()]
TIMEOUT_S = 10


def _now():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_alert(context, severity="error"):
    """Flatten an Airflow task context into something a human can act on."""
    ti = context.get("task_instance")
    dag = context.get("dag")
    exc = context.get("exception")
    alert = {
        "schema": "armor.alert.v1",
        "severity": severity,
        "raised_utc": _now(),
        "host": socket.gethostname(),
        "dag_id": getattr(dag, "dag_id", None) or (ti.dag_id if ti else None),
        "task_id": ti.task_id if ti else None,
        "run_id": context.get("run_id"),
        "try_number": getattr(ti, "try_number", None),
        "max_tries": getattr(ti, "max_tries", None),
        "log_url": getattr(ti, "log_url", None),
        "exception_type": type(exc).__name__ if exc else None,
        "exception": str(exc)[:2000] if exc else None,
    }
    # A retry that will be tried again is different from a final failure, and
    # paging on the first of three attempts is how alerting gets muted.
    t, m = alert["try_number"], alert["max_tries"]
    alert["final"] = not (isinstance(t, int) and isinstance(m, int) and t <= m)
    return alert


def sink_log(alert, log=print):
    line = "=" * 72
    log(f"\n{line}\nARMOR ALERT [{alert['severity']}] {alert['dag_id']}.{alert['task_id']}")
    log(f"run     : {alert['run_id']}  try {alert['try_number']}/{alert['max_tries']}"
        f"  {'FINAL' if alert['final'] else 'will retry'}")
    if alert["exception"]:
        log(f"error   : {alert['exception_type']}: {alert['exception'][:500]}")
    if alert["log_url"]:
        log(f"log     : {alert['log_url']}")
    log(f"{line}\n")
    return True


def sink_file(alert, log=print):
    try:
        os.makedirs(ALERT_DIR, exist_ok=True)
        path = os.path.join(ALERT_DIR, "alerts.jsonl")
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(alert, sort_keys=True) + "\n")
        log(f"alert appended to {path}")
        return True
    except Exception as exc:                                      # noqa: BLE001
        log(f"alert file sink failed: {type(exc).__name__}: {exc}")
        return False


def sink_webhook(alert, log=print):
    if not WEBHOOK:
        log("webhook sink skipped: ALERT_WEBHOOK_URL is not set")
        return None
    text = (f"ARMOR {alert['severity'].upper()}: "
            f"{alert['dag_id']}.{alert['task_id']} "
            f"({'final' if alert['final'] else 'will retry'})\n"
            f"{alert['exception_type'] or 'no exception'}: "
            f"{(alert['exception'] or '')[:300]}")
    # "text" and "content" cover Slack/Teams and Discord respectively, so one
    # payload works for whichever URL is configured without a provider setting.
    payload = json.dumps({"text": text, "content": text, "alert": alert}).encode()
    req = urllib.request.Request(WEBHOOK, data=payload,
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
            log(f"webhook sink: HTTP {r.status}")
            return 200 <= r.status < 300
    except Exception as exc:                                      # noqa: BLE001
        log(f"webhook sink failed: {type(exc).__name__}: {exc}")
        return False


def sink_email(alert, log=print):
    # localhost:25 is the Airflow default and means "no mail server", which is
    # exactly the configuration that makes alerting look wired up while sending
    # nothing. Treat it as unconfigured rather than pretending.
    if not SMTP_HOST or SMTP_HOST in ("localhost", "127.0.0.1"):
        log("email sink skipped: SMTP_HOST is unset or localhost")
        return None
    if not ALERT_TO:
        log("email sink skipped: ALERT_EMAIL_TO is empty")
        return None
    import smtplib
    from email.message import EmailMessage
    msg = EmailMessage()
    msg["Subject"] = f"[ARMOR {alert['severity']}] {alert['dag_id']}.{alert['task_id']}"
    msg["From"] = SMTP_FROM
    msg["To"] = ", ".join(ALERT_TO)
    msg.set_content(json.dumps(alert, indent=1, sort_keys=True))
    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=TIMEOUT_S) as s:
            user, pw = os.environ.get("SMTP_USER"), os.environ.get("SMTP_PASSWORD")
            if user and pw:
                s.starttls()
                s.login(user, pw)
            s.send_message(msg)
        log(f"email sink: sent to {len(ALERT_TO)} recipient(s)")
        return True
    except Exception as exc:                                      # noqa: BLE001
        log(f"email sink failed: {type(exc).__name__}: {exc}")
        return False


def dispatch(alert, log=print):
    results = {"log": sink_log(alert, log), "file": sink_file(alert, log),
               "webhook": sink_webhook(alert, log), "email": sink_email(alert, log)}
    delivered = [k for k, v in results.items() if v is True]
    skipped = [k for k, v in results.items() if v is None]
    failed = [k for k, v in results.items() if v is False]
    log(f"alert delivered via {delivered or 'nothing'}"
        + (f"; skipped {skipped}" if skipped else "")
        + (f"; FAILED {failed}" if failed else ""))
    return results


def on_failure(context):
    """Airflow on_failure_callback. Must never raise."""
    try:
        dispatch(build_alert(context, "error"))
    except Exception:                                             # noqa: BLE001
        print("alerting itself failed:\n" + traceback.format_exc())


def on_retry(context):
    try:
        dispatch(build_alert(context, "warning"))
    except Exception:                                             # noqa: BLE001
        print("alerting itself failed:\n" + traceback.format_exc())


def on_sla_miss(*args, **kwargs):
    try:
        dispatch({"schema": "armor.alert.v1", "severity": "warning", "raised_utc": _now(),
                  "host": socket.gethostname(), "dag_id": "armor_d1_data_pipeline",
                  "task_id": "(sla)", "run_id": None, "try_number": None, "max_tries": None,
                  "log_url": None, "exception_type": "SLAMiss",
                  "exception": "a task missed its SLA", "final": True})
    except Exception:                                             # noqa: BLE001
        print("alerting itself failed:\n" + traceback.format_exc())


def _selftest():
    print("firing a real alert through the real sinks\n")
    alert = {"schema": "armor.alert.v1", "severity": "error", "raised_utc": _now(),
             "host": socket.gethostname(), "dag_id": "armor_d1_data_pipeline",
             "task_id": "selftest", "run_id": "selftest", "try_number": 1, "max_tries": 1,
             "log_url": None, "exception_type": "SelfTest",
             "exception": "deliberate alert from armor_alerts.py --selftest", "final": True}
    results = dispatch(alert)
    print()
    print(f"alert directory: {ALERT_DIR}")
    print(f"webhook configured: {'yes' if WEBHOOK else 'no (set ALERT_WEBHOOK_URL)'}")
    print(f"email configured  : {'yes' if (SMTP_HOST and SMTP_HOST not in ('localhost','127.0.0.1') and ALERT_TO) else 'no'}")
    # The log and file sinks have no external dependency, so they are the two
    # that must work anywhere; the others are configuration-dependent.
    return 0 if results["log"] and results["file"] else 1


if __name__ == "__main__":
    sys.exit(_selftest() if "--selftest" in sys.argv else _selftest())
