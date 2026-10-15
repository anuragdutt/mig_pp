"""
Sends one email through Gmail SMTP. Used by run_measure_constants.sbatch at job start and end.

    python3 notify_mail.py "<subject>" [file whose last lines go in the body]

Setup, once, on the cluster (the password never goes on a command line):
    1. Google account -> Security -> 2-Step Verification on -> App passwords -> create one.
    2. printf '%s' '<16-char app password>' > ~/.mail_app_password && chmod 600 ~/.mail_app_password
    3. Set CONFIG below, then test from the login node:
       python3 mig_pp/milp/notify_mail.py "test from seawulf"
"""

import os
import smtplib
import socket
import ssl
import sys
from email.message import EmailMessage


def tail(path, n):
    try:
        with open(path, errors="replace") as f:
            return "".join(f.readlines()[-n:])
    except OSError as e:
        return f"(could not read {path}: {e})"


def main():
    CONFIG = dict(
        sender="limarkdcunha@gmail.com",  # the Gmail account the app password belongs to
        to="limarkdcunha@gmail.com",
        password_file="~/.mail_app_password",
        smtp_host="smtp.gmail.com",
        smtp_port=465,
        tail_lines=60,
    )
    c = CONFIG

    subject = sys.argv[1] if len(sys.argv) > 1 else "seawulf job"
    body = f"host {socket.gethostname()}, job {os.environ.get('SLURM_JOB_ID', '-')}\n"
    if len(sys.argv) > 2:
        body += f"\nlast {c['tail_lines']} lines of {sys.argv[2]}:\n\n" + tail(sys.argv[2], c["tail_lines"])

    with open(os.path.expanduser(c["password_file"])) as f:
        password = f.read().strip()

    msg = EmailMessage()
    msg["From"], msg["To"], msg["Subject"] = c["sender"], c["to"], subject
    msg.set_content(body)
    with smtplib.SMTP_SSL(c["smtp_host"], c["smtp_port"], context=ssl.create_default_context(),
                          timeout=30) as s:
        s.login(c["sender"], password)
        s.send_message(msg)
    print(f"mail sent to {c['to']}: {subject}")


if __name__ == "__main__":
    main()
