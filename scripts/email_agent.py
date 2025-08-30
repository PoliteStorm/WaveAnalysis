#!/usr/bin/env python3
"""
EmailAgent: Compose and send an outreach email via SMTP.

Defaults to emailing Prof. Andrew Adamatzky using the prepared draft at
deliverables/emails/andrew_adamatzky.md, but allows full override via
environment variables or CLI flags.

Environment variables (CLI flags override):
  SMTP_HOST            e.g., smtp.gmail.com, smtp.protonmail.ch, smtp.office365.com
  SMTP_PORT            e.g., 465 (SSL) or 587 (STARTTLS)
  SMTP_USER            SMTP username (often your email address)
  SMTP_PASS            SMTP password or app password
  SENDER_EMAIL         From address (must match SMTP account policy)
  SENDER_NAME          Human-friendly name (default: value before '@' in SENDER_EMAIL)
  RECIPIENT_EMAIL      To address (default: andrew.adamatzky@uwe.ac.uk)
  SUBJECT              Email subject (if not set, derived from draft)
  BODY_MD_FILE         Path to markdown body file (default: deliverables/emails/andrew_adamatzky.md)
  ATTACH_PATHS         Comma-separated list of file paths to attach
  USE_SSL              "1" for SSL (port 465), otherwise STARTTLS

Usage examples:
  # Dry run: print the composed email
  python scripts/email_agent.py --dry-run

  # Send using env vars
  SMTP_HOST=smtp.gmail.com SMTP_PORT=465 USE_SSL=1 \
  SMTP_USER=you@gmail.com SMTP_PASS=app_password \
  SENDER_EMAIL=you@gmail.com SENDER_NAME="Joe Knowles" \
  python scripts/email_agent.py

  # Override recipient and add attachments
  python scripts/email_agent.py --to someone@example.com \
    --attach deliverables/one_pager/one_pager.pdf \
    --attach deliverables/manuscript/preprint.pdf
"""

from __future__ import annotations

import argparse
import os
import sys
import mimetypes
import ssl
import smtplib
from email.message import EmailMessage
from typing import List


DEFAULT_DRAFT_PATH = \
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'deliverables', 'emails', 'andrew_adamatzky.md'))


def read_text(path: str) -> str:
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def extract_subject_and_body_from_markdown(md_text: str) -> tuple[str, str]:
    lines = md_text.splitlines()
    subject = ""
    body_lines: List[str] = []
    for idx, line in enumerate(lines):
        if line.lower().startswith("subject:"):
            subject = line.split(":", 1)[1].strip()
            # skip this line in body
            continue
        body_lines.append(line)
    body = "\n".join(body_lines).strip()
    # remove leading empty lines
    while body.startswith("\n\n"):
        body = body[2:]
    return subject, body


def to_html_fallback(md_text: str) -> str:
    """Very simple Markdown to HTML fallback (no external deps).
    Preserves paragraphs and line breaks; does not aim for full markdown.
    """
    import html
    escaped = html.escape(md_text)
    # Preserve paragraphs: split on double newline
    paragraphs = [p.replace("\n", "<br>\n") for p in escaped.split("\n\n")]
    html_body = "<html><body>" + "\n\n".join(f"<p>{p}</p>" for p in paragraphs) + "</body></html>"
    return html_body


def attach_files(msg: EmailMessage, paths: List[str]) -> None:
    for path in paths:
        if not path:
            continue
        abspath = os.path.abspath(path)
        if not os.path.isfile(abspath):
            continue
        ctype, encoding = mimetypes.guess_type(abspath)
        if ctype is None or encoding is not None:
            ctype = 'application/octet-stream'
        maintype, subtype = ctype.split('/', 1)
        with open(abspath, 'rb') as f:
            data = f.read()
        msg.add_attachment(data, maintype=maintype, subtype=subtype, filename=os.path.basename(abspath))


def compose_email(subject: str, body_md: str, sender_email: str, sender_name: str, recipient_email: str, attachments: List[str]) -> EmailMessage:
    msg = EmailMessage()
    msg['From'] = f"{sender_name} <{sender_email}>"
    msg['To'] = recipient_email
    msg['Subject'] = subject

    # Plain and HTML alternatives
    msg.set_content(body_md)
    msg.add_alternative(to_html_fallback(body_md), subtype='html')

    attach_files(msg, attachments)
    return msg


def send_email(msg: EmailMessage, host: str, port: int, username: str, password: str, use_ssl: bool) -> None:
    if use_ssl:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(host, port, context=context) as server:
            if username and password:
                server.login(username, password)
            server.send_message(msg)
    else:
        with smtplib.SMTP(host, port) as server:
            server.ehlo()
            server.starttls(context=ssl.create_default_context())
            server.ehlo()
            if username and password:
                server.login(username, password)
            server.send_message(msg)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EmailAgent: compose and send an outreach email via SMTP")
    p.add_argument('--to', dest='to', default=os.getenv('RECIPIENT_EMAIL', 'andrew.adamatzky@uwe.ac.uk'))
    p.add_argument('--from', dest='from_', default=os.getenv('SENDER_EMAIL'))
    p.add_argument('--from-name', dest='from_name', default=os.getenv('SENDER_NAME'))
    p.add_argument('--smtp-host', dest='smtp_host', default=os.getenv('SMTP_HOST'))
    p.add_argument('--smtp-port', dest='smtp_port', type=int, default=int(os.getenv('SMTP_PORT') or 0))
    p.add_argument('--smtp-user', dest='smtp_user', default=os.getenv('SMTP_USER'))
    p.add_argument('--smtp-pass', dest='smtp_pass', default=os.getenv('SMTP_PASS'))
    p.add_argument('--use-ssl', dest='use_ssl', action='store_true', default=os.getenv('USE_SSL') == '1')
    p.add_argument('--subject', dest='subject', default=os.getenv('SUBJECT'))
    p.add_argument('--body-md-file', dest='body_md_file', default=os.getenv('BODY_MD_FILE') or DEFAULT_DRAFT_PATH)
    p.add_argument('--attach', dest='attachments', action='append', default=[])
    p.add_argument('--dry-run', dest='dry_run', action='store_true', help='Do not send; print composed message')
    return p.parse_args()


def resolve_sender_name(sender_email: str, sender_name: str | None) -> str:
    if sender_name:
        return sender_name
    if sender_email and '@' in sender_email:
        return sender_email.split('@', 1)[0]
    return 'Researcher'


def main() -> int:
    args = parse_args()

    # Load body and subject from markdown draft
    try:
        md_text = read_text(args.body_md_file)
    except Exception as exc:
        print(f"ERROR: failed to read body markdown file: {args.body_md_file}: {exc}", file=sys.stderr)
        return 2
    draft_subject, body_md = extract_subject_and_body_from_markdown(md_text)

    subject = args.subject or draft_subject or "Collaboration/feedback on √t-transformed fungal electrophysiology"

    sender_email = getattr(args, 'from_')
    sender_name = resolve_sender_name(sender_email or '', args.from_name)

    if not sender_email:
        print("ERROR: SENDER_EMAIL is required (set env SENDER_EMAIL or pass --from)", file=sys.stderr)
        return 2

    # Resolve SMTP settings
    smtp_host = args.smtp_host or ''
    smtp_port = args.smtp_port or (465 if args.use_ssl else 587)
    smtp_user = args.smtp_user or ''
    smtp_pass = args.smtp_pass or ''
    use_ssl = bool(args.use_ssl)

    if not smtp_host:
        print("ERROR: SMTP_HOST is required (set env SMTP_HOST or pass --smtp-host)", file=sys.stderr)
        return 2

    # Default attachments: one-pager and preprint if they exist
    default_attach_candidates = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'deliverables', 'one_pager', 'one_pager.pdf')),
        os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'deliverables', 'manuscript', 'preprint.pdf')),
    ]
    attachments = [p for p in (args.attachments or []) if p]
    for cand in default_attach_candidates:
        if os.path.isfile(cand) and cand not in attachments:
            attachments.append(cand)

    # Compose
    msg = compose_email(subject=subject,
                        body_md=body_md,
                        sender_email=sender_email,
                        sender_name=sender_name,
                        recipient_email=args.to,
                        attachments=attachments)

    if args.dry_run:
        # Print headers and a snippet
        print("From:", msg['From'])
        print("To:", msg['To'])
        print("Subject:", msg['Subject'])
        print("--- BODY (text/plain) ---")
        print(msg.get_body(preferencelist=('plain',)).get_content())
        print("--- ATTACHMENTS ---")
        for part in msg.iter_attachments():
            print(part.get_filename())
        return 0

    try:
        send_email(msg, host=smtp_host, port=smtp_port, username=smtp_user, password=smtp_pass, use_ssl=use_ssl)
        print("OK: message sent")
        return 0
    except Exception as exc:
        print(f"ERROR: sending failed: {exc}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())

