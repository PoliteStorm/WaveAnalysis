EmailAgent — Send outreach email via SMTP

What it does
- Composes an email from `deliverables/emails/andrew_adamatzky.md` (or a custom markdown file) and sends via SMTP using env vars or CLI flags.
- Supports SSL (465) and STARTTLS (587), and file attachments (PDFs/images).

Usage
1) Dry run (no send):
```bash
python scripts/email_agent.py --dry-run | cat
```

2) Send (example: Gmail with app password):
```bash
SMTP_HOST=smtp.gmail.com SMTP_PORT=465 USE_SSL=1 \
SMTP_USER="you@gmail.com" SMTP_PASS="app_password" \
SENDER_EMAIL="you@gmail.com" SENDER_NAME="Joe Knowles" \
python scripts/email_agent.py \
  --attach /workspace/deliverables/one_pager/one_pager.pdf \
  --attach /workspace/deliverables/manuscript/preprint.pdf | cat
```

3) Proton (bridge or SMTP creds), Outlook, etc.: set host/port/user/pass accordingly.

Environment variables (overridable by CLI)
- SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS
- SENDER_EMAIL, SENDER_NAME
- RECIPIENT_EMAIL (default: andrew.adamatzky@uwe.ac.uk)
- SUBJECT (default taken from markdown)
- BODY_MD_FILE (default: deliverables/emails/andrew_adamatzky.md)
- ATTACH_PATHS (comma-separated; optional)
- USE_SSL ("1" to use SSL; else STARTTLS)

Notes
- This script does not create an email account; you must supply credentials.
- For Gmail, enable 2FA and use an app password.
- PDFs can be created with pandoc:
```bash
pandoc -s /workspace/deliverables/one_pager/one_pager.md -o /workspace/deliverables/one_pager/one_pager.pdf
pandoc -s /workspace/deliverables/manuscript/preprint.md -o /workspace/deliverables/manuscript/preprint.pdf
```
