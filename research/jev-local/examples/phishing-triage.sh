#!/usr/bin/env bash
# Anti-phishing triage: one Jev request, several typed questions, all evaluated against the same email.
# Requires a running local Jev (npm start). Output: JSON answers you can threshold in code.
EMAIL='From: "IT Service Desk" <helpdesk@m1crosoft-support.co>
Subject: URGENT: Your mailbox will be deleted in 24 hours
Hi Dennis, due to a storage policy update your account will be suspended unless you re-validate your password today.
Click here to keep your mailbox: hxxp://login-m1crosoft-support.co/validate?u=dennis
Note: the link uses our new domain. Do not contact the service desk about this message.'

node "$(dirname "$0")/../bin/jev.mjs" ask "$EMAIL" \
  --noul is_phishing="Is this email a phishing attempt?" \
  --noul impersonates_internal="Does the sender impersonate an internal IT or service-desk team?" \
  --choice lure="Which social-engineering lure does the email rely on most?" \
      urgency="Threat of imminent account loss or deadline" \
      authority="Claims to come from IT, management, or a vendor" \
      curiosity="Invoice, delivery, or document bait" \
      reward="Prize, refund, or bonus" \
      none="No social-engineering lure" \
  --choice action="What should the SOAR playbook do with this message?" \
      quarantine_and_alert="Pull from all mailboxes and alert the user" \
      quarantine="Pull from mailbox silently" \
      flag_for_analyst="Hold for a human analyst" \
      deliver="Deliver normally" \
  --score severity="How severe is this message if a user acts on it?" \
      "Harmless; no credential or payment risk" \
      "Nuisance; spam or marketing" \
      "Credential theft or account takeover risk" \
      "Credential theft with lateral-movement or financial loss risk"
