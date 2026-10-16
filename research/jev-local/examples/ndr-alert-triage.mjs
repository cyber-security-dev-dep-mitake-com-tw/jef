// NDR (network detection and response) alert triage with Jev: batch several alerts,
// one Jev request each, and sort them by severity × true-positive probability.
//   node examples/ndr-alert-triage.mjs
const BASE = process.env.JEV_BASE || 'http://127.0.0.1:4141/v1';
const alerts = [
  { id: 'ndr-1', summary: 'DNS: 1,240 TXT queries in 5 min from 10.8.3.21 to random subdomains of 8xk2.net, avg label length 52', asset: 'developer workstation', baseline: 'host normally makes <50 DNS queries/5 min' },
  { id: 'ndr-2', summary: 'SMB: 10.8.1.5 (file server) read 3.1 GB from \\\\FS01\\finance by svc_backup between 02:00 and 02:40', asset: 'backup service account', baseline: 'nightly backup job runs 02:00-03:00, typical volume 2-4 GB' },
  { id: 'ndr-3', summary: 'TLS: 10.8.3.44 → 45.142.212.100:8443, JA3 matches Sliver implant, 60-second beacon interval, 14 hours', asset: 'HR laptop', baseline: 'no prior connections to this ASN' },
  { id: 'ndr-4', summary: 'HTTP: internal scanner 10.8.0.9 probing 10.8.3.0/24 ports 22,80,443,3389', asset: 'vulnerability scanner (Nessus)', baseline: 'weekly scheduled scan, Thursdays 01:00' },
];
const questions = {
  true_positive: { type: 'noul', instructions: 'Is this alert malicious activity rather than expected or benign behaviour given the `baseline`?' },
  technique: { type: 'choice', instructions: 'Which technique best matches the `summary`?', criteria: { dns_tunneling: 'Data or C2 over DNS queries', c2_beacon: 'Periodic implant check-in to an external host', exfiltration: 'Bulk data leaving or staged for leaving', lateral_movement: 'Internal reconnaissance or movement between hosts', scheduled_job: 'Expected automation such as backups or scans', unknown: 'None of the above' } },
  severity: { type: 'score', instructions: 'How urgently should an analyst act?', criteria: ['Ignore', 'Review within a week', 'Review today', 'Respond within the hour', 'Respond immediately'] },
};
const rows = [];
for (const a of alerts) {
  const r = await fetch(`${BASE}/systemone`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ model: 'jev-latest', state: a, questions }) });
  const { answers } = await r.json();
  rows.push({ id: a.id, p_true: answers.true_positive.noul, technique: answers.technique.choice, conf: answers.technique.confidence, severity: answers.severity.score, priority: +(answers.true_positive.noul * answers.severity.score).toFixed(2) });
}
rows.sort((x, y) => y.priority - x.priority);
console.table(rows);
