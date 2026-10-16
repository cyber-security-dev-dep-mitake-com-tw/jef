// SOAR decision chain: a playbook where each step is a Jev decision gated on confidence.
// Code stays in control; Jev answers narrow typed questions. Run with local Jev started:
//   node examples/soar-decision-chain.mjs
const BASE = process.env.JEV_BASE || 'http://127.0.0.1:4141/v1';

async function jev(state, questions) {
  const r = await fetch(`${BASE}/systemone`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ model: 'jev-latest', state, questions }) });
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return (await r.json()).answers;
}

// An alert as a SOAR platform would hand it over: structured state, not prose.
const alert = {
  source: 'EDR',
  host: 'FIN-LAPTOP-07',
  user: 'a.chen',
  detection: 'powershell.exe spawned from WINWORD.EXE with -EncodedCommand; outbound TLS to 185.220.101.4:443 within 3s',
  process_tree: ['explorer.exe', 'WINWORD.EXE (invoice_Q3.docm)', 'powershell.exe -nop -w hidden -enc JABjAGwAaQBlAG4AdAA...'],
  threat_intel: { '185.220.101.4': 'known C2 (Cobalt Strike), first seen 2026-09-30' },
  asset_criticality: 'high (finance)',
  recent_tickets: ['Change window: payroll run tonight 22:00'],
};

// Step 1: triage
const triage = await jev(alert, {
  true_positive: { type: 'noul', instructions: 'Is this alert a true positive (actual malicious activity) rather than benign admin or user behaviour?' },
  stage: { type: 'choice', instructions: 'Which attack stage does the evidence most directly show?', criteria: { initial_access: 'Phishing document opened, macro executed', execution: 'Malicious command or script running', c2: 'Beaconing or command-and-control traffic', lateral_movement: 'Access to other hosts or credentials', exfiltration: 'Data leaving the environment', benign: 'No malicious stage' } },
  severity: { type: 'score', instructions: 'How severe is this incident for the organisation?', criteria: ['Informational', 'Low: contained nuisance', 'Medium: single host compromise', 'High: active C2 on a sensitive asset', 'Critical: confirmed spread or data loss'] },
});
console.log('step 1 triage:', JSON.stringify(triage));

if (triage.true_positive.noul < 0.5) { console.log('→ close as false positive'); process.exit(0); }
if (triage.true_positive.noul < 0.8) { console.log('→ uncertain: route to analyst queue'); process.exit(0); }

// Step 2: containment decision, with the business context in the state
const containment = await jev({ ...alert, triage: { stage: triage.stage.choice, severity: triage.severity.score } }, {
  action: { type: 'choice', instructions: 'Which containment action should the playbook take now?', criteria: { isolate_host: 'Network-isolate the endpoint via EDR', kill_process_only: 'Terminate the process tree, keep host online', block_c2_only: 'Block the destination at the firewall only', monitor: 'Increase logging, no active containment' } },
  disable_user: { type: 'noul', instructions: 'Should the user account be disabled immediately?', criteria: { true: 'Credential theft or account misuse is likely', false: 'Account misuse is unlikely; disabling would mostly disrupt business' } },
  wait_for_change_window: { type: 'noul', instructions: 'Does the pending `recent_tickets` change window justify delaying containment?' },
});
console.log('step 2 containment:', JSON.stringify(containment));

const act = containment.action;
if (act.confidence < 0.4) console.log(`→ containment unclear (confidence ${act.confidence}); page on-call for a decision`);
else console.log(`→ execute ${act.choice} (p=${act.probabilities[act.choice]}), disable_user=${containment.disable_user.noul >= 0.7}, delay_for_change_window=${containment.wait_for_change_window.noul >= 0.5}`);

// Step 3: escalation and communications
const comms = await jev({ ...alert, decision: act.choice }, {
  notify: { type: 'choice', instructions: 'Who must be notified now?', criteria: { ciso_and_legal: 'Executive and legal notification required', soc_lead: 'SOC lead only', it_ops: 'IT operations for the affected host', none: 'No notification beyond the ticket' } },
  regulatory_clock: { type: 'noul', instructions: 'Could this incident start a regulatory breach-notification clock?' },
});
console.log('step 3 comms:', JSON.stringify(comms));
