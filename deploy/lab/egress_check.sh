#!/usr/bin/env bash
# Prove the lab segment has no path out (NEXT-07, §8.4 "Egress").
# Run inside the JEF-LAB-CORE VM. Every probe must FAIL; the script exits 0
# only when none of them reached anything. Output is the evidence.
set -u
targets=(
  "https://api.typesafe.ai"            # the real Jev
  "https://ai-gateway.vercel.sh"       # Vercel AI Gateway
  "https://api.anthropic.com"          # external LLM
  "https://huggingface.co"             # model hub (weights must be pre-staged)
  "https://pypi.org"                   # package index (deps must be pre-staged)
  "https://github.com"
  "http://1.1.1.1"
)
leaks=0
for t in "${targets[@]}"; do
  if curl -s -m 5 -o /dev/null "$t" 2>/dev/null; then
    echo "LEAK  $t reachable"; leaks=$((leaks + 1))
  else
    echo "ok    $t blocked"
  fi
done
if getent hosts api.typesafe.ai >/dev/null 2>&1; then
  echo "LEAK  DNS resolves external names"; leaks=$((leaks + 1))
else
  echo "ok    external DNS does not resolve"
fi
echo "---"
if [ "$leaks" -eq 0 ]; then echo "PASS: no egress from this segment"; exit 0; fi
echo "FAIL: $leaks path(s) out of the lab; fix the vSwitch/port-group ACL before any replay"; exit 1
