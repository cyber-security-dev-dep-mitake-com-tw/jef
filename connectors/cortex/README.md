# Cortex analyzers and responders for JEF

Two analyzers and one responder for [Cortex](https://github.com/TheHive-Project/Cortex),
the analysis engine behind TheHive.

| Flavor | Kind | What it does |
|---|---|---|
| `JEF_Scene` | analyzer | Runs a JEF scene over the observable and returns the full decision trace as taxonomies, plus the trace in the report. |
| `JEF_Question` | analyzer | Asks one ad-hoc typed question — a choice, a score or a yes/no — without needing a scene. |
| `JEF_Triage` | responder | Runs a scene over a TheHive case or alert and writes the verdict back as tags. |

## Why the taxonomies look like that

Cortex taxonomies are the short verdict chips analysts actually read, so the
level is chosen from what the decision *means*, not from the score:

- `safe` when the scene closed the case
- `suspicious` when it wants a human
- `malicious` when it paged on-call
- `info` otherwise

A second taxonomy carries `p_correct` when the server could supply it, and says
`unknown` when it could not. Those are different, and an analyst deciding
whether to trust an automated verdict needs to see which one they are looking at.

## Install

```bash
cp -r connectors/cortex/JEF /opt/Cortex-Analyzers/analyzers/
pip install -r /opt/Cortex-Analyzers/analyzers/JEF/requirements.txt
```

Then configure `url` (your JEF server) and `scene` in the Cortex organisation
settings.

## A note on automation

If the JEF server has no calibration loaded, every scene routes to a human by
design — `p_correct` is unavailable and the conformal set excludes nothing, so
no automating gate can fire. The analyzer surfaces that as a taxonomy rather
than letting it look like a model that happens to be cautious today.
