{{- define "jef.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "jef.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := default .Chart.Name .Values.nameOverride -}}
{{- if contains $name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{- define "jef.labels" -}}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" }}
{{ include "jef.selectorLabels" . }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}

{{- define "jef.selectorLabels" -}}
app.kubernetes.io/name: {{ include "jef.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{/*
Refuse to render with the test backbone.

'hashing' returns well-formed answers with no semantics, so a deployment using
it fails as confident nonsense rather than as an error. Better to break the
install.
*/}}
{{- define "jef.validateBackbone" -}}
{{- if eq .Values.jef.backbone "hashing" -}}
{{- fail "jef.backbone is 'hashing', the deterministic test backbone. It returns well-formed answers with no meaning. Set a real model id." -}}
{{- end -}}
{{- end -}}

{{/*
Warn, loudly and at install time, when threads meets or exceeds the CPU limit.

Oversubscribing the intra-op pool on a CPU-only node degrades p99 rather than
improving throughput, because the threads contend on the same caches.
*/}}
{{- define "jef.validateThreads" -}}
{{- $limit := .Values.resources.limits.cpu | toString | replace "m" "" | float64 -}}
{{- if gt (.Values.resources.limits.cpu | toString | contains "m" | ternary 0.0 (.Values.resources.limits.cpu | float64)) 0.0 -}}
{{- if ge (.Values.jef.threads | float64) (.Values.resources.limits.cpu | float64) -}}
{{- fail (printf "jef.threads (%v) must be below resources.limits.cpu (%v): the event loop and the OS need headroom, and oversubscribing the intra-op pool worsens p99." .Values.jef.threads .Values.resources.limits.cpu) -}}
{{- end -}}
{{- end -}}
{{- end -}}
