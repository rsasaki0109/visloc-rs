#!/usr/bin/env bash
# Run one command inside an otherwise fresh container and record its memory peak.
#
# The caller should bind-mount a writable directory and set
# VISLOC_METRICS_OUTPUT to a path in that directory. The process VmHWM is the
# primary RSS measurement; cgroup memory.peak is also recorded as a conservative
# whole-container cross-check.

set -u

if [[ $# -eq 0 ]]; then
  echo "usage: docker_process_metrics.sh COMMAND [ARG ...]" >&2
  exit 2
fi

metrics_output=${VISLOC_METRICS_OUTPUT:-}
poll_seconds=${VISLOC_METRICS_POLL_SECONDS:-0.05}
if [[ -z ${metrics_output} ]]; then
  echo "VISLOC_METRICS_OUTPUT must name a writable, bind-mounted path" >&2
  exit 2
fi

start_ns=$(date +%s%N)
"$@" &
child=$!
peak_process_hwm_kib=0

# Docker sends signals to this PID-1 wrapper. Forward graceful termination so
# an interrupted benchmark does not leave the measured child running until the
# container's hard-kill timeout. The child's status remains the phase status.
forward_signal() {
  kill "-$1" "${child}" 2>/dev/null || true
}
trap 'forward_signal TERM' TERM
trap 'forward_signal INT' INT
trap 'forward_signal HUP' HUP

while kill -0 "${child}" 2>/dev/null; do
  process_hwm_kib=$(awk '/^VmHWM:/ {print $2}' "/proc/${child}/status" 2>/dev/null || true)
  if [[ -n ${process_hwm_kib} ]] && (( process_hwm_kib > peak_process_hwm_kib )); then
    peak_process_hwm_kib=${process_hwm_kib}
  fi
  sleep "${poll_seconds}"
done

set +e
wait "${child}"
status=$?
set -e
end_ns=$(date +%s%N)
if [[ -r /sys/fs/cgroup/memory.peak ]]; then
  cgroup_peak_bytes=$(< /sys/fs/cgroup/memory.peak)
else
  # Direct host-side smoke tests do not necessarily run in a delegated cgroup.
  # A real Docker control always exposes this cgroup-v2 counter.
  cgroup_peak_bytes=0
fi

temporary="${metrics_output}.tmp-${BASHPID}"
{
  printf 'schema\tvisloc_docker_process_metrics_v1\n'
  printf 'status\t%s\n' "${status}"
  printf 'wall_ns\t%s\n' "$((end_ns - start_ns))"
  printf 'peak_process_hwm_kib\t%s\n' "${peak_process_hwm_kib}"
  printf 'cgroup_peak_bytes\t%s\n' "${cgroup_peak_bytes}"
  printf 'poll_seconds\t%s\n' "${poll_seconds}"
} > "${temporary}"
mv "${temporary}" "${metrics_output}"

exit "${status}"
