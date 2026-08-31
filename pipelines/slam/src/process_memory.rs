//! Small, opt-in process-memory diagnostics for long-running SfM/BA jobs.
//!
//! The mapper normally has no reason to read process state.  When
//! `VISLOC_SFM_MEMORY=1` is set, the helpers below sample Linux
//! `/proc/self/status` at phase boundaries so a benchmark can distinguish
//! retained model state from a solver's transient peak.  Missing `/proc` (for
//! example on non-Linux targets) is intentionally a no-op.

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct ProcessMemory {
    pub(crate) vm_size_kb: Option<u64>,
    pub(crate) vm_peak_kb: Option<u64>,
    pub(crate) rss_kb: Option<u64>,
    pub(crate) hwm_kb: Option<u64>,
    pub(crate) anon_rss_kb: Option<u64>,
    pub(crate) file_rss_kb: Option<u64>,
    pub(crate) shmem_rss_kb: Option<u64>,
}

fn status_value(line: &str) -> Option<(&str, u64)> {
    let (label, value) = line.split_once(':')?;
    let value = value.trim().strip_suffix(" kB")?.trim().parse().ok()?;
    Some((label, value))
}

fn parse_status(contents: &str) -> ProcessMemory {
    let mut memory = ProcessMemory::default();
    for line in contents.lines() {
        let Some((label, value)) = status_value(line) else {
            continue;
        };
        match label {
            "VmSize" => memory.vm_size_kb = Some(value),
            "VmPeak" => memory.vm_peak_kb = Some(value),
            "VmRSS" => memory.rss_kb = Some(value),
            "VmHWM" => memory.hwm_kb = Some(value),
            "RssAnon" => memory.anon_rss_kb = Some(value),
            "RssFile" => memory.file_rss_kb = Some(value),
            "RssShmem" => memory.shmem_rss_kb = Some(value),
            _ => {}
        }
    }
    memory
}

pub(crate) fn enabled() -> bool {
    std::env::var_os("VISLOC_SFM_MEMORY").is_some()
}

pub(crate) fn snapshot() -> Option<ProcessMemory> {
    let contents = std::fs::read_to_string("/proc/self/status").ok()?;
    let memory = parse_status(&contents);
    (memory.rss_kb.is_some() || memory.hwm_kb.is_some()).then_some(memory)
}

/// Emit a one-line, machine-readable sample.  This is deliberately gated at
/// the call boundary so normal SfM runs do not perform filesystem I/O.
pub(crate) fn log(stage: &str) {
    if !enabled() {
        return;
    }
    let Some(memory) = snapshot() else {
        eprintln!("sfm-memory: stage={stage} unavailable=1");
        return;
    };
    eprintln!(
        "sfm-memory: stage={stage} vm_size_kb={:?} vm_peak_kb={:?} rss_kb={:?} hwm_kb={:?} anon_rss_kb={:?} file_rss_kb={:?} shmem_rss_kb={:?}",
        memory.vm_size_kb,
        memory.vm_peak_kb,
        memory.rss_kb,
        memory.hwm_kb,
        memory.anon_rss_kb,
        memory.file_rss_kb,
        memory.shmem_rss_kb,
    );
}

#[cfg(test)]
mod tests {
    use super::{parse_status, status_value, ProcessMemory};

    #[test]
    fn parses_linux_status_memory_fields_and_ignores_other_units() {
        let status = "VmPeak:\t42 kB\nVmSize: 40 kB\nVmRSS: 30 kB\nVmHWM: 35 kB\nRssAnon: 20 kB\nRssFile: 9 kB\nRssShmem: 1 kB\nThreads: 4\n";
        assert_eq!(
            parse_status(status),
            ProcessMemory {
                vm_size_kb: Some(40),
                vm_peak_kb: Some(42),
                rss_kb: Some(30),
                hwm_kb: Some(35),
                anon_rss_kb: Some(20),
                file_rss_kb: Some(9),
                shmem_rss_kb: Some(1),
            }
        );
    }

    #[test]
    fn malformed_status_lines_are_ignored() {
        assert_eq!(status_value("VmRSS: 12 MB"), None);
        assert_eq!(status_value("VmRSS: nope kB"), None);
        assert_eq!(
            parse_status("VmRSS: nope kB\nVmSize: 7 kB\n"),
            ProcessMemory {
                vm_size_kb: Some(7),
                ..ProcessMemory::default()
            }
        );
    }
}
