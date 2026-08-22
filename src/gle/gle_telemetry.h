// ============================================================================
// gle_telemetry.h — Gauntlet Loop Enhanced telemetry (TRANSCRIPT.md PART-J)
// ----------------------------------------------------------------------------
// Append-only JSONL event stream with an FNV-1a hash chain.
// Design rules (PART-J.1): append-only, evidence-first, machine-readable,
// zero-dependency (std only), cheap (<1ms/event).
// ============================================================================
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace gle {

// Event types — TRANSCRIPT.md PART-J.3 enum table.
enum class EventType {
    Unknown,
    RunStarted,
    BarFrozen,
    UnitSplit,
    BuildDone,
    CriticVerdict,
    ArbiterRuling,
    AnticheatSweep,
    BuildResult,
    TestResult,
    BenchSnapshot,
    RegressionCheck,
    ClaimVerified,
    PhaseExit,
    PanicStop,
    RunEnded,
    AgentStarted,
    InvalidRound
};

const char* event_type_name(EventType t);   // "RUN_STARTED" etc.

struct Event {
    std::string ts;          // ISO8601 local time with offset
    std::string run_id;
    std::string phase;
    std::string task;
    EventType type;
    std::string actor;
    std::string verdict;     // PASS/FAIL/... ("-"/empty when N/A)
    std::string evidence;    // artifact path or command-output reference (MANDATORY for verdicts)
    std::string numbers;     // raw JSON object string, e.g. {"decode_us":1820}
    std::string prev_hash;   // hex of previous event's hash ("0" for genesis)
    std::string hash;        // hex FNV-1a(prev_hash + canonical payload)
};

// Serialize one event as a single JSON line (no trailing newline).
std::string to_json_line(const Event& e);

class TelemetryWriter {
public:
    // Opens/creates `path` and appends a genesis-consistent stream.
    // If the file already exists, the chain continues from its last hash.
    explicit TelemetryWriter(const std::string& path);
    ~TelemetryWriter();

    // Appends one event: fills ts / prev_hash / hash automatically.
    // Returns the stored hash for the caller's log.
    std::string append(const Event& e);

    // Flush + close. Append-only discipline: no truncate, no rewrite.
    void close();

private:
    struct Impl;
    Impl* impl_;
};

struct ChainReport {
    size_t lines_read = 0;       // total non-empty lines
    size_t events_parsed = 0;    // valid JSON events
    size_t skipped_lines = 0;    // unparseable lines (partial tail tolerated)
    bool truncated_tail = false; // last line was partial/corrupt
    size_t first_bad_line = 0;   // 1-based line where chain breaks (0 = intact)
    bool chain_valid = true;
};

class TelemetryReader {
public:
    explicit TelemetryReader(const std::string& path);

    // Parses the stream; tolerates a partial final line (crash mid-write),
    // flags it via ChainReport::truncated_tail.
    const std::vector<Event>& events() const { return events_; }
    const ChainReport& report() const { return report_; }

    // Re-walks prev_hash/hash chain; sets first_bad_line on mismatch.
    ChainReport verify_chain() const;

private:
    std::vector<Event> events_;
    ChainReport report_;
};

} // namespace gle
