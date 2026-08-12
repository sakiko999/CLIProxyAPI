package helps

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/router-for-me/CLIProxyAPI/v7/internal/config"
	"github.com/router-for-me/CLIProxyAPI/v7/sdk/translator"
	"github.com/tidwall/gjson"
)

// fixtureLogPath is the real request log that produced the reasoning-only 503
// class. Its REQUEST BODY is a Claude /v1/messages payload whose tool history
// carries the pollution chain: the assistant tool turns got the placeholder
// from the request-side repair, and the upstream echoed it back as thinking.
// Resolved from the repository root so the test runs regardless of working dir.
const fixtureLogPath = "temp/report/error-v1-messages-2026-08-11T220927-49b196b9.log"

// extractFixtureRequestBody pulls the single-line REQUEST BODY JSON out of a
// request-log file (=== REQUEST BODY === section).
func extractFixtureRequestBody(t *testing.T, path string) []byte {
	t.Helper()
	// Resolve relative to the repository root (walk up from the package dir).
	full := path
	if !strings.HasPrefix(path, "/") {
		if cwd, err := os.Getwd(); err == nil {
			full = filepath.Join(cwd, "..", "..", "..", "..", path)
		}
	}
	raw, err := os.ReadFile(full)
	if err != nil {
		t.Skipf("fixture log not present: %v", err)
	}
	marker := "=== REQUEST BODY ===\n"
	idx := strings.Index(string(raw), marker)
	if idx < 0 {
		t.Fatalf("fixture log has no REQUEST BODY section")
	}
	bodyStart := idx + len(marker)
	rest := string(raw)[bodyStart:]
	lineEnd := strings.IndexByte(rest, '\n')
	if lineEnd < 0 {
		lineEnd = len(rest)
	}
	return []byte(rest[:lineEnd])
}

// countPlaceholderReasoning counts assistant tool-call turns whose
// reasoning_content is exactly the placeholder.
func countPlaceholderReasoning(t *testing.T, payload []byte) int {
	t.Helper()
	count := 0
	for _, msg := range gjson.GetBytes(payload, "messages").Array() {
		if msg.Get("role").String() != "assistant" || !msg.Get("tool_calls").IsArray() {
			continue
		}
		if IsReasoningUnavailable(msg.Get("reasoning_content").String()) {
			count++
		}
	}
	return count
}

// countRealReasoning counts assistant tool-call turns whose reasoning_content
// is non-empty and not the placeholder.
func countRealReasoning(t *testing.T, payload []byte) int {
	t.Helper()
	count := 0
	for _, msg := range gjson.GetBytes(payload, "messages").Array() {
		if msg.Get("role").String() != "assistant" || !msg.Get("tool_calls").IsArray() {
			continue
		}
		rc := strings.TrimSpace(msg.Get("reasoning_content").String())
		if rc != "" && !IsReasoningUnavailable(rc) {
			count++
		}
	}
	return count
}

// TestRepairOpenAICompatReasoningContentFixtureRegression replays the real
// 503-producing request through the same claude -> openai translation + repair
// chain the executor uses, and asserts that the B3 source-order change moves
// tool turns from placeholder reasoning to real content/reasoning text. It is
// a regression guard, not a unit test of the fixture data.
func TestRepairOpenAICompatReasoningContentFixtureRegression(t *testing.T) {
	body := extractFixtureRequestBody(t, fixtureLogPath)

	translated := TranslateRequestWithAPIKeyModelCompatibility(t.Context(), nil, &config.Config{}, translator.FormatClaude, translator.FormatOpenAI, "deepseek-v4-flash", body, false, false)
	if len(translated) == 0 {
		t.Fatal("claude->openai translation produced empty payload")
	}
	repaired := RepairOpenAICompatReasoningContent("claude", "https://opencode.ai/zen/go/v1", "deepseek-v4-flash", body, translated)

	placeholderBefore := countPlaceholderReasoning(t, translated)
	placeholderAfter := countPlaceholderReasoning(t, repaired)
	realAfter := countRealReasoning(t, repaired)

	t.Logf("tool turns: placeholder before=%d after=%d, real reasoning after=%d", placeholderBefore, placeholderAfter, realAfter)

	// On a real 503-producing request, the tool history must never end up
	// carrying placeholder reasoning: every assistant tool turn either keeps the
	// real reasoning the translator produced or is repaired from real history
	// text. A placeholder here means we fed the upstream a fake reasoning, which
	// is exactly the pollution chain this fix removes.
	if placeholderAfter != 0 {
		t.Fatalf("repair left placeholder reasoning on the fixture: before=%d after=%d", placeholderBefore, placeholderAfter)
	}
	if realAfter == 0 {
		t.Fatal("repair produced no real reasoning on the fixture")
	}
}
