package helps

import (
	"strings"
	"testing"

	"github.com/tidwall/gjson"
)

// --- Baseline: source-specific extraction & tool-turn repair ---

func TestRepairOpenAICompatReasoningContentClaudeToolTurn(t *testing.T) {
	source := []byte(`{"messages":[{"role":"assistant","content":[{"type":"thinking","thinking":"decide","signature":""},{"type":"text","text":"calling"},{"type":"tool_use","id":"c1","name":"lookup","input":{}}]}]}`)
	translated := []byte(`{"reasoning_effort":"high","messages":[{"role":"assistant","content":[{"type":"text","text":"calling"}],"reasoning_content":"decide"},{"role":"assistant","content":"","tool_calls":[{"id":"c1"}]}]}`)
	out := RepairOpenAICompatReasoningContent("claude", "", "", source, translated)
	if got := gjson.GetBytes(out, "messages.1.reasoning_content").String(); got != "decide" {
		t.Fatalf("reasoning_content = %q, want decide", got)
	}
}

func TestRepairOpenAICompatReasoningContentResponsesToolTurn(t *testing.T) {
	source := []byte(`{"input":[{"type":"reasoning","summary":[{"type":"summary_text","text":"plan"}]},{"type":"message","role":"assistant","content":"calling"},{"type":"function_call","call_id":"c1","name":"lookup","arguments":"{}"},{"type":"function_call_output","call_id":"c1","output":"ok"}]}`)
	translated := []byte(`{"reasoning_effort":"high","messages":[{"role":"assistant","reasoning_content":"plan"},{"role":"assistant","tool_calls":[{"id":"c1"}]},{"role":"tool","content":"ok"}]}`)
	out := RepairOpenAICompatReasoningContent("openai-response", "", "", source, translated)
	if got := gjson.GetBytes(out, "messages.1.reasoning_content").String(); got != "plan" {
		t.Fatalf("reasoning_content = %q, want plan", got)
	}
}

func TestRepairOpenAICompatReasoningContentDoesNotOverwrite(t *testing.T) {
	source := []byte(`{"messages":[{"role":"assistant","content":[{"type":"thinking","thinking":"new"}]}]}`)
	translated := []byte(`{"reasoning_effort":"high","messages":[{"role":"assistant","reasoning_content":"existing","tool_calls":[{}]}]}`)
	out := RepairOpenAICompatReasoningContent("claude", "", "", source, translated)
	if got := gjson.GetBytes(out, "messages.0.reasoning_content").String(); got != "existing" {
		t.Fatalf("reasoning_content = %q, want existing", got)
	}
}

func TestRepairOpenAICompatReasoningContentNoOpWithoutReasoningMode(t *testing.T) {
	source := []byte(`{"messages":[{"role":"assistant","content":[{"type":"thinking","thinking":"new"}]}]}`)
	translated := []byte(`{"messages":[{"role":"assistant","tool_calls":[{}]}]}`)
	if got := RepairOpenAICompatReasoningContent("claude", "", "", source, translated); string(got) != string(translated) {
		t.Fatal("repair changed payload without reasoning_effort")
	}
}

func TestRepairOpenAICompatReasoningContentPreservesEffort(t *testing.T) {
	source := []byte(`{"input":[{"type":"reasoning","summary":[{"type":"summary_text","text":"one"}]}]}`)
	translated := []byte(`{"reasoning_effort":"high","model":"deepseek-v4-flash-free","messages":[{"role":"assistant","reasoning_content":"one","tool_calls":[{}]}]}`)
	out := RepairOpenAICompatReasoningContent("openai-response", "", "", source, translated)
	if gjson.GetBytes(out, "reasoning_effort").String() != "high" || gjson.GetBytes(out, "model").String() != "deepseek-v4-flash-free" {
		t.Fatalf("unrelated payload fields changed: %s", out)
	}
}

func TestRepairOpenAICompatReasoningContentIgnoresRedactedThinking(t *testing.T) {
	source := []byte(`{"messages":[{"role":"assistant","content":[{"type":"redacted_thinking","data":"x"},{"type":"text","text":"answer"}]}]}`)
	translated := []byte(`{"reasoning_effort":"high","messages":[{"role":"assistant","content":"answer"}]}`)
	out := RepairOpenAICompatReasoningContent("claude", "", "", source, translated)
	if gjson.GetBytes(out, "messages.0.reasoning_content").Exists() {
		t.Fatal("redacted thinking should not produce reasoning_content")
	}
}

// --- Impr-A: OpenAI-format source reasoning_content extraction ---

// Impr-A: an OpenAI-format source carries reasoning_content; the tool turn
// that follows must inherit it, just like Claude/Responses sources do.
func TestRepairOpenAICompatReasoningContentOpenAISourceToolTurn(t *testing.T) {
	source := []byte(`{"messages":[{"role":"assistant","content":"calling","reasoning_content":"plan"},{"role":"assistant","tool_calls":[{"id":"c1"}]}]}`)
	translated := []byte(`{"reasoning_effort":"high","messages":[{"role":"assistant","content":"calling","reasoning_content":"plan"},{"role":"assistant","content":"","tool_calls":[{"id":"c1"}]}]}`)
	out := RepairOpenAICompatReasoningContent("openai", "", "", source, translated)
	if got := gjson.GetBytes(out, "messages.1.reasoning_content").String(); got != "plan" {
		t.Fatalf("reasoning_content = %q, want plan", got)
	}
}

// Impr-A must not overwrite reasoning_content that is already present on a
// tool turn in the translated payload.
func TestRepairOpenAICompatReasoningContentOpenAISourceDoesNotOverwrite(t *testing.T) {
	source := []byte(`{"messages":[{"role":"assistant","content":"calling","reasoning_content":"plan"},{"role":"assistant","tool_calls":[{"id":"c1"}]}]}`)
	translated := []byte(`{"reasoning_effort":"high","messages":[{"role":"assistant","content":"calling","reasoning_content":"plan"},{"role":"assistant","content":"","tool_calls":[{"id":"c1"}],"reasoning_content":"existing"}]}`)
	out := RepairOpenAICompatReasoningContent("openai", "", "", source, translated)
	if got := gjson.GetBytes(out, "messages.1.reasoning_content").String(); got != "existing" {
		t.Fatalf("reasoning_content = %q, want existing", got)
	}
}

// --- Impr-B: safe placeholder when no reasoning text is available ---

// Impr-B: no thinking exists anywhere in the source; the tool turn still gets
// a safe placeholder so DeepSeek does not reject the request.
func TestRepairOpenAICompatReasoningContentPlaceholderWhenNoReasoning(t *testing.T) {
	source := []byte(`{"messages":[{"role":"assistant","content":[{"type":"text","text":"calling"},{"type":"tool_use","id":"c1","name":"lookup","input":{}}]}]}`)
	translated := []byte(`{"reasoning_effort":"high","messages":[{"role":"assistant","content":"calling"},{"role":"assistant","content":"","tool_calls":[{"id":"c1"}]}]}`)
	out := RepairOpenAICompatReasoningContent("claude", "", "", source, translated)
	if got := gjson.GetBytes(out, "messages.1.reasoning_content").String(); got != "[reasoning unavailable]" {
		t.Fatalf("reasoning_content = %q, want placeholder", got)
	}
}

// Impr-B: a tool turn appearing before any reasoning-bearing assistant turn
// gets the placeholder, while later turns keep their real reasoning text.
func TestRepairOpenAICompatReasoningContentPlaceholderToolTurnBeforeFirstReasoning(t *testing.T) {
	source := []byte(`{"messages":[
		{"role":"assistant","content":[{"type":"tool_use","id":"c1","name":"lookup","input":{}}]},
		{"role":"user","content":"got it"},
		{"role":"assistant","content":[{"type":"thinking","thinking":"plan","signature":""},{"type":"text","text":"next"},{"type":"tool_use","id":"c2","name":"read","input":{}}]}
	]}`)
	translated := []byte(`{"reasoning_effort":"high","messages":[
		{"role":"assistant","content":"","tool_calls":[{"id":"c1"}]},
		{"role":"user","content":"got it"},
		{"role":"assistant","content":"next","reasoning_content":"plan"},
		{"role":"assistant","content":"","tool_calls":[{"id":"c2"}]}
	]}`)
	out := RepairOpenAICompatReasoningContent("claude", "", "", source, translated)
	if got := gjson.GetBytes(out, "messages.0.reasoning_content").String(); got != "[reasoning unavailable]" {
		t.Fatalf("messages.0 reasoning_content = %q, want placeholder", got)
	}
	if got := gjson.GetBytes(out, "messages.3.reasoning_content").String(); got != "plan" {
		t.Fatalf("messages.3 reasoning_content = %q, want plan", got)
	}
}

// Impr-B: redacted thinking never leaks into reasoning_content; the tool turn
// receives only the safe placeholder.
func TestRepairOpenAICompatReasoningContentPlaceholderRedactedSafe(t *testing.T) {
	source := []byte(`{"messages":[{"role":"assistant","content":[{"type":"redacted_thinking","data":"secret"},{"type":"text","text":"answer"},{"type":"tool_use","id":"c1","name":"lookup","input":{}}]}]}`)
	translated := []byte(`{"reasoning_effort":"high","messages":[{"role":"assistant","content":"answer"},{"role":"assistant","content":"","tool_calls":[{"id":"c1"}]}]}`)
	out := RepairOpenAICompatReasoningContent("claude", "", "", source, translated)
	if got := gjson.GetBytes(out, "messages.1.reasoning_content").String(); got != "[reasoning unavailable]" {
		t.Fatalf("reasoning_content = %q, want placeholder", got)
	}
	if strings.Contains(string(out), "secret") {
		t.Fatal("output must never contain redacted thinking")
	}
}

// Impr-B: the placeholder must still respect the reasoning-mode gate; without
// reasoning_effort the payload stays untouched.
func TestRepairOpenAICompatReasoningContentPlaceholderRequiresReasoningMode(t *testing.T) {
	source := []byte(`{"messages":[{"role":"assistant","content":[{"type":"tool_use","id":"c1","name":"lookup","input":{}}]}]}`)
	translated := []byte(`{"messages":[{"role":"assistant","content":"","tool_calls":[{"id":"c1"}]}]}`)
	if got := RepairOpenAICompatReasoningContent("claude", "", "", source, translated); string(got) != string(translated) {
		t.Fatal("placeholder must not be added without reasoning_effort")
	}
}

// --- Deliberate non-goals (green by design) ---

// The repair only fills tool-call turns. A first (non-tool) assistant message
// that lost its reasoning_content during translation stays untouched; the
// translator is expected to have produced it.
func TestRepairOpenAICompatReasoningContentDoesNotFillFirstTextMessage(t *testing.T) {
	source := []byte(`{"messages":[{"role":"assistant","content":[{"type":"thinking","thinking":"plan","signature":""},{"type":"text","text":"calling"},{"type":"tool_use","id":"c1","name":"lookup","input":{}}]}]}`)
	translated := []byte(`{"reasoning_effort":"high","messages":[{"role":"assistant","content":"calling"},{"role":"assistant","content":"","tool_calls":[{"id":"c1"}]}]}`)
	out := RepairOpenAICompatReasoningContent("claude", "", "", source, translated)
	if gjson.GetBytes(out, "messages.0.reasoning_content").Exists() {
		t.Fatal("non-tool assistant message must not be filled (out of scope)")
	}
	if got := gjson.GetBytes(out, "messages.1.reasoning_content").String(); got != "plan" {
		t.Fatalf("tool turn reasoning_content = %q, want plan", got)
	}
}

// Consecutive tool turns reuse the latest reasoning text instead of inventing
// per-turn unique content; DeepSeek only requires presence.
func TestRepairOpenAICompatReasoningContentReusesLatestForConsecutiveToolTurns(t *testing.T) {
	source := []byte(`{"messages":[{"role":"assistant","content":[{"type":"thinking","thinking":"plan","signature":""},{"type":"text","text":"calling"},{"type":"tool_use","id":"c1","name":"lookup","input":{}}]}]}`)
	translated := []byte(`{"reasoning_effort":"high","messages":[
		{"role":"assistant","content":"calling","reasoning_content":"plan"},
		{"role":"assistant","content":"","tool_calls":[{"id":"c1"}]},
		{"role":"tool","tool_call_id":"c1","content":"ok"},
		{"role":"assistant","content":"","tool_calls":[{"id":"c2"}]}
	]}`)
	out := RepairOpenAICompatReasoningContent("claude", "", "", source, translated)
	if got := gjson.GetBytes(out, "messages.3.reasoning_content").String(); got != "plan" {
		t.Fatalf("consecutive tool turn reasoning_content = %q, want plan (latest reused)", got)
	}
}

// Regression for a real-world 400 (DeepSeek: "reasoning_content ... must be
// passed back"): the translator merges each thinking+tool_use Claude message
// into a single OpenAI assistant message that already carries reasoning_content
// AND tool_calls. Those turns are skipped below, so their reasoning must still
// be absorbed into lastReasoning for the following pure-tool turns that lost
// their own reasoning_content.
func TestRepairOpenAICompatReasoningContentMultiTurnToolChain(t *testing.T) {
	// Translated payload mirrors a real Claude->OpenAI translation: four
	// thinking+tool turns each self-carry reasoning_content, followed by pure
	// tool turns emitted without it.
	translated := []byte(`{"reasoning_effort":"high","messages":[
		{"role":"assistant","content":"first","reasoning_content":"r0","tool_calls":[{"id":"c0"}]},
		{"role":"tool","tool_call_id":"c0","content":"ok"},
		{"role":"assistant","content":"second","reasoning_content":"r1","tool_calls":[{"id":"c1"}]},
		{"role":"tool","tool_call_id":"c1","content":"ok"},
		{"role":"assistant","content":"","tool_calls":[{"id":"c2"}]},
		{"role":"tool","tool_call_id":"c2","content":"ok"},
		{"role":"assistant","content":"","tool_calls":[{"id":"c3"}]},
		{"role":"tool","tool_call_id":"c3","content":"ok"},
		{"role":"assistant","content":"final","reasoning_content":"r2","tool_calls":[{"id":"c4"}]}
	]}`)
	// Source has a matching thinking block on the first turn so the extraction
	// branch is exercised, but translated already carries the reasoning.
	source := []byte(`{"messages":[
		{"role":"assistant","content":[{"type":"thinking","thinking":"r0","signature":""},{"type":"text","text":"first"},{"type":"tool_use","id":"c0","name":"lookup","input":{}}]}
	]}`)
	out := RepairOpenAICompatReasoningContent("claude", "", "", source, translated)
	// Pure-tool turns c2/c3 (no self reasoning_content) must inherit the latest
	// absorbed reasoning (r1 from the last thinking turn), not stay empty.
	if got := gjson.GetBytes(out, "messages.4.reasoning_content").String(); got != "r1" {
		t.Fatalf("tool turn c2 reasoning_content = %q, want r1 (absorbed)", got)
	}
	if got := gjson.GetBytes(out, "messages.6.reasoning_content").String(); got != "r1" {
		t.Fatalf("tool turn c3 reasoning_content = %q, want r1 (absorbed)", got)
	}
	// Turns that already carry reasoning_content must never be overwritten.
	if got := gjson.GetBytes(out, "messages.0.reasoning_content").String(); got != "r0" {
		t.Fatalf("turn c0 reasoning_content = %q, want r0 (unchanged)", got)
	}
	if got := gjson.GetBytes(out, "messages.2.reasoning_content").String(); got != "r1" {
		t.Fatalf("turn c1 reasoning_content = %q, want r1 (unchanged)", got)
	}
}

// --- Vendor whitelist gating ---

// A non-reasoning-vendor upstream with no reasoning_effort in the payload must
// be left untouched: the non-standard reasoning_content field is not emitted to
// strict OpenAI-compatible backends.
func TestRepairOpenAICompatReasoningContentSkipsNonVendorWithoutEffort(t *testing.T) {
	source := []byte(`{"messages":[{"role":"assistant","content":[{"type":"text","text":"calling"},{"type":"tool_use","id":"c1","name":"lookup","input":{}}]}]}`)
	translated := []byte(`{"messages":[{"role":"assistant","content":"calling"},{"role":"assistant","content":"","tool_calls":[{"id":"c1"}]}]}`)
	got := RepairOpenAICompatReasoningContent("claude", "https://api.example.com/v1", "claude-sonnet-4.6", source, translated)
	if string(got) != string(translated) {
		t.Fatalf("non-vendor without reasoning_effort must be untouched, got: %s", got)
	}
}

// DeepSeek model with no reasoning_effort must still be repaired: DeepSeek
// rejects assistant tool-call turns missing reasoning_content regardless of
// whether the request carries an effort level.
func TestRepairOpenAICompatReasoningContentDeepSeekModelWithoutEffort(t *testing.T) {
	source := []byte(`{"messages":[{"role":"assistant","content":[{"type":"thinking","thinking":"plan","signature":""},{"type":"text","text":"calling"},{"type":"tool_use","id":"c1","name":"lookup","input":{}}]}]}`)
	translated := []byte(`{"messages":[{"role":"assistant","content":"calling"},{"role":"assistant","content":"","tool_calls":[{"id":"c1"}]}]}`)
	out := RepairOpenAICompatReasoningContent("claude", "https://api.deepseek.com/v1", "deepseek-v4-flash", source, translated)
	if got := gjson.GetBytes(out, "messages.1.reasoning_content").String(); got != "plan" {
		t.Fatalf("deepseek model without effort: tool turn reasoning_content = %q, want plan", got)
	}
}

// DeepSeek base URL (even with a generic model name) must be repaired the same
// way: the whitelist matches the provider endpoint too.
func TestRepairOpenAICompatReasoningContentDeepSeekBaseURLWithoutEffort(t *testing.T) {
	source := []byte(`{"messages":[{"role":"assistant","content":[{"type":"thinking","thinking":"plan","signature":""},{"type":"text","text":"calling"},{"type":"tool_use","id":"c1","name":"lookup","input":{}}]}]}`)
	translated := []byte(`{"messages":[{"role":"assistant","content":"calling"},{"role":"assistant","content":"","tool_calls":[{"id":"c1"}]}]}`)
	out := RepairOpenAICompatReasoningContent("claude", "https://api.deepseek.com/anthropic", "gpt-5.6-luna", source, translated)
	if got := gjson.GetBytes(out, "messages.1.reasoning_content").String(); got != "plan" {
		t.Fatalf("deepseek base url without effort: tool turn reasoning_content = %q, want plan", got)
	}
}

// A non-deepseek upstream with reasoning_effort keeps repairing through the
// original effort path; behavior is unchanged from before the whitelist was
// added. Kimi/Moonshot are not whitelisted, but a request that carries an
// effort level still goes through the repair.
func TestRepairOpenAICompatReasoningContentNonVendorWithEffort(t *testing.T) {
	source := []byte(`{"messages":[{"role":"assistant","content":[{"type":"thinking","thinking":"plan","signature":""},{"type":"text","text":"calling"},{"type":"tool_use","id":"c1","name":"lookup","input":{}}]}]}`)
	translated := []byte(`{"reasoning_effort":"high","messages":[{"role":"assistant","content":"calling"},{"role":"assistant","content":"","tool_calls":[{"id":"c1"}]}]}`)
	out := RepairOpenAICompatReasoningContent("claude", "https://api.moonshot.cn/v1", "kimi-k2.6", source, translated)
	if got := gjson.GetBytes(out, "messages.1.reasoning_content").String(); got != "plan" {
		t.Fatalf("non-vendor with effort: tool turn reasoning_content = %q, want plan", got)
	}
}

// Kimi/Moonshot are NOT on the whitelist: without reasoning_effort they must
// stay untouched, even though their APIs may also want reasoning content.
// The whitelist is deliberately scoped to DeepSeek only.
func TestRepairOpenAICompatReasoningContentKimiWithoutEffortNotRepaired(t *testing.T) {
	source := []byte(`{"messages":[{"role":"assistant","content":[{"type":"text","text":"calling"},{"type":"tool_use","id":"c1","name":"lookup","input":{}}]}]}`)
	translated := []byte(`{"messages":[{"role":"assistant","content":"calling"},{"role":"assistant","content":"","tool_calls":[{"id":"c1"}]}]}`)
	got := RepairOpenAICompatReasoningContent("claude", "https://api.moonshot.cn/v1", "kimi-k2.6", source, translated)
	if string(got) != string(translated) {
		t.Fatalf("kimi without reasoning_effort must be untouched (whitelist is deepseek-only), got: %s", got)
	}
}

// --- B3: recoverable content text as reasoning source ---

// Impr-B3: the OpenCode-Go upstream never returns reasoning for the current
// turn, and history is empty, but the assistant tool turn carries its own
// natural-language content text. That text is used as reasoning instead of the
// placeholder — the same fallback order as the Kimi executor and cc-switch.
func TestRepairOpenAICompatReasoningContentContentTextFallback(t *testing.T) {
	source := []byte(`{"messages":[{"role":"assistant","content":[{"type":"text","text":"I'll search the file."},{"type":"tool_use","id":"c1","name":"lookup","input":{}}]}]}`)
	translated := []byte(`{"reasoning_effort":"high","messages":[{"role":"assistant","content":"I'll search the file."},{"role":"assistant","content":"I'll search the file.","tool_calls":[{"id":"c1"}]}]}`)
	out := RepairOpenAICompatReasoningContent("claude", "", "", source, translated)
	if got := gjson.GetBytes(out, "messages.1.reasoning_content").String(); got != "I'll search the file." {
		t.Fatalf("tool turn reasoning_content = %q, want content text fallback", got)
	}
}

// The content-text fallback yields to real reasoning from history: when an
// earlier thinking block was seen, the tool turn reuses that reasoning even
// though the current turn has content text.
func TestRepairOpenAICompatReasoningContentHistoryBeatsContentText(t *testing.T) {
	source := []byte(`{"messages":[{"role":"assistant","content":[{"type":"thinking","thinking":"plan","signature":""},{"type":"text","text":"first"},{"type":"tool_use","id":"c1","name":"lookup","input":{}}]}]}`)
	translated := []byte(`{"reasoning_effort":"high","messages":[
		{"role":"assistant","content":"first","reasoning_content":"plan"},
		{"role":"assistant","content":"second attempt","tool_calls":[{"id":"c2"}]}
	]}`)
	out := RepairOpenAICompatReasoningContent("claude", "", "", source, translated)
	if got := gjson.GetBytes(out, "messages.1.reasoning_content").String(); got != "plan" {
		t.Fatalf("tool turn reasoning_content = %q, want plan (history beats content text)", got)
	}
}

// The placeholder is never adopted as a reasoning source: when the only prior
// reasoning is a placeholder, the tool turn falls through to content text
// instead of echoing the placeholder back upstream.
func TestRepairOpenAICompatReasoningContentPlaceholderNeverAdopted(t *testing.T) {
	source := []byte(`{"messages":[{"role":"assistant","content":[{"type":"text","text":"calling"},{"type":"tool_use","id":"c1","name":"lookup","input":{}}]}]}`)
	translated := []byte(`{"reasoning_effort":"high","messages":[
		{"role":"assistant","content":"","reasoning_content":"[reasoning unavailable]","tool_calls":[{"id":"c1"}]},
		{"role":"assistant","content":"doing real work","tool_calls":[{"id":"c2"}]}
	]}`)
	out := RepairOpenAICompatReasoningContent("claude", "", "", source, translated)
	if got := gjson.GetBytes(out, "messages.1.reasoning_content").String(); got != "doing real work" {
		t.Fatalf("tool turn reasoning_content = %q, want content text (placeholder never adopted)", got)
	}
}
