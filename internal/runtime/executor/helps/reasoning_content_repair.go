package helps

import (
	"strconv"
	"strings"

	"github.com/router-for-me/CLIProxyAPI/v7/internal/thinking"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

// reasoningUnavailable is the safe placeholder inserted when reasoning mode is
// active but no reasoning text can be recovered for a tool-call turn. It never
// carries source content, so redacted/empty thinking cannot leak through it.
//
// The same constant drives the response-side guard: when a reasoning vendor
// echoes this exact text back as its own thinking, it is filtered out so the
// placeholder cannot accumulate in the client session (see
// IsReasoningUnavailable and the stream-side filtering in claude_input_tokens.go).
const reasoningUnavailable = "[reasoning unavailable]"

// IsReasoningUnavailable reports whether text is exactly the reasoning
// placeholder. It is the single source of truth shared by the request-side
// repair (so the placeholder is never adopted as a real reasoning source) and
// the response-side filter (so an upstream echo of the placeholder is dropped
// instead of written into the client session as thinking).
func IsReasoningUnavailable(text string) bool {
	return strings.TrimSpace(text) == reasoningUnavailable
}

// reasoningVendorHints match provider identifiers (model name or base URL) that
// require non-empty reasoning_content on every assistant tool-call turn.
// DeepSeek rejects requests whose tool history lacks reasoning text even when
// the request carries no reasoning_effort, so matching providers are always
// repaired.
var reasoningVendorHints = []string{"deepseek"}

// IsReasoningVendor reports whether the upstream is a reasoning-vendor whose
// API requires reasoning text on assistant tool-call turns. Matching is
// case-insensitive substring over the resolved base URL and base model. It is
// shared by the request-side repair and the stream-side thinking-only guard.
func IsReasoningVendor(baseURL, baseModel string) bool {
	haystack := strings.ToLower(baseURL + " " + baseModel)
	for _, hint := range reasoningVendorHints {
		if strings.Contains(haystack, hint) {
			return true
		}
	}
	return false
}

// RepairOpenAICompatReasoningContent restores reasoning text on assistant tool
// turns after source-specific translation. It only adds missing
// reasoning_content; all other provider payload fields remain unchanged.
//
// Extraction priority mirrors the original source format:
//  1. "input" array           -> Responses reasoning items
//  2. messages[] reasoning_content -> OpenAI Chat Completions source
//  3. sourceFormat "claude"   -> Claude thinking blocks
//
// Execution gate: reasoning-vendor upstreams (deepseek/kimi/mimo, matched via
// baseURL or baseModel) are always repaired, because those APIs reject
// tool-call turns that lack reasoning_content even when the request carries no
// reasoning_effort. For other upstreams the repair only runs when reasoning
// mode is active (reasoning_effort present), so the non-standard
// reasoning_content field is never sent to strict OpenAI-compatible backends.
// When the repair runs but no reasoning text is available for a tool turn, the
// safe placeholder is inserted so upstream does not reject the request.
func RepairOpenAICompatReasoningContent(sourceFormat, baseURL, baseModel string, originalSource, translated []byte) []byte {
	if len(translated) == 0 {
		return translated
	}
	if !IsReasoningVendor(baseURL, baseModel) && !gjson.GetBytes(translated, "reasoning_effort").Exists() {
		return translated
	}
	messages := gjson.GetBytes(translated, "messages")
	if !messages.IsArray() {
		return translated
	}

	var reasoning []string
	if gjson.GetBytes(originalSource, "input").IsArray() {
		reasoning = responsesReasoning(originalSource)
	} else if hasOpenAIReasoningContent(originalSource) {
		reasoning = openAIReasoning(originalSource)
	} else if strings.EqualFold(strings.TrimSpace(sourceFormat), "claude") || gjson.GetBytes(originalSource, "messages").IsArray() {
		reasoning = claudeReasoning(originalSource)
	}

	out := translated
	turnIndex := 0
	lastReasoning := ""
	// usable reports whether reasoning text is a real source: non-empty and not
	// the placeholder (which must never be absorbed or reused as reasoning).
	usable := func(text string) bool {
		t := strings.TrimSpace(text)
		return t != "" && !IsReasoningUnavailable(t)
	}
	for messageIndex, message := range messages.Array() {
		if !strings.EqualFold(message.Get("role").String(), "assistant") {
			continue
		}

		// Absorb reasoning carried by this translated turn: the translator emits
		// reasoning_content for thinking-bearing assistant messages, and those
		// turns are skipped below. Without absorbing, later pure-tool turns that
		// lost their own reasoning would find lastReasoning still empty. The
		// placeholder is never absorbed: it is only a last-resort fill for the
		// current turn and must not leak into later turns as a real reasoning
		// source.
		if own := strings.TrimSpace(message.Get("reasoning_content").String()); usable(own) {
			lastReasoning = own
		}

		isToolCall := message.Get("tool_calls").IsArray()

		// Consume the source reasoning slot aligned to this assistant turn.
		// The source extraction functions return one entry per source
		// assistant turn (dense), and the Claude/OpenAI translators map each
		// source assistant turn to exactly one translated assistant turn, so
		// reasoning[turnIndex] is the reasoning that belongs to this turn.
		// Consuming on every assistant turn — tool-call turns included — is what
		// keeps tool-call-only histories (Claude Code / DeepSeek multi-turn
		// sessions) from stalling the walk: previously turnIndex only advanced
		// on non-tool turns, so the first tool turn fell through to the
		// placeholder and polluted every later turn.
		candidate := ""
		if turnIndex < len(reasoning) {
			candidate = reasoning[turnIndex]
			turnIndex++
			if usable(candidate) {
				lastReasoning = candidate
			}
		}

		if isToolCall && !usable(candidate) {
			// This tool turn has no reasoning of its own. Fall back to the last
			// real text recovered from history, then to the turn's own
			// natural-language text (same source order as the Kimi executor and
			// cc-switch: latest reasoning, then content text, then placeholder).
			candidate = lastReasoning
			if !usable(candidate) {
				if own := strings.TrimSpace(assistantContentText(message)); usable(own) {
					candidate = own
				}
			}
		}
		if !isToolCall {
			continue
		}
		// Preserve real reasoning the translator already produced; overwrite
		// the placeholder (cross-request pollution echo) with a real source.
		// Checks for any reasoning_content field, not just non-empty, so a
		// zero-value placeholder from the translator is also overwritten.
		if own := strings.TrimSpace(message.Get("reasoning_content").String()); usable(own) {
			continue
		}
		if !usable(candidate) {
			candidate = reasoningUnavailable
		}

		path := "messages." + strconv.Itoa(messageIndex) + ".reasoning_content"
		updated, errSet := sjson.SetBytes(out, path, candidate)
		if errSet != nil {
			return translated
		}
		out = updated
	}
	return out
}

// hasOpenAIReasoningContent reports whether the original source carries OpenAI
// Chat Completions reasoning_content on any assistant message.
func hasOpenAIReasoningContent(body []byte) bool {
	for _, message := range gjson.GetBytes(body, "messages").Array() {
		if strings.EqualFold(message.Get("role").String(), "assistant") && message.Get("reasoning_content").Exists() {
			return true
		}
	}
	return false
}

// openAIReasoning extracts reasoning_content in assistant order, one entry per
// source assistant turn (dense, empty string when the turn carries none). The
// caller consumes reasoning[turnIndex] aligned to the translated turn order, so
// skipping turnless entries would desync the walk and let a tool turn inherit
// reasoning that belongs to a later assistant turn.
func openAIReasoning(body []byte) []string {
	var result []string
	for _, message := range gjson.GetBytes(body, "messages").Array() {
		if !strings.EqualFold(message.Get("role").String(), "assistant") {
			continue
		}
		result = append(result, strings.TrimSpace(message.Get("reasoning_content").String()))
	}
	return result
}

// assistantContentText recovers the natural-language text of an OpenAI-style
// assistant message (content as a plain string or an array of text parts). It
// returns "" when the message has no recoverable text. Reasoning vendors that
// echo a placeholder only learn real context from such text, so the repair uses
// it as the middle fallback between real reasoning and the placeholder — the
// same order the Kimi executor and cc-switch use.
func assistantContentText(message gjson.Result) string {
	content := message.Get("content")
	if content.Type == gjson.String {
		return content.String()
	}
	if content.IsArray() {
		var parts []string
		for _, item := range content.Array() {
			text := strings.TrimSpace(item.Get("text").String())
			if text != "" {
				parts = append(parts, text)
			}
		}
		return strings.Join(parts, "\n")
	}
	return ""
}

func claudeReasoning(body []byte) []string {
	var result []string
	for _, message := range gjson.GetBytes(body, "messages").Array() {
		if !strings.EqualFold(message.Get("role").String(), "assistant") {
			continue
		}
		var parts []string
		for _, part := range message.Get("content").Array() {
			// redacted_thinking is intentionally ignored: its "data" carries
			// AES-GCM ciphertext (Claude Code encrypts historical thinking),
			// which is meaningless to a reasoning vendor and would only be
			// echoed back as pollution. Only plaintext thinking is recovered.
			if !strings.EqualFold(strings.TrimSpace(part.Get("type").String()), "thinking") {
				continue
			}
			text := strings.TrimSpace(thinking.GetThinkingText(part))
			if text != "" {
				parts = append(parts, text)
			}
		}
		// Dense: append one entry per source assistant turn, even when no
		// thinking was recovered (empty string). The caller consumes
		// reasoning[turnIndex] aligned to the translated turn order, so
		// skipping turnless entries here would desync the walk and let a tool
		// turn inherit reasoning that belongs to a later assistant turn.
		result = append(result, strings.Join(parts, "\n\n"))
	}
	return result
}

// responsesReasoning extracts reasoning text from Responses input reasoning
// items, preserving source order. Items with no non-empty summary text are
// skipped so the reasoning list stays empty until the caller's fallback
// placeholder kicks in (see reasoningUnavailable).
//
// Responses input arrays are NOT dense per assistant turn: reasoning items sit
// beside message/function_call items rather than being attached to them, so a
// dense walk would misalign. Keep the filtered list here; the main loop's
// lastReasoning fallback covers tool turns in the Responses path.
func responsesReasoning(body []byte) []string {
	var result []string
	for _, item := range gjson.GetBytes(body, "input").Array() {
		if !strings.EqualFold(item.Get("type").String(), "reasoning") {
			continue
		}
		var parts []string
		for _, summary := range item.Get("summary").Array() {
			if summary.Get("type").String() == "summary_text" && strings.TrimSpace(summary.Get("text").String()) != "" {
				parts = append(parts, summary.Get("text").String())
			}
		}
		if joined := strings.Join(parts, ""); strings.TrimSpace(joined) != "" {
			result = append(result, joined)
		}
	}
	return result
}
