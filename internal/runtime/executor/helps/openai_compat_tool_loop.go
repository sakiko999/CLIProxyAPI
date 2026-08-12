package helps

import (
	"strings"

	"github.com/tidwall/gjson"
)

// toolLoopLimit is the number of consecutive identical (command, result)
// tool-call pairs that triggers a tool-call loop abort. Repeating the same
// command a few times can be legitimate (retry after a transient failure);
// five identical pairs with identical output indicates the model is stuck.
const toolLoopLimit = 5

// ToolCallLoopErrorMsg is surfaced to the client when a tool-call loop is
// detected so it stops retrying instead of burning requests forever.
const ToolCallLoopErrorMsg = "detected tool call loop: the same tool command was called 5 times with identical output; aborting to prevent runaway"

// DetectOpenAIToolCallLoop reports whether the translated OpenAI payload's
// message history contains a tool-call loop: at least toolLoopLimit consecutive
// assistant tool_calls where the tool name, arguments, and the corresponding
// tool result are all identical. The history is walked newest-first so an
// ongoing loop is caught even when older turns differ.
//
// Matching is deliberately strict (user-selected): both the command AND the
// result must be byte-identical. A retry loop whose output changes between
// calls (e.g. re-running tests until they pass) is NOT flagged.
func DetectOpenAIToolCallLoop(payload []byte) bool {
	messages := gjson.GetBytes(payload, "messages")
	if !messages.IsArray() {
		return false
	}
	all := messages.Array()

	// Pair each assistant tool_call with its tool result by matching the
	// tool_call_id, then walk from the newest pair backwards counting runs of
	// identical fingerprints.
	type pair struct {
		name, args, result string
	}
	pairs := make([]pair, 0, len(all))
	for i := range all {
		msg := all[i]
		if !strings.EqualFold(msg.Get("role").String(), "assistant") {
			continue
		}
		tc := msg.Get("tool_calls")
		if !tc.IsArray() {
			continue
		}
		tcArr := tc.Array()
		if len(tcArr) == 0 {
			continue
		}
		first := tcArr[0]
		name := first.Get("function.name").String()
		args := first.Get("function.arguments").String()
		callID := first.Get("id").String()

		// Find the corresponding tool result that follows this assistant turn.
		// A pair without a matching result is NOT appended: the tool call may
		// still be in flight (the model just issued it and CPA is about to
		// forward the request), so it is not evidence of a completed loop.
		// `found` is tracked separately from the result text so a tool whose
		// result is genuinely empty ("") still counts as a completed pair.
		result := ""
		found := false
		for j := i + 1; j < len(all); j++ {
			next := all[j]
			if !strings.EqualFold(next.Get("role").String(), "tool") {
				continue
			}
			if callID != "" && next.Get("tool_call_id").String() != callID {
				continue
			}
			result = next.Get("content").String()
			found = true
			break
		}
		if !found {
			continue
		}
		pairs = append(pairs, pair{name: name, args: args, result: result})
	}

	// Count consecutive identical pairs from the end, stopping at the first
	// differing pair. pair is comparable (all string fields), so a single ==
	// covers name, args and result.
	run := 0
	for i := len(pairs) - 1; i >= 0 && (i == len(pairs)-1 || pairs[i] == pairs[i+1]); i-- {
		run++
	}
	return run >= toolLoopLimit
}
