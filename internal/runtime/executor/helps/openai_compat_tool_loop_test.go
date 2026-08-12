package helps

import (
	"testing"
)

func toolLoopPayload(assistantPairs int) []byte {
	// Build a message history with assistantPairs identical (command, result)
	// tool-call pairs, matching the real-world loop shape seen in request logs
	// (e.g. repeated `ls` on the same worktree directory with identical output).
	payload := `{"model":"deepseek-v4-flash","messages":[` +
		`{"role":"user","content":"clean up other sessions"},`
	var msgs string
	for i := 0; i < assistantPairs; i++ {
		id := string(rune('a' + i))
		msgs += `{"role":"assistant","content":"","reasoning_content":"checking worktrees","tool_calls":[{"id":"call_00_` + id + `","type":"function","function":{"name":"Bash","arguments":"{\"command\":\"ls -la worktrees/\"}"}}]},` +
			`{"role":"tool","tool_call_id":"call_00_` + id + `","content":"total 0\ndrwxr-xr-x .\n---\nmaster"}` + ","
	}
	return []byte(payload + msgs + `{"role":"user","content":"done"}]}`)
}

func TestDetectOpenAIToolCallLoopDetectsLoop(t *testing.T) {
	got := DetectOpenAIToolCallLoop(toolLoopPayload(5))
	if !got {
		t.Fatal("expected tool call loop to be detected with 5 identical pairs")
	}
}

func TestDetectOpenAIToolCallLoopDetectsLoopBeyondLimit(t *testing.T) {
	got := DetectOpenAIToolCallLoop(toolLoopPayload(7))
	if !got {
		t.Fatal("expected tool call loop to be detected with 7 identical pairs")
	}
}

func TestDetectOpenAIToolCallLoopBelowLimit(t *testing.T) {
	got := DetectOpenAIToolCallLoop(toolLoopPayload(3))
	if got {
		t.Fatal("3 identical pairs must not be flagged as a loop")
	}
}

func TestDetectOpenAIToolCallLoopNoLoop(t *testing.T) {
	// Distinct commands / results must never be flagged.
	payload := []byte(`{"model":"deepseek-v4-flash","messages":[` +
		`{"role":"user","content":"hi"},` +
		`{"role":"assistant","content":"","tool_calls":[{"id":"c1","function":{"name":"Bash","arguments":"{\"command\":\"ls\"}"}}]},` +
		`{"role":"tool","tool_call_id":"c1","content":"dir1"}` +
		`]}`)
	if DetectOpenAIToolCallLoop(payload) {
		t.Fatal("distinct tool calls must not be flagged")
	}
}

func TestDetectOpenAIToolCallLoopChangingResultNotLoop(t *testing.T) {
	// Same command but the result changes between calls (e.g. a test that
	// starts failing then passes) must NOT be flagged.
	payload := []byte(`{"model":"deepseek-v4-flash","messages":[` +
		`{"role":"user","content":"hi"},` +
		`{"role":"assistant","content":"","tool_calls":[{"id":"c1","function":{"name":"Bash","arguments":"{\"command\":\"go test\"}"}}]},` +
		`{"role":"tool","tool_call_id":"c1","content":"FAIL"},` +
		`{"role":"assistant","content":"","tool_calls":[{"id":"c2","function":{"name":"Bash","arguments":"{\"command\":\"go test\"}"}}]},` +
		`{"role":"tool","tool_call_id":"c2","content":"PASS"}` +
		`]}`)
	if DetectOpenAIToolCallLoop(payload) {
		t.Fatal("same command with changing result must not be flagged")
	}
}

func TestDetectOpenAIToolCallLoopEmptyPayload(t *testing.T) {
	if DetectOpenAIToolCallLoop(nil) {
		t.Fatal("nil payload must not be flagged")
	}
	if DetectOpenAIToolCallLoop([]byte(`{}`)) {
		t.Fatal("empty payload must not be flagged")
	}
}

// Five identical commands with NO tool results must not be flagged: a missing
// result means the tool call is still in flight, not that a completed loop ran.
func TestDetectOpenAIToolCallLoopAllResultsMissingNotLoop(t *testing.T) {
	payload := []byte(`{"model":"m","messages":[` +
		`{"role":"user","content":"hi"},`)
	for i := 0; i < 5; i++ {
		payload = append(payload, []byte(`{"role":"assistant","content":"","tool_calls":[{"id":"c`+string(rune('a'+i))+`","function":{"name":"Bash","arguments":"{\"cmd\":\"ls\"}"}}]},`)...)
	}
	payload = append(payload, []byte(`{"role":"user","content":"go"}]}`)...)
	if DetectOpenAIToolCallLoop(payload) {
		t.Fatal("tool calls with no results must not be flagged as a loop")
	}
}

// Five identical pairs followed by a plain-text assistant turn must still be
// flagged: the text turn is not a tool pair and must not break the loop run.
func TestDetectOpenAIToolCallLoopPrecedingTextAssistant(t *testing.T) {
	payload := []byte(`{"model":"m","messages":[` +
		`{"role":"user","content":"hi"},`)
	for i := 0; i < 5; i++ {
		payload = append(payload, []byte(`{"role":"assistant","content":"","tool_calls":[{"id":"c`+string(rune('a'+i))+`","function":{"name":"Bash","arguments":"{\"cmd\":\"ls\"}"}}]},`)...)
		payload = append(payload, []byte(`{"role":"tool","tool_call_id":"c`+string(rune('a'+i))+`","content":"same"},`)...)
	}
	payload = append(payload, []byte(`{"role":"assistant","content":"final text"},{"role":"user","content":"done"}]}`)...)
	if !DetectOpenAIToolCallLoop(payload) {
		t.Fatal("5 identical pairs followed by a text assistant must still be flagged")
	}
}

// Four completed identical pairs plus one in-flight call (no result yet) must
// NOT be flagged: the loop threshold counts only fully executed pairs.
func TestDetectOpenAIToolCallLoopLastResultMissing(t *testing.T) {
	payload := []byte(`{"model":"m","messages":[` +
		`{"role":"user","content":"hi"},`)
	for i := 0; i < 4; i++ {
		payload = append(payload, []byte(`{"role":"assistant","content":"","tool_calls":[{"id":"c`+string(rune('a'+i))+`","function":{"name":"Bash","arguments":"{\"cmd\":\"ls\"}"}}]},`)...)
		payload = append(payload, []byte(`{"role":"tool","tool_call_id":"c`+string(rune('a'+i))+`","content":"same"},`)...)
	}
	payload = append(payload, []byte(`{"role":"assistant","content":"","tool_calls":[{"id":"ce","function":{"name":"Bash","arguments":"{\"cmd\":\"ls\"}"}}]},`)...)
	payload = append(payload, []byte(`{"role":"user","content":"done"}]}`)...)
	if DetectOpenAIToolCallLoop(payload) {
		t.Fatal("4 completed + 1 in-flight identical calls must not be flagged")
	}
}
