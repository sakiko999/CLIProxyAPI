package executor

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/router-for-me/CLIProxyAPI/v7/internal/config"
	"github.com/router-for-me/CLIProxyAPI/v7/internal/registry"
	cliproxyauth "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/auth"
	cliproxyexecutor "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/executor"
	sdktranslator "github.com/router-for-me/CLIProxyAPI/v7/sdk/translator"
	"github.com/tidwall/gjson"
)

func TestOpenAICompatExecutorUsesCompatibleClaudeTranslation(t *testing.T) {
	var upstreamBody []byte
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		upstreamBody, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"chatcmpl-test","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`))
	}))
	defer server.Close()

	executor := NewOpenAICompatExecutor("openai-compatibility", &config.Config{})
	auth := &cliproxyauth.Auth{
		Provider: "openai-compatibility",
		Attributes: map[string]string{
			"base_url": server.URL,
			"api_key":  "test-key",
		},
	}
	request := cliproxyexecutor.Request{
		Model:   "deepseek-v4-flash",
		Payload: []byte(`{"model":"deepseek-v4-flash","messages":[{"role":"assistant","content":[{"type":"thinking","thinking":"prior reasoning","signature":""},{"type":"tool_use","id":"call_1","name":"Read","input":{}}]},{"role":"user","content":[{"type":"tool_result","tool_use_id":"call_1","content":"ok"}]}]}`),
		Metadata: map[string]any{
			"cliproxy.resolved_api_key_model_info": &registry.ModelInfo{IsCompat: true},
		},
	}
	options := cliproxyexecutor.Options{
		SourceFormat:   sdktranslator.FormatClaude,
		ResponseFormat: sdktranslator.FormatOpenAI,
	}

	if _, errExecute := executor.Execute(context.Background(), auth, request, options); errExecute != nil {
		t.Fatalf("Execute error: %v", errExecute)
	}

	assistant := gjson.GetBytes(upstreamBody, "messages.0")
	if got := assistant.Get("reasoning_content").String(); got != "prior reasoning" {
		t.Fatalf("reasoning_content = %q, want %q; body=%s", got, "prior reasoning", upstreamBody)
	}
	if !assistant.Get("tool_calls").Exists() {
		t.Fatalf("tool_calls missing from upstream request: %s", upstreamBody)
	}
}

// TestOpenAICompatExecutorStreamThinkingOnlyNormalDoneNoError verifies that a
// DeepSeek upstream stream carrying only reasoning_content and then ending
// NORMALLY with [DONE] (no content, no tool_calls, but a proper terminal marker)
// is treated as a legal completed turn and closed cleanly — NOT surfaced as a
// retryable 503. This is the placeholder-echo case: the model "thinks" (echoes a
// placeholder reasoning) and chooses not to produce content or a tool call this
// turn. Retrying it would turn a legitimate empty turn into a spurious error.
func TestOpenAICompatExecutorStreamThinkingOnlyNormalDoneNoError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		body := "" +
			"data: {\"id\":\"x\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"reasoning_content\":\"[reasoning unavailable]\"}}]}\n\n" +
			"data: {\"id\":\"x\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":null},\"finish_reason\":\"stop\"}]}\n\n" +
			"data: [DONE]\n\n"
		_, _ = w.Write([]byte(body))
	}))
	defer server.Close()

	executor := NewOpenAICompatExecutor("openai-compatibility", &config.Config{})
	auth := &cliproxyauth.Auth{
		Provider: "openai-compatibility",
		Attributes: map[string]string{
			"base_url": server.URL,
			"api_key":  "test-key",
		},
	}
	request := cliproxyexecutor.Request{
		Model:   "deepseek-v4-flash",
		Payload: []byte(`{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"hi"}]}`),
		Metadata: map[string]any{
			"cliproxy.resolved_api_key_model_info": &registry.ModelInfo{IsCompat: true},
		},
	}
	options := cliproxyexecutor.Options{
		SourceFormat:   sdktranslator.FormatClaude,
		ResponseFormat: sdktranslator.FormatClaude,
	}

	result, errExecute := executor.ExecuteStream(context.Background(), auth, request, options)
	if errExecute != nil {
		t.Fatalf("ExecuteStream error: %v", errExecute)
	}
	var gotErr error
	for chunk := range result.Chunks {
		if chunk.Err != nil {
			gotErr = chunk.Err
			break
		}
	}
	if gotErr != nil {
		t.Fatalf("normal thinking-only + [DONE] must NOT surface an error, got: %v", gotErr)
	}
}

// A normal stream with actual content must not produce the retryable error.
func TestOpenAICompatExecutorStreamWithContentNoError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		body := "" +
			"data: {\"id\":\"x\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"hello\"}}]}\n\n" +
			"data: [DONE]\n\n"
		_, _ = w.Write([]byte(body))
	}))
	defer server.Close()

	executor := NewOpenAICompatExecutor("openai-compatibility", &config.Config{})
	auth := &cliproxyauth.Auth{
		Provider: "openai-compatibility",
		Attributes: map[string]string{
			"base_url": server.URL,
			"api_key":  "test-key",
		},
	}
	request := cliproxyexecutor.Request{
		Model:   "deepseek-v4-flash",
		Payload: []byte(`{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"hi"}]}`),
		Metadata: map[string]any{
			"cliproxy.resolved_api_key_model_info": &registry.ModelInfo{IsCompat: true},
		},
	}
	options := cliproxyexecutor.Options{
		SourceFormat:   sdktranslator.FormatClaude,
		ResponseFormat: sdktranslator.FormatClaude,
	}

	result, errExecute := executor.ExecuteStream(context.Background(), auth, request, options)
	if errExecute != nil {
		t.Fatalf("ExecuteStream error: %v", errExecute)
	}
	for chunk := range result.Chunks {
		if chunk.Err != nil {
			t.Fatalf("unexpected error for stream with content: %v", chunk.Err)
		}
	}
}

// A thinking-only stream from a non-reasoning-vendor model must NOT trigger the
// retryable error: generic OpenAI-compatible backends may legitimately return
// empty responses, and we must not turn those into errors.
func TestOpenAICompatExecutorStreamThinkingOnlyNonVendorNoError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		body := "" +
			"data: {\"id\":\"x\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"reasoning_content\":\"thinking\"}}]}\n\n" +
			"data: [DONE]\n\n"
		_, _ = w.Write([]byte(body))
	}))
	defer server.Close()

	executor := NewOpenAICompatExecutor("openai-compatibility", &config.Config{})
	auth := &cliproxyauth.Auth{
		Provider: "openai-compatibility",
		Attributes: map[string]string{
			"base_url": server.URL,
			"api_key":  "test-key",
		},
	}
	request := cliproxyexecutor.Request{
		Model:   "gpt-5.6-luna",
		Payload: []byte(`{"model":"gpt-5.6-luna","messages":[{"role":"user","content":"hi"}]}`),
		Metadata: map[string]any{
			"cliproxy.resolved_api_key_model_info": &registry.ModelInfo{IsCompat: true},
		},
	}
	options := cliproxyexecutor.Options{
		SourceFormat:   sdktranslator.FormatClaude,
		ResponseFormat: sdktranslator.FormatClaude,
	}

	result, errExecute := executor.ExecuteStream(context.Background(), auth, request, options)
	if errExecute != nil {
		t.Fatalf("ExecuteStream error: %v", errExecute)
	}
	for chunk := range result.Chunks {
		if chunk.Err != nil {
			t.Fatalf("non-vendor thinking-only stream must not error, got: %v", chunk.Err)
		}
	}
}

// TestOpenAICompatExecutorStreamRealThinkingOnlyEOF replays a real-world
// DeepSeek truncation (log v1-messages-2026-08-10T174856-bbe81ec7): the
// upstream emits reasoning_content chunks with content:null and then closes the
// connection WITHOUT a terminal [DONE] marker. This exercises the EOF path
// (distinct from the [DONE] path) and must surface a retryable error.
func TestOpenAICompatExecutorStreamRealThinkingOnlyEOF(t *testing.T) {
	// Real chunk payloads captured from the upstream response body.
	realChunks := []string{
		`data: {"id":"18800329-f00d-4b82-8ea4-2bbb725e8e6b","object":"chat.completion.chunk","created":1786355333,"model":"deepseek-v4-flash","choices":[{"index":0,"finish_reason":null,"logprobs":null,"delta":{"role":"assistant","content":null,"reasoning_content":""}}],"usage":null}`,
		`data: {"id":"18800329-f00d-4b82-8ea4-2bbb725e8e6b","object":"chat.completion.chunk","created":1786355333,"model":"deepseek-v4-flash","choices":[{"index":0,"finish_reason":null,"logprobs":null,"delta":{"content":null,"reasoning_content":"用户"}}],"usage":null}`,
		`data: {"id":"18800329-f00d-4b82-8ea4-2bbb725e8e6b","object":"chat.completion.chunk","created":1786355333,"model":"deepseek-v4-flash","choices":[{"index":0,"finish_reason":null,"logprobs":null,"delta":{"content":null,"reasoning_content":"问"}}],"usage":null}`,
		`data: {"id":"18800329-f00d-4b82-8ea4-2bbb725e8e6b","object":"chat.completion.chunk","created":1786355333,"model":"deepseek-v4-flash","choices":[{"index":0,"finish_reason":null,"logprobs":null,"delta":{"content":null,"reasoning_content":"的是"}}],"usage":null}`,
		`data: {"id":"18800329-f00d-4b82-8ea4-2bbb725e8e6b","object":"chat.completion.chunk","created":1786355333,"model":"deepseek-v4-flash","choices":[{"index":0,"finish_reason":null,"logprobs":null,"delta":{"content":null,"reasoning_content":"CPA"}}],"usage":null}`,
		`data: {"id":"18800329-f00d-4b82-8ea4-2bbb725e8e6b","object":"chat.completion.chunk","created":1786355333,"model":"deepseek-v4-flash","choices":[{"index":0,"finish_reason":null,"logprobs":null,"delta":{"content":null,"reasoning_content":"现在只能"}}],"usage":null}`,
	}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, ok := w.(http.Flusher)
		if !ok {
			t.Fatal("handler is not a Flusher")
		}
		// Stream the reasoning chunks then close WITHOUT [DONE] to mimic the
		// real truncation where the upstream connection just ends.
		for _, chunk := range realChunks {
			_, _ = w.Write([]byte(chunk + "\n\n"))
		}
		flusher.Flush()
	}))
	defer server.Close()

	executor := NewOpenAICompatExecutor("openai-compatibility", &config.Config{})
	auth := &cliproxyauth.Auth{
		Provider: "openai-compatibility",
		Attributes: map[string]string{
			"base_url": server.URL,
			"api_key":  "test-key",
		},
	}
	request := cliproxyexecutor.Request{
		Model:   "deepseek-v4-flash",
		Payload: []byte(`{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"hi"}]}`),
		Metadata: map[string]any{
			"cliproxy.resolved_api_key_model_info": &registry.ModelInfo{IsCompat: true},
		},
	}
	options := cliproxyexecutor.Options{
		SourceFormat:   sdktranslator.FormatClaude,
		ResponseFormat: sdktranslator.FormatClaude,
	}

	result, errExecute := executor.ExecuteStream(context.Background(), auth, request, options)
	if errExecute != nil {
		t.Fatalf("ExecuteStream error: %v", errExecute)
	}
	var gotErr error
	for chunk := range result.Chunks {
		if chunk.Err != nil {
			gotErr = chunk.Err
			break
		}
	}
	if gotErr == nil {
		t.Fatal("expected retryable error for real thinking-only EOF stream, got none")
	}
	se, ok := gotErr.(cliproxyexecutor.StatusError)
	if !ok {
		t.Fatalf("error is not a StatusError: %T", gotErr)
	}
	if se.StatusCode() != http.StatusServiceUnavailable {
		t.Fatalf("status code = %d, want %d", se.StatusCode(), http.StatusServiceUnavailable)
	}
}

// TestOpenAICompatExecutorStreamRealThinkingOnlyReadError covers the
// context-canceled flavor of the truncation (log v1-messages-2026-08-10T180233
// tail "Error: context canceled"): the scanner hits a read error after emitting
// only reasoning, while the client context is still live. That must also be
// turned into a retryable error, not passed through as a raw connection error.
func TestOpenAICompatExecutorStreamRealThinkingOnlyReadError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, ok := w.(http.Flusher)
		if !ok {
			t.Fatal("handler is not a Flusher")
		}
		// Emit a reasoning chunk then abort the underlying connection so the
		// scanner surfaces a read error (context canceled / connection reset)
		// instead of a clean EOF.
		_, _ = w.Write([]byte(`data: {"id":"x","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":null,"reasoning_content":"let me think"}}]}` + "\n\n"))
		flusher.Flush()
		if hj, ok := w.(http.Hijacker); ok {
			conn, _, errHijack := hj.Hijack()
			if errHijack == nil {
				_ = conn.Close() // force a mid-stream read error on the client side
				return
			}
		}
		panic(http.ErrAbortHandler)
	}))
	defer server.Close()

	executor := NewOpenAICompatExecutor("openai-compatibility", &config.Config{})
	auth := &cliproxyauth.Auth{
		Provider: "openai-compatibility",
		Attributes: map[string]string{
			"base_url": server.URL,
			"api_key":  "test-key",
		},
	}
	request := cliproxyexecutor.Request{
		Model:   "deepseek-v4-flash",
		Payload: []byte(`{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"hi"}]}`),
		Metadata: map[string]any{
			"cliproxy.resolved_api_key_model_info": &registry.ModelInfo{IsCompat: true},
		},
	}
	options := cliproxyexecutor.Options{
		SourceFormat:   sdktranslator.FormatClaude,
		ResponseFormat: sdktranslator.FormatClaude,
	}

	result, errExecute := executor.ExecuteStream(context.Background(), auth, request, options)
	if errExecute != nil {
		t.Fatalf("ExecuteStream error: %v", errExecute)
	}
	var gotErr error
	for chunk := range result.Chunks {
		if chunk.Err != nil {
			gotErr = chunk.Err
			break
		}
	}
	if gotErr == nil {
		t.Fatal("expected retryable error for mid-stream read error after reasoning-only, got none")
	}
	se, ok := gotErr.(cliproxyexecutor.StatusError)
	if !ok {
		t.Fatalf("error is not a StatusError: %T", gotErr)
	}
	if se.StatusCode() != http.StatusServiceUnavailable {
		t.Fatalf("status code = %d, want %d", se.StatusCode(), http.StatusServiceUnavailable)
	}
}

// TestOpenAICompatExecutorToolCallLoopReturns422 verifies that a DeepSeek
// request whose history repeats the same tool command with the same output
// five times is aborted with a non-retryable 422 instead of being sent
// upstream, so the runaway loop stops instead of burning requests forever.
func TestOpenAICompatExecutorToolCallLoopReturns422(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Error("upstream must not be called when a tool-call loop is detected")
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{}`))
	}))
	defer server.Close()

	executor := NewOpenAICompatExecutor("openai-compatibility", &config.Config{})
	auth := &cliproxyauth.Auth{
		Provider: "openai-compatibility",
		Attributes: map[string]string{
			"base_url": server.URL,
			"api_key":  "test-key",
		},
	}
	// Build a Claude payload whose history contains 5 identical Bash tool
	// calls with identical results (the real-world loop shape).
	history := ""
	for i := 0; i < 5; i++ {
		history += `{"role":"assistant","content":[{"type":"thinking","thinking":"checking","signature":""},{"type":"tool_use","id":"call_` + string(rune('a'+i)) + `","name":"Bash","input":{"command":"ls -la worktrees/"}}]},` +
			`{"role":"user","content":[{"type":"tool_result","tool_use_id":"call_` + string(rune('a'+i)) + `","content":"total 0"}]},`
	}
	request := cliproxyexecutor.Request{
		Model:   "deepseek-v4-flash",
		Payload: []byte(`{"model":"deepseek-v4-flash","messages":[` + history + `{"role":"user","content":"continue"}]}`),
		Metadata: map[string]any{
			"cliproxy.resolved_api_key_model_info": &registry.ModelInfo{IsCompat: true},
		},
	}
	options := cliproxyexecutor.Options{
		SourceFormat:   sdktranslator.FormatClaude,
		ResponseFormat: sdktranslator.FormatClaude,
	}

	_, errExecute := executor.Execute(context.Background(), auth, request, options)
	if errExecute == nil {
		t.Fatal("expected tool-call loop to abort the request, got nil error")
	}
	se, ok := errExecute.(cliproxyexecutor.StatusError)
	if !ok {
		t.Fatalf("error is not a StatusError: %T", errExecute)
	}
	if se.StatusCode() != http.StatusUnprocessableEntity {
		t.Fatalf("status code = %d, want %d", se.StatusCode(), http.StatusUnprocessableEntity)
	}
}

// A history with only two identical tool calls (below the loop threshold)
// must still be sent upstream.
func TestOpenAICompatExecutorToolCallLoopBelowThresholdSendsUpstream(t *testing.T) {
	var called bool
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		called = true
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"x","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`))
	}))
	defer server.Close()

	executor := NewOpenAICompatExecutor("openai-compatibility", &config.Config{})
	auth := &cliproxyauth.Auth{
		Provider: "openai-compatibility",
		Attributes: map[string]string{
			"base_url": server.URL,
			"api_key":  "test-key",
		},
	}
	history := ""
	for i := 0; i < 2; i++ {
		history += `{"role":"assistant","content":[{"type":"thinking","thinking":"checking","signature":""},{"type":"tool_use","id":"call_` + string(rune('a'+i)) + `","name":"Bash","input":{"command":"ls"}}]},` +
			`{"role":"user","content":[{"type":"tool_result","tool_use_id":"call_` + string(rune('a'+i)) + `","content":"total 0"}]},`
	}
	request := cliproxyexecutor.Request{
		Model:   "deepseek-v4-flash",
		Payload: []byte(`{"model":"deepseek-v4-flash","messages":[` + history + `{"role":"user","content":"continue"}]}`),
		Metadata: map[string]any{
			"cliproxy.resolved_api_key_model_info": &registry.ModelInfo{IsCompat: true},
		},
	}
	options := cliproxyexecutor.Options{
		SourceFormat:   sdktranslator.FormatClaude,
		ResponseFormat: sdktranslator.FormatClaude,
	}

	if _, errExecute := executor.Execute(context.Background(), auth, request, options); errExecute != nil {
		t.Fatalf("Execute error: %v", errExecute)
	}
	if !called {
		t.Fatal("below-threshold history must be sent upstream")
	}
}
