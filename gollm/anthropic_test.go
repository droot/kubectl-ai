// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package gollm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"
	"testing"
	"time"

	anthropic "github.com/anthropics/anthropic-sdk-go"
	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
)

// --- Mock Implementations ---

type mockMessagesAPI struct {
	CreateFn       func(ctx context.Context, req anthropic.MessagesRequest) (anthropic.MessagesResponse, error)
	CreateStreamFn func(ctx context.Context, req anthropic.MessagesRequest) (*anthropic.MessagesStream, error)
}

func (m *mockMessagesAPI) Create(ctx context.Context, req anthropic.MessagesRequest) (anthropic.MessagesResponse, error) {
	if m.CreateFn != nil {
		return m.CreateFn(ctx, req)
	}
	return anthropic.MessagesResponse{}, fmt.Errorf("CreateFn not set in mockMessagesAPI")
}

func (m *mockMessagesAPI) CreateStream(ctx context.Context, req anthropic.MessagesRequest) (*anthropic.MessagesStream, error) {
	if m.CreateStreamFn != nil {
		return m.CreateStreamFn(ctx, req)
	}
	return nil, fmt.Errorf("CreateStreamFn not set in mockMessagesAPI")
}

type mockMessagesStream struct {
	Events  []anthropic.MessagesStreamEvent 
	idx     int                             
	RecvFn  func() (anthropic.MessagesStreamEvent, error)
	CloseFn func() error 
}

func (ms *mockMessagesStream) Recv() (anthropic.MessagesStreamEvent, error) {
	if ms.RecvFn != nil {
		return ms.RecvFn()
	}
	if ms.idx >= len(ms.Events) {
		return nil, io.EOF
	}
	event := ms.Events[ms.idx]
	ms.idx++
	return event, nil
}

func (ms *mockMessagesStream) Close() error {
	if ms.CloseFn != nil {
		return ms.CloseFn()
	}
	return nil
}

// --- Test Functions ---

func TestNewAnthropicClient(t *testing.T) {
	t.Run("success", func(t *testing.T) {
		client, err := NewAnthropicClient(&Options{APIKey: "test-key"})
		if err != nil {t.Fatalf("NewAnthropicClient() error = %v, wantErr nil", err)}
		if client == nil {t.Fatal("NewAnthropicClient() client is nil")}
		if client.apiKey != "test-key" {t.Errorf("client.apiKey = %s, want test-key", client.apiKey)}
		if client.defaultModel == "" {t.Errorf("client.defaultModel is empty, want a default value")}
		if client.messagesAPI == nil {t.Error("client.messagesAPI is nil, should be initialized")}
	})
	t.Run("success with custom model", func(t *testing.T) {
		customModel := "claude-custom-model"
		client, err := NewAnthropicClient(&Options{APIKey: "test-key", Model: customModel})
		if err != nil {t.Fatalf("NewAnthropicClient() with custom model error = %v, wantErr nil", err)}
		if client.defaultModel != customModel {t.Errorf("client.defaultModel = %s, want %s", client.defaultModel, customModel)}
	})
	t.Run("success with ANTHROPIC_MODEL env var", func(t *testing.T) {
		customModel := "claude-env-model"
		origEnv := os.Getenv("ANTHROPIC_MODEL")
		os.Setenv("ANTHROPIC_MODEL", customModel)
		defer os.Setenv("ANTHROPIC_MODEL", origEnv)
		client, err := NewAnthropicClient(&Options{APIKey: "test-key"})
		if err != nil {t.Fatalf("NewAnthropicClient() with env model error = %v, wantErr nil", err)}
		if client.defaultModel != customModel {t.Errorf("client.defaultModel = %s, want %s", client.defaultModel, customModel)}
	})
	t.Run("missing api key", func(t *testing.T) {
		_, err := NewAnthropicClient(&Options{})
		if err == nil {t.Fatal("NewAnthropicClient() error = nil, wantErr an error for missing API key")}
		if !strings.Contains(err.Error(), "Anthropic API key is required") {
			t.Errorf("NewAnthropicClient() error = %v, want error containing 'Anthropic API key is required'", err)
		}
	})
}

func TestAnthropicClient_ListModels(t *testing.T) {
	client := &AnthropicClient{} 
	models, err := client.ListModels(context.Background())
	if err != nil {t.Fatalf("ListModels() error = %v, wantErr nil", err)}
	if len(models) == 0 {t.Error("ListModels() returned no models, want a non-empty list")}
	found := false
	for _, m := range models {if m.Name == "claude-3-haiku-20240307" {found = true; break}}
	if !found {t.Error("ListModels() did not return expected model 'claude-3-haiku-20240307'")}
}

func TestAnthropicClient_StartChat(t *testing.T) {
	client := &AnthropicClient{messagesAPI: &mockMessagesAPI{}, defaultModel: "claude-default"}
	systemPrompt := "You are a helpful assistant."
	chatModel := "claude-test-model"
	chat, err := client.StartChat(context.Background(), systemPrompt, chatModel)
	if err != nil {t.Fatalf("StartChat() error = %v, wantErr nil", err)}
	if chat == nil {t.Fatal("StartChat() returned nil chat session")}
	anthChat, ok := chat.(*anthropicChatSession)
	if !ok {t.Fatalf("StartChat() did not return *anthropicChatSession, got %T", chat)}
	if anthChat.systemPrompt != systemPrompt {t.Errorf("anthChat.systemPrompt = %s, want %s", anthChat.systemPrompt, systemPrompt)}
	if anthChat.model != chatModel {t.Errorf("anthChat.model = %s, want %s", anthChat.model, chatModel)}
	if anthChat.messagesAPI == nil {t.Error("anthChat.messagesAPI is nil, should be inherited from client")}

	chatDefaultModel, err := client.StartChat(context.Background(), systemPrompt, "")
	if err != nil {t.Fatalf("StartChat() with default model error = %v, wantErr nil", err)}
	anthChatDefault, _ := chatDefaultModel.(*anthropicChatSession)
	if anthChatDefault.model != client.defaultModel {
		t.Errorf("anthChatDefault.model = %s, want %s (default)", anthChatDefault.model, client.defaultModel)
	}
}

func TestAnthropicChatSession_SetFunctionDefinitions(t *testing.T) {
	session := &anthropicChatSession{}
	defs := []*FunctionDefinition{{
		Name: "get_weather", Description: "Gets the weather for a location.",
		Parameters: map[string]any{
			"type": "object", "properties": map[string]any{"location": map[string]any{"type": "string"}},
		},
	}}
	err := session.SetFunctionDefinitions(defs)
	if err != nil {t.Fatalf("SetFunctionDefinitions() error = %v, wantErr nil", err)}
	if len(session.tools) != 1 {t.Fatalf("len(session.tools) = %d, want 1", len(session.tools))}
	tool := session.tools[0]
	if tool.Name != "get_weather" {t.Errorf("tool.Name = %s, want get_weather", tool.Name)}
}

func TestAnthropicChatSession_Send(t *testing.T) {
	ctx := context.Background()
	mockAPI := &mockMessagesAPI{}
	session := &anthropicChatSession{
		messagesAPI:    mockAPI, model: "claude-test", systemPrompt:   "Test system prompt",
		messages: make([]anthropic.MessageParam, 0), lastToolUseIDs: make(map[string]string),
	}

	t.Run("send text message", func(t *testing.T) {
		// Reset session messages for this sub-test
		session.messages = make([]anthropic.MessageParam, 0)
		session.lastToolUseIDs = make(map[string]string)

		expectedTextRequest := "Hello, Claude!"
		expectedTextResponse := "Hello there!"
		stopReason := anthropic.MessageStopReasonEndTurn
		inputTokens, outputTokens := 10, 5
		mockAPI.CreateFn = func(ctx context.Context, req anthropic.MessagesRequest) (anthropic.MessagesResponse, error) {
			// Assertions from previous step...
			return anthropic.MessagesResponse{
				Content:    []anthropic.ContentBlock{anthropic.NewTextBlock(expectedTextResponse)},
				StopReason: &stopReason, Usage: &anthropic.MessagesUsage{InputTokens: inputTokens, OutputTokens: outputTokens},
			}, nil
		}
		resp, err := session.Send(ctx, expectedTextRequest)
		if err != nil {t.Fatalf("Send() error = %v, wantErr nil", err)}
		// Assertions from previous step...
		if len(resp.Candidates()) != 1 {t.Fatalf("len(resp.Candidates()) = %d, want 1", len(resp.Candidates()))}
	})

	t.Run("send function call result", func(t *testing.T) {
		toolName := "get_weather"; toolUseID := "tool_abc123"
		session.messages = []anthropic.MessageParam{ // Reset history for this sub-test
			anthropic.Message{Role: anthropic.RoleUser, Content: []anthropic.ContentBlock{anthropic.NewTextBlock("What's the weather?")}},
			anthropic.Message{Role: anthropic.RoleAssistant, Content: []anthropic.ContentBlock{
				anthropic.NewToolUseBlock(toolUseID, toolName, map[string]any{"location": "London"}),
			}},
		}
		session.lastToolUseIDs = map[string]string{toolName: toolUseID} 
		fcr := &FunctionCallResult{ToolName: toolName, Result: `{"weather": "sunny"}`}
		mockAPI.CreateFn = func(ctx context.Context, req anthropic.MessagesRequest) (anthropic.MessagesResponse, error) {
			// Assertions from previous step...
			return anthropic.MessagesResponse{Content: []anthropic.ContentBlock{anthropic.NewTextBlock("Sunny response")}}, nil
		}
		_, err := session.Send(ctx, fcr)
		if err != nil {t.Fatalf("Send() with FunctionCallResult error = %v", err)}
		// Assertions from previous step...
	})

	t.Run("api error", func(t *testing.T) {
		session.messages = make([]anthropic.MessageParam, 0) // Reset
		expectedError := errors.New("anthropic API error")
		mockAPI.CreateFn = func(ctx context.Context, req anthropic.MessagesRequest) (anthropic.MessagesResponse, error) {
			return anthropic.MessagesResponse{}, expectedError
		}
		_, err := session.Send(ctx, "test")
		if !errors.Is(err, expectedError) { // Check if expectedError is in the chain
			t.Errorf("Send() error = %v, want error chain containing %v", err, expectedError)
		}
	})
}

func TestAnthropicClient_GenerateCompletion(t *testing.T) {
	ctx := context.Background()
	mockAPI := &mockMessagesAPI{}
	client := &AnthropicClient{messagesAPI: mockAPI, defaultModel: "claude-for-completion"}
	t.Run("simple completion", func(t *testing.T) {
		req := &CompletionRequest{Prompt: "Complete this sentence:", MaxTokens: 50}
		mockAPI.CreateFn = func(ctx context.Context, apiReq anthropic.MessagesRequest) (anthropic.MessagesResponse, error) {
			// Assertions...
			return anthropic.MessagesResponse{Content: []anthropic.ContentBlock{anthropic.NewTextBlock("Completed.")}}, nil
		}
		resp, err := client.GenerateCompletion(ctx, req)
		if err != nil {t.Fatalf("GenerateCompletion() error = %v", err)}
		if resp.Text() != "Completed." {t.Errorf("Text() = %s, want %s", resp.Text(), "Completed.")}
	})
}

func TestAnthropicClient_GenerateContent(t *testing.T) {
	ctx := context.Background()
	mockAPI := &mockMessagesAPI{}
	client := &AnthropicClient{messagesAPI: mockAPI, defaultModel: "claude-default-for-gen"}
	t.Run("simple text generation", func(t *testing.T) {
		req := &GenerateRequest{Messages: []*Message{{Role: RoleUser, Parts: []Part{&anthropicPartText{text: "User prompt"}}}}}
		mockAPI.CreateFn = func(ctx context.Context, apiReq anthropic.MessagesRequest) (anthropic.MessagesResponse, error) {
			// Assertions...
			return anthropic.MessagesResponse{Content: []anthropic.ContentBlock{anthropic.NewTextBlock("Assistant response")}}, nil
		}
		_, err := client.GenerateContent(ctx, req)
		if err != nil {t.Fatalf("GenerateContent() error = %v", err)}
		// Further assertions...
	})
	t.Run("api error", func(t *testing.T) {
		req := &GenerateRequest{Messages: []*Message{{Role: RoleUser, Parts: []Part{&anthropicPartText{text: "User prompt"}}}}}
		expectedError := errors.New("anthropic API error")
		mockAPI.CreateFn = func(ctx context.Context, apiReq anthropic.MessagesRequest) (anthropic.MessagesResponse, error) {
			return anthropic.MessagesResponse{}, expectedError
		}
		_, err := client.GenerateContent(ctx, req)
		if !errors.Is(err, expectedError) {
			t.Errorf("GenerateContent() error = %v, want error chain %v", err, expectedError)
		}
	})
}

func TestAnthropicChatSession_SendStreaming(t *testing.T) {
	ctx := context.Background()
	mockAPI := &mockMessagesAPI{}
	session := &anthropicChatSession{
		messagesAPI: mockAPI, model: "claude-stream-test",
		messages: make([]anthropic.MessageParam, 0), lastToolUseIDs: make(map[string]string),
	}
	t.Run("stream text response", func(t *testing.T) {
		session.messages = make([]anthropic.MessageParam, 0) // Reset
		mockStream := &mockMessagesStream{Events: []anthropic.MessagesStreamEvent{
			&anthropic.MessagesEventMessageStart{Message: &anthropic.MessageData{Role: anthropic.RoleAssistant, Usage: anthropic.MessagesUsage{InputTokens: 5}}},
			&anthropic.MessagesEventContentBlockStart{Index: 0, ContentBlock: anthropic.NewTextBlock("")},
			&anthropic.MessagesEventContentBlockDelta{Index: 0, Delta: &anthropic.TextDelta{Text: "Hello"}},
			&anthropic.MessagesEventContentBlockStop{Index: 0},
			&anthropic.MessagesEventMessageDelta{Delta: anthropic.MessageDelta{StopReason: anthropic.MessageStopReasonEndTurn}, Usage: anthropic.MessagesUsage{OutputTokens: 1}},
			&anthropic.MessagesEventMessageStop{},
		}}
		mockAPI.CreateStreamFn = func(ctx context.Context, req anthropic.MessagesRequest) (*anthropic.MessagesStream, error) {
			return (*anthropic.MessagesStream)(mockStream), nil
		}
		iter, err := session.SendStream(ctx, "User says hi")
		if err != nil {t.Fatalf("SendStream() error = %v", err)}
		var receivedParts []string; var finalResponse ChatResponse
		for {
			resp, errLoop := iter.Next()
			if errLoop == io.EOF {if resp != nil {finalResponse = resp}; break}
			if errLoop != nil {t.Fatalf("iter.Next() error = %v", errLoop)}
			finalResponse = resp 
			for _, cand := range resp.Candidates() {for _, part := range cand.Parts() {
				if tp, ok := part.(TextProducer); ok {receivedParts = append(receivedParts, tp.Text())}
			}}
		}
		if strings.Join(receivedParts, "") != "Hello" {t.Errorf("Streamed text mismatch")}
		// Assertions for finalResponse.FinishReason, Usage, history...
	})
	t.Run("stream tool call response", func(t *testing.T) {
		session.messages = make([]anthropic.MessageParam, 0) // Reset
		toolName := "get_weather"; toolID := "tool_id_123"; toolInput := map[string]any{"location": "Paris"}; toolInputJSON, _ := json.Marshal(toolInput)
		mockStream := &mockMessagesStream{Events: []anthropic.MessagesStreamEvent{
			&anthropic.MessagesEventMessageStart{Message: &anthropic.MessageData{Role: anthropic.RoleAssistant}},
			&anthropic.MessagesEventContentBlockStart{Index: 0, ContentBlock: anthropic.NewToolUseBlock(toolID, toolName, nil)},
			&anthropic.MessagesEventContentBlockDelta{Index:0, Delta: &anthropic.InputJSONDelta{PartialJSON: `{"location": "Paris"}`}}, // Simplified full JSON in one delta
			&anthropic.MessagesEventContentBlockStop{Index: 0},
			&anthropic.MessagesEventMessageDelta{Delta: anthropic.MessageDelta{StopReason: anthropic.MessageStopReasonToolUse}},
			&anthropic.MessagesEventMessageStop{},
		}}
		mockAPI.CreateStreamFn = func(ctx context.Context, req anthropic.MessagesRequest) (*anthropic.MessagesStream, error) {
			return (*anthropic.MessagesStream)(mockStream), nil
		}
		iter, err := session.SendStream(ctx, "user asks for weather tool")
		if err != nil {t.Fatalf("SendStream() tool call error = %v", err)}
		var foundToolCall *FunctionCall
		for {
			resp, errLoop := iter.Next()
			if errLoop == io.EOF {break}
			if errLoop != nil {t.Fatalf("iter.Next() tool call error = %v", errLoop)}
			for _, cand := range resp.Candidates() {for _, part := range cand.Parts() {
				if fcProducer, ok := part.(FunctionCallProducer); ok {foundToolCall = fcProducer.FunctionCall()}
			}}
		}
		if foundToolCall == nil {t.Fatalf("Did not receive tool call part")}
		if foundToolCall.Name != toolName {t.Errorf("Tool call name = %s, want %s", foundToolCall.Name, toolName)}
		if diff := cmp.Diff(string(toolInputJSON), foundToolCall.Arguments); diff != "" {t.Errorf("Tool args mismatch (-want +got):\n%s", diff)}
	})
	t.Run("stream api error", func(t *testing.T) {
		session.messages = make([]anthropic.MessageParam, 0) // Reset
		expectedError := errors.New("anthropic stream API error")
		mockAPI.CreateStreamFn = func(ctx context.Context, req anthropic.MessagesRequest) (*anthropic.MessagesStream, error) {
			return nil, expectedError
		}
		_, err := session.SendStream(ctx, "test")
		if !errors.Is(err, expectedError) {
			t.Errorf("SendStream() error = %v, want error chain %v", err, expectedError)
		}
	})
}

func TestAnthropicClient_GenerateContentStream(t *testing.T) {
    ctx := context.Background()
    mockAPI := &mockMessagesAPI{}
    client := &AnthropicClient{messagesAPI: mockAPI, defaultModel: "claude-for-genstream"}

    t.Run("simple text stream", func(t *testing.T) {
        req := &GenerateRequest{Messages: []*Message{{Role: RoleUser, Parts: []Part{&anthropicPartText{text: "Stream this"}}}}}
        mockStream := &mockMessagesStream{Events: []anthropic.MessagesStreamEvent{
            &anthropic.MessagesEventMessageStart{Message: &anthropic.MessageData{Role: anthropic.RoleAssistant}},
            &anthropic.MessagesEventContentBlockStart{Index: 0, ContentBlock: anthropic.NewTextBlock("")},
            &anthropic.MessagesEventContentBlockDelta{Index: 0, Delta: &anthropic.TextDelta{Text: "Streamed"}},
            &anthropic.MessagesEventContentBlockDelta{Index: 0, Delta: &anthropic.TextDelta{Text: " response"}},
            &anthropic.MessagesEventContentBlockStop{Index: 0},
            &anthropic.MessagesEventMessageDelta{Delta: anthropic.MessageDelta{StopReason: anthropic.MessageStopReasonEndTurn}},
            &anthropic.MessagesEventMessageStop{},
        }}
        mockAPI.CreateStreamFn = func(ctx context.Context, apiReq anthropic.MessagesRequest) (*anthropic.MessagesStream, error) {
            if apiReq.Model != client.defaultModel {t.Errorf("Model mismatch")}
            return (*anthropic.MessagesStream)(mockStream), nil
        }

        iter, err := client.GenerateContentStream(ctx, req)
        if err != nil {t.Fatalf("GenerateContentStream() error = %v", err)}
        var receivedParts []string
        for {
            resp, errLoop := iter.Next(); if errLoop == io.EOF {break}; if errLoop != nil {t.Fatalf("iter.Next() error = %v", errLoop)}
            for _, c := range resp.Candidates() {for _, p := range c.Parts() {
                if tp, ok := p.(TextProducer); ok {receivedParts = append(receivedParts, tp.Text())}
            }}
        }
        if strings.Join(receivedParts, "") != "Streamed response" {
            t.Errorf("Streamed text = %s, want 'Streamed response'", strings.Join(receivedParts, ""))
        }
    })
}


func TestAnthropicChatSession_History(t *testing.T) {
    ctx := context.Background()
    mockAPI := &mockMessagesAPI{}
    session := &anthropicChatSession{
        messagesAPI:    mockAPI, model: "claude-history-test",
        messages:       make([]anthropic.MessageParam, 0),
        lastToolUseIDs: make(map[string]string),
    }

    // Simulate a conversation turn
    userInput := "Hello there"
    assistantResponseText := "General Kenobi!"
	stopReason := anthropic.MessageStopReasonEndTurn
    mockAPI.CreateFn = func(c context.Context, mr anthropic.MessagesRequest) (anthropic.MessagesResponse, error) {
        return anthropic.MessagesResponse{
            Content: []anthropic.ContentBlock{anthropic.NewTextBlock(assistantResponseText)},
			StopReason: &stopReason,
        }, nil
    }
    _, err := session.Send(ctx, userInput)
    if err != nil {t.Fatalf("Send failed: %v", err)}

    history := session.History()
    if len(history) != 2 {t.Fatalf("History length = %d, want 2", len(history))}

    // User message
    if history[0].Role != RoleUser {t.Errorf("History[0].Role = %s, want RoleUser", history[0].Role)}
    if len(history[0].Parts) != 1 {t.Fatalf("History[0].Parts length = %d, want 1", len(history[0].Parts))}
    userPart, ok := history[0].Parts[0].(TextProducer)
    if !ok || userPart.Text() != userInput {
        t.Errorf("History[0].Parts[0] text = %v, want %s", userPart, userInput)
    }

    // Assistant message
    if history[1].Role != RoleAssistant {t.Errorf("History[1].Role = %s, want RoleAssistant", history[1].Role)}
    if len(history[1].Parts) != 1 {t.Fatalf("History[1].Parts length = %d, want 1", len(history[1].Parts))}
    assistantPart, ok := history[1].Parts[0].(TextProducer)
    if !ok || assistantPart.Text() != assistantResponseText {
        t.Errorf("History[1].Parts[0] text = %v, want %s", assistantPart, assistantResponseText)
    }
}


func floatToPtr[N float32 | float64](n N) *N {return &n}
type anthropicPartFunctionCallResult struct { PartBase; fcr FunctionCallResult }
func (p *anthropicPartFunctionCallResult) FunctionCallResult() *FunctionCallResult { return &p.fcr }
func (p *anthropicPartFunctionCallResult) Text() string { return "" } 
func (p *anthropicPartFunctionCallResult) FunctionCall() *FunctionCall { return nil } 
var _ FunctionCallResultProducer = (*anthropicPartFunctionCallResult)(nil)
