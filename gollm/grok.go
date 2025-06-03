// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
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
	"os"

	openai "github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"k8s.io/klog/v2"
)

// Register the Grok provider factory on package initialization.
// The new factory function supports ClientOptions, including skipVerifySSL.
func init() {
	if err := RegisterProvider("grok", newGrokClientFactory); err != nil {
		klog.Fatalf("Failed to register Grok provider: %v", err)
	}
}

// newGrokClientFactory is the factory function for creating Grok clients with options.
func newGrokClientFactory(ctx context.Context, opts Options) (Client, error) {
	return NewGrokClient(ctx, opts)
}

// GrokClient implements the gollm.Client interface for X.AI's Grok model.
type GrokClient struct {
	client openai.Client
}

// Ensure GrokClient implements the Client interface.
var _ Client = &GrokClient{}

// NewGrokClient creates a new client for interacting with X.AI's Grok model.
// Supports custom HTTP client and skipVerifySSL via ClientOptions.
func NewGrokClient(ctx context.Context, opts Options) (*GrokClient, error) {
	apiKey := os.Getenv("GROK_API_KEY")
	if apiKey == "" {
		return nil, errors.New("GROK_API_KEY environment variable not set")
	}

	// Default API endpoint for X.AI
	endpoint := "https://api.x.ai/v1"

	// Allow endpoint override
	customEndpoint := os.Getenv("GROK_ENDPOINT")
	if customEndpoint != "" {
		endpoint = customEndpoint
		klog.Infof("Using custom Grok endpoint: %s", endpoint)
	}

	// Use the OpenAI client with custom base URL and custom HTTP client
	httpClient := createCustomHTTPClient(opts.SkipVerifySSL)
	return &GrokClient{
		client: openai.NewClient(
			option.WithAPIKey(apiKey),
			option.WithBaseURL(endpoint),
			option.WithHTTPClient(httpClient),
		),
	}, nil
}

// Close cleans up any resources used by the client.
func (c *GrokClient) Close() error {
	// No specific cleanup needed for the Grok client currently.
	return nil
}

// StartChat starts a new chat session.
func (c *GrokClient) StartChat(systemPrompt string, model string) Chat {
	// Default to Grok-3-beta if no model is specified
	if model == "" {
		model = "grok-3-beta"
		klog.V(1).Info("No model specified, defaulting to grok-3-beta")
	}
	klog.V(1).Infof("Starting new Grok chat session with model: %s", model)

	// Initialize history with system prompt if provided
	history := []openai.ChatCompletionMessageParamUnion{}
	if systemPrompt != "" {
		history = append(history, openai.SystemMessage(systemPrompt))
	}

	return &grokChatSession{
		client:  c.client,
		history: history,
		model:   model,
	}
}

// simpleGrokCompletionResponse is a basic implementation of CompletionResponse.
type simpleGrokCompletionResponse struct {
	content string
	usage   *openai.UsageInfo
}

// Response returns the completion content.
func (r *simpleGrokCompletionResponse) Response() string {
	return r.content
}

// UsageMetadata returns usage information.
func (r *simpleGrokCompletionResponse) UsageMetadata() any {
	return r.usage
}

// GenerateCompletion sends a completion request to the Grok API.
func (c *GrokClient) GenerateCompletion(ctx context.Context, req *CompletionRequest) (CompletionResponse, error) {
	klog.Infof("Grok GenerateCompletion called with model: %s", req.Model)
	klog.V(1).Infof("Prompt:\n%s", req.Prompt)

	// Use the Chat Completions API as shown in examples
	chatReq := openai.ChatCompletionNewParams{
		Model: openai.ChatModel(req.Model), // Use the model specified in the request
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.UserMessage(req.Prompt),
		},
	}

	completion, err := c.client.Chat.Completions.New(ctx, chatReq)
	if err != nil {
		return nil, fmt.Errorf("failed to generate Grok completion: %w", err)
	}

	// Check if there are choices and a message
	if len(completion.Choices) == 0 || completion.Choices[0].Message.Content == "" {
		return nil, errors.New("received an empty response from Grok")
	}

	// Return the content of the first choice
	resp := &simpleGrokCompletionResponse{
		content: completion.Choices[0].Message.Content,
		usage:   completion.Usage,
	}

	return resp, nil
}

// SetResponseSchema is not implemented yet for Grok.
func (c *GrokClient) SetResponseSchema(schema *Schema) error {
	klog.Warning("GrokClient.SetResponseSchema is not implemented yet")
	return nil
}

// ListModels returns a list of available Grok models.
func (c *GrokClient) ListModels(ctx context.Context) ([]string, error) {
	return []string{"grok-3-beta"}, nil
}

// --- Chat Session Implementation ---

type grokChatSession struct {
	client              openai.Client
	history             []openai.ChatCompletionMessageParamUnion
	model               string
	functionDefinitions []*FunctionDefinition            
	tools               []openai.ChatCompletionToolParam 
}

var _ Chat = (*grokChatSession)(nil)

func (cs *grokChatSession) SetFunctionDefinitions(defs []*FunctionDefinition) error {
	cs.functionDefinitions = defs
	cs.tools = nil 
	if len(defs) > 0 {
		cs.tools = make([]openai.ChatCompletionToolParam, len(defs))
		for i, gollmDef := range defs {
			var params openai.FunctionParameters
			if gollmDef.Parameters != nil {
				bytes, err := gollmDef.Parameters.ToRawSchema()
				if err != nil {
					return fmt.Errorf("failed to convert schema for function %s: %w", gollmDef.Name, err)
				}
				if err := json.Unmarshal(bytes, &params); err != nil {
					return fmt.Errorf("failed to unmarshal schema for function %s: %w", gollmDef.Name, err)
				}
			}
			cs.tools[i] = openai.ChatCompletionToolParam{
				Function: openai.FunctionDefinitionParam{
					Name:        gollmDef.Name,
					Description: openai.String(gollmDef.Description),
					Parameters:  params,
				},
			}
		}
	}
	klog.V(1).Infof("Set %d function definitions for Grok chat session", len(cs.functionDefinitions))
	return nil
}

func (cs *grokChatSession) Send(ctx context.Context, contents ...any) (ChatResponse, error) {
	klog.V(1).InfoS("grokChatSession.Send called", "model", cs.model, "history_len", len(cs.history))
	for _, content := range contents {
		switch c := content.(type) {
		case string:
			klog.V(2).Infof("Adding user message to history: %s", c)
			cs.history = append(cs.history, openai.UserMessage(c))
		case FunctionCallResult:
			klog.V(2).Infof("Adding tool call result to history: Name=%s, ID=%s", c.Name, c.ID)
			resultJSON, err := json.Marshal(c.Result)
			if err != nil {
				klog.Errorf("Failed to marshal function call result: %v", err)
				return nil, fmt.Errorf("failed to marshal function call result %q: %w", c.Name, err)
			}
			cs.history = append(cs.history, openai.ToolMessage(string(resultJSON), c.ID))
		default:
			klog.Warningf("Unhandled content type in Send: %T", content)
			return nil, fmt.Errorf("unhandled content type: %T", content)
		}
	}
	chatReq := openai.ChatCompletionNewParams{Model:    openai.ChatModel(cs.model), Messages: cs.history}
	if len(cs.tools) > 0 {chatReq.Tools = cs.tools}
	klog.V(1).InfoS("Sending request to Grok Chat API", "model", cs.model, "messages", len(chatReq.Messages), "tools", len(chatReq.Tools))
	completion, err := cs.client.Chat.Completions.New(ctx, chatReq)
	if err != nil {klog.Errorf("Grok ChatCompletion API error: %v", err); return nil, fmt.Errorf("Grok chat completion failed: %w", err)}
	klog.V(1).InfoS("Received response from Grok Chat API", "id", completion.ID, "choices", len(completion.Choices))
	if len(completion.Choices) == 0 {klog.Warning("Received response with no choices from Grok"); return nil, errors.New("received empty response from Grok (no choices)")}
	assistantMsg := completion.Choices[0].Message
	cs.history = append(cs.history, assistantMsg.ToParam())
	klog.V(2).InfoS("Added assistant message to history", "content_present", assistantMsg.Content != "", "tool_calls", len(assistantMsg.ToolCalls))
	return &grokChatResponse{grokCompletion: completion}, nil
}

func (cs *grokChatSession) SendStreaming(ctx context.Context, contents ...any) (ChatResponseIterator, error) {
	klog.V(1).InfoS("Starting Grok streaming request", "model", cs.model, "streamingEnabled", true)
	for _, content := range contents {
		switch c := content.(type) {
		case string:
			klog.V(2).Infof("Adding user message to history: %s", c)
			cs.history = append(cs.history, openai.UserMessage(c))
		case FunctionCallResult:
			klog.V(2).Infof("Adding tool call result to history: Name=%s, ID=%s", c.Name, c.ID)
			resultJSON, err := json.Marshal(c.Result)
			if err != nil {klog.Errorf("Failed to marshal function call result: %v", err); return nil, fmt.Errorf("failed to marshal function call result %q: %w", c.Name, err)}
			cs.history = append(cs.history, openai.ToolMessage(string(resultJSON), c.ID))
		default:
			klog.Warningf("Unhandled content type in SendStreaming: %T", content)
			return nil, fmt.Errorf("unhandled content type: %T", content)
		}
	}
	chatReq := openai.ChatCompletionNewParams{Model: openai.ChatModel(cs.model), Messages: cs.history}
	if len(cs.tools) > 0 {chatReq.Tools = cs.tools}
	klog.V(1).InfoS("Sending streaming request to Grok API", "model", cs.model, "messageCount", len(chatReq.Messages), "toolCount", len(chatReq.Tools))
	stream := cs.client.Chat.Completions.NewStreaming(ctx, chatReq)
	acc := openai.ChatCompletionAccumulator{}
	return func(yield func(ChatResponse, error) bool) {
		var lastResponseChunk *grokChatStreamResponse
		for stream.Next() {
			chunk := stream.Current()
			acc.AddChunk(chunk)
			streamResponse := &grokChatStreamResponse{streamChunk: chunk, accumulator: acc}
			lastResponseChunk = streamResponse
			if !yield(streamResponse, nil) {return}
		}
		if err := stream.Err(); err != nil {klog.Errorf("Error in Grok streaming: %v", err); yield(nil, fmt.Errorf("Grok streaming error: %w", err)); return}
		if lastResponseChunk != nil && acc.Choices != nil && len(acc.Choices) > 0 {
			completeMessage := openai.ChatCompletionMessage{Content:   acc.Choices[0].Message.Content, Role: acc.Choices[0].Message.Role, ToolCalls: acc.Choices[0].Message.ToolCalls}
			cs.history = append(cs.history, completeMessage.ToParam())
			klog.V(2).InfoS("Added complete assistant message to history", "content_present", completeMessage.Content != "", "tool_calls", len(completeMessage.ToolCalls))
		}
	}, nil
}

func (cs *grokChatSession) IsRetryableError(err error) bool {
	if err == nil {return false}
	return DefaultIsRetryableError(err)
}

type grokChatResponse struct {
	grokCompletion *openai.ChatCompletion
}
var _ ChatResponse = (*grokChatResponse)(nil)

func (r *grokChatResponse) Usage() UsageData {
	if r.grokCompletion != nil && r.grokCompletion.Usage.TotalTokens > 0 {
		return UsageData{
			PromptTokens:     int(r.grokCompletion.Usage.PromptTokens),
			CompletionTokens: int(r.grokCompletion.Usage.CompletionTokens),
			TotalTokens:      int(r.grokCompletion.Usage.TotalTokens),
		}
	}
	return UsageData{}
}

func (r *grokChatResponse) Candidates() []Candidate {
	if r.grokCompletion == nil {return nil}
	candidates := make([]Candidate, len(r.grokCompletion.Choices))
	for i, choice := range r.grokCompletion.Choices {
		candidates[i] = &grokCandidate{grokChoice: &choice}
	}
	return candidates
}

type grokCandidate struct {
	grokChoice *openai.ChatCompletionChoice
}
var _ Candidate = (*grokCandidate)(nil)

func (c *grokCandidate) Parts() []Part {
	if c.grokChoice == nil {return nil}
	var parts []Part
	if c.grokChoice.Message.Content != "" {
		parts = append(parts, &grokPart{PartBase{}, c.grokChoice.Message.Content, nil})
	}
	if len(c.grokChoice.Message.ToolCalls) > 0 {
		parts = append(parts, &grokPart{PartBase{}, "", c.grokChoice.Message.ToolCalls})
	}
	return parts
}
func (c *grokCandidate) String() string {
	if c.grokChoice == nil {return "<nil candidate>"}
	content := "<no content>"; if c.grokChoice.Message.Content != "" {content = c.grokChoice.Message.Content}
	return fmt.Sprintf("Candidate(FinishReason: %s, ToolCalls: %d, Content: %q)", c.grokChoice.FinishReason, len(c.grokChoice.Message.ToolCalls), content)
}

type grokPart struct {
	PartBase
	content   string
	toolCalls []openai.ChatCompletionMessageToolCall
}
var _ Part = (*grokPart)(nil)

func (p *grokPart) Text() string { return p.content }
func (p *grokPart) FunctionCall() *FunctionCall {
	if len(p.toolCalls) > 0 && p.toolCalls[0].Function.Name != "" {
		tc := p.toolCalls[0] // Assuming one function call per part for simplicity here
		var args map[string]any
		_ = json.Unmarshal([]byte(tc.Function.Arguments), &args)
		return &FunctionCall{ID: tc.ID, Name: tc.Function.Name, Arguments: args}
	}
	return nil
}
func (p *grokPart) FunctionCallResult() *FunctionCallResult { return nil }


type grokChatStreamResponse struct {
	streamChunk openai.ChatCompletionChunk
	accumulator openai.ChatCompletionAccumulator
}
var _ ChatResponse = (*grokChatStreamResponse)(nil)

func (r *grokChatStreamResponse) Usage() UsageData {
	if r.accumulator.Usage.TotalTokens > 0 { // Usage is populated in accumulator
		return UsageData{
			PromptTokens:     int(r.accumulator.Usage.PromptTokens),
			CompletionTokens: int(r.accumulator.Usage.CompletionTokens),
			TotalTokens:      int(r.accumulator.Usage.TotalTokens),
		}
	}
	// Fallback for chunks that might not have usage yet (though accumulator should have final)
	if len(r.streamChunk.Choices) > 0 && r.streamChunk.Choices[0].Delta.Usage != nil {
		usage := r.streamChunk.Choices[0].Delta.Usage
		return UsageData{
			PromptTokens:     int(usage.PromptTokens),
			CompletionTokens: int(usage.CompletionTokens),
			TotalTokens:      int(usage.TotalTokens),
		}
	}
	return UsageData{}
}

func (r *grokChatStreamResponse) Candidates() []Candidate {
	if len(r.streamChunk.Choices) == 0 {return nil}
	candidates := make([]Candidate, len(r.streamChunk.Choices))
	for i, choice := range r.streamChunk.Choices {
		candidates[i] = &grokStreamCandidate{streamChoice: choice}
	}
	return candidates
}

type grokStreamCandidate struct {
	streamChoice openai.ChatCompletionChunkChoice
}
var _ Candidate = (*grokStreamCandidate)(nil)

func (c *grokStreamCandidate) String() string {
	return fmt.Sprintf("StreamingCandidate(Index: %d, FinishReason: %s)", c.streamChoice.Index, c.streamChoice.FinishReason)
}
func (c *grokStreamCandidate) Parts() []Part {
	var parts []Part
	if c.streamChoice.Delta.Content != "" {parts = append(parts, &grokStreamPart{PartBase{}, c.streamChoice.Delta.Content, nil})}
	if len(c.streamChoice.Delta.ToolCalls) > 0 {
		toolCalls := make([]openai.ChatCompletionMessageToolCall, len(c.streamChoice.Delta.ToolCalls))
		for i, deltaToolCall := range c.streamChoice.Delta.ToolCalls {
			toolCalls[i] = openai.ChatCompletionMessageToolCall{
				ID:       deltaToolCall.ID,
				Type:     "function", // Delta doesn't specify type, assume function
				Function: openai.ChatCompletionMessageToolCallFunction{
					Name:      deltaToolCall.Function.Name,
					Arguments: deltaToolCall.Function.Arguments,
				},
			}
		}
		parts = append(parts, &grokStreamPart{PartBase{}, "", toolCalls})
	}
	return parts
}

type grokStreamPart struct {
	PartBase
	content   string
	toolCalls []openai.ChatCompletionMessageToolCall
}
var _ Part = (*grokStreamPart)(nil)

func (p *grokStreamPart) Text() string { return p.content }
func (p *grokStreamPart) FunctionCall() *FunctionCall {
	if len(p.toolCalls) > 0 && p.toolCalls[0].Function.Name != "" {
		tc := p.toolCalls[0]
		var args map[string]any
		_ = json.Unmarshal([]byte(tc.Function.Arguments), &args)
		return &FunctionCall{ID: tc.ID, Name: tc.Function.Name, Arguments: args}
	}
	return nil
}
func (p *grokStreamPart) FunctionCallResult() *FunctionCallResult { return nil }

[end of gollm/grok.go]
