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
	"fmt"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"k8s.io/klog/v2"
)

func init() {
	if err := RegisterProvider("ollama", ollamaFactory); err != nil {
		klog.Fatalf("Failed to register ollama provider: %v", err)
	}
}

// ollamaFactory is the provider factory function for Ollama.
// Supports ClientOptions for custom configuration, including skipVerifySSL.
func ollamaFactory(ctx context.Context, opts Options) (Client, error) {
	return NewOllamaClient(ctx, opts)
}

const (
	defaultOllamaModel = "gemma3:latest"
)

type OllamaClient struct {
	client *api.Client
}

type OllamaChat struct {
	client  *api.Client
	model   string
	history []api.Message
	tools   []api.Tool
}

var _ Client = &OllamaClient{}

// NewOllamaClient creates a new client for Ollama.
// Supports custom HTTP client and skipVerifySSL via ClientOptions if the SDK supports it.
func NewOllamaClient(ctx context.Context, opts Options) (*OllamaClient, error) {
	// Create custom HTTP client with SSL verification option from client options
	httpClient := createCustomHTTPClient(opts.SkipVerifySSL)
	client := api.NewClient(envconfig.Host(), httpClient)

	return &OllamaClient{
		client: client,
	}, nil
}

func (c *OllamaClient) Close() error {
	return nil
}

func (c *OllamaClient) GenerateCompletion(ctx context.Context, request *CompletionRequest) (CompletionResponse, error) {
	req := &api.GenerateRequest{
		Model:  request.Model,
		Prompt: request.Prompt,
		Stream: ptrTo(false),
	}

	var ollamaResponse *OllamaCompletionResponse

	respFunc := func(resp api.GenerateResponse) error {
		ollamaResponse = &OllamaCompletionResponse{response: resp.Response, usage: resp} // Store usage
		return nil
	}

	err := c.client.Generate(ctx, req, respFunc)
	if err != nil {
		return nil, err
	}

	return ollamaResponse, nil
}

func (c *OllamaClient) ListModels(ctx context.Context) ([]string, error) {
	modelResponse, err := c.client.List(ctx)
	if err != nil {
		return nil, err
	}

	var models []string
	for _, model := range modelResponse.Models {
		models = append(models, model.Name)
	}

	return models, nil
}

func (c *OllamaClient) SetResponseSchema(schema *Schema) error {
	return nil
}

func (c *OllamaClient) StartChat(systemPrompt string, model string) Chat {
	return &OllamaChat{
		client: c.client,
		model:  model,
		history: []api.Message{
			{
				Role:    "system",
				Content: systemPrompt,
			},
		},
	}
}

type OllamaCompletionResponse struct {
	response string
	usage    api.GenerateResponse // To store usage data from Generate call
}

func (r *OllamaCompletionResponse) Response() string {
	return r.response
}

func (r *OllamaCompletionResponse) UsageMetadata() any {
	return r.usage // Return the whole GenerateResponse as it contains usage-like fields
}

func (c *OllamaChat) Send(ctx context.Context, contents ...any) (ChatResponse, error) {
	log := klog.FromContext(ctx)
	for _, content := range contents {
		switch v := content.(type) {
		case string:
			message := api.Message{
				Role:    "user",
				Content: v,
			}
			c.history = append(c.history, message)
		case FunctionCallResult:
			// Ollama's API expects tool_calls and their results to be passed back as role: "tool"
			// The content should be the result of the tool call.
			// The ToolCallID should match the ID of the tool_call from the assistant.
			// This simple conversion might need adjustment based on how Ollama expects tool results.
			resultJSON, err := json.Marshal(v.Result)
			if err != nil {
				klog.Errorf("Failed to marshal function call result for %s: %v", v.Name, err)
				// Decide how to handle this error, e.g., send an error message as content
				resultJSON = []byte(fmt.Sprintf(`{"error": "failed to marshal result for tool %s"}`, v.Name))
			}
			message := api.Message{
				Role:    "tool",
				Content: string(resultJSON),
				// TODO: Ollama's api.Message doesn't have a direct ToolCallID field.
				// Tool results are typically correlated by order or by including the call info in content.
				// For now, we are just sending the result.
			}
			log.V(2).Infof("Adding tool call result to history for tool %s: %s", v.Name, string(resultJSON))
			c.history = append(c.history, message)
		default:
			return nil, fmt.Errorf("unsupported content type: %T", v)
		}
	}

	req := &api.ChatRequest{
		Model:    c.model,
		Messages: c.history,
		Stream: new(bool), // Explicitly false for non-streaming
		Tools:  c.tools,
	}

	var ollamaResponse *OllamaChatResponse

	respFunc := func(resp api.ChatResponse) error {
		log.Info("received response from ollama", "resp", resp)
		ollamaResponse = &OllamaChatResponse{
			ollamaResponse: resp,
		}
		c.history = append(c.history, resp.Message)
		return nil
	}

	err := c.client.Chat(ctx, req, respFunc)
	if err != nil {
		return nil, err
	}

	log.Info("ollama response", "parsed_response", ollamaResponse)
	return ollamaResponse, nil
}

func (c *OllamaChat) IsRetryableError(err error) bool {
	// TODO(droot): Implement this
	return false
}

func (c *OllamaChat) SendStreaming(ctx context.Context, contents ...any) (ChatResponseIterator, error) {
	// TODO: Implement streaming
	response, err := c.Send(ctx, contents...)
	if err != nil {
		return nil, err
	}
	return singletonChatResponseIterator(response), nil
}

type OllamaChatResponse struct {
	ollamaResponse api.ChatResponse
}

var _ ChatResponse = &OllamaChatResponse{}

func (r *OllamaChatResponse) MarshalJSON() ([]byte, error) {
	formatted := RecordChatResponse{
		Raw: r.ollamaResponse,
	}
	return json.Marshal(&formatted)
}

func (r *OllamaChatResponse) String() string {
	if r.ollamaResponse.Message.Content != "" {
		return r.ollamaResponse.Message.Content
	}
	if len(r.ollamaResponse.Message.ToolCalls) > 0 {
		return fmt.Sprintf("[tool_calls: %d]", len(r.ollamaResponse.Message.ToolCalls))
	}
	return ""
}

func (r *OllamaChatResponse) Usage() UsageData {
	return UsageData{
		PromptTokens:     r.ollamaResponse.PromptEvalCount,
		CompletionTokens: r.ollamaResponse.EvalCount, // EvalCount is for the response tokens
		TotalTokens:      r.ollamaResponse.PromptEvalCount + r.ollamaResponse.EvalCount,
	}
}

func (r *OllamaChatResponse) Candidates() []Candidate {
	return []Candidate{&OllamaCandidate{response: r.ollamaResponse}}
}

type OllamaCandidate struct {
	response api.ChatResponse
}

var _ Candidate = &OllamaCandidate{}


func (r *OllamaCandidate) String() string {
	if r.response.Message.Content != "" {
		return r.response.Message.Content
	}
	if len(r.response.Message.ToolCalls) > 0 {
		return fmt.Sprintf("[tool_calls: %d]", len(r.response.Message.ToolCalls))
	}
	return ""
}

func (r *OllamaCandidate) FinishReason() string {
	// Ollama API's ChatResponse has a "Done" bool, but not a specific string reason like others.
	// We can infer "stop" if Done is true and no error.
	if r.response.Done {
		return "stop" 
	}
	return "" 
}


func (r *OllamaCandidate) Parts() []Part {
	var parts []Part
	if r.response.Message.Content != "" {
		parts = append(parts, &OllamaPart{PartBase{}, r.response.Message.Content, nil})
	}
	if len(r.response.Message.ToolCalls) > 0 {
		parts = append(parts, &OllamaPart{PartBase{}, "", r.response.Message.ToolCalls})
	}
	return parts
}

type OllamaPart struct {
	PartBase
	text      string
	toolCalls []api.ToolCall
}

var _ Part = (*OllamaPart)(nil)


func (p *OllamaPart) Text() string {
	return p.text
}

func (p *OllamaPart) FunctionCall() *FunctionCall {
	if len(p.toolCalls) > 0 {
		// Assuming one function call per part for simplicity,
		// or that gollm.FunctionCall should be a slice if multiple are expected in one Part.
		// For now, taking the first one.
		tc := p.toolCalls[0]
		return &FunctionCall{
			Name:      tc.Function.Name,
			Arguments: tc.Function.Arguments,
			// Ollama ToolCall doesn't have an explicit ID.
		}
	}
	return nil
}

func (p *OllamaPart) FunctionCallResult() *FunctionCallResult {
	return nil // OllamaPart represents assistant's output, not user-provided results.
}


func (c *OllamaChat) SetFunctionDefinitions(functionDefinitions []*FunctionDefinition) error {
	var tools []api.Tool
	for _, functionDefinition := range functionDefinitions {
		tools = append(tools, fnDefToOllamaTool(functionDefinition))
	}
	c.tools = tools
	return nil
}

func fnDefToOllamaTool(fnDef *FunctionDefinition) api.Tool {
	tool := api.Tool{
		Type: "function",
		Function: api.ToolFunction{
			Name:        fnDef.Name,
			Description: fnDef.Description,
			Parameters: struct {
				Type       string   `json:"type"`
				Required   []string `json:"required"`
				Properties map[string]struct {
					Type        string   `json:"type"`
					Description string   `json:"description"`
					Enum        []string `json:"enum,omitempty"`
				} `json:"properties"`
			}{
				Type:     "object",
				Required: fnDef.Parameters.Required,
				Properties: map[string]struct {
					Type        string   `json:"type"`
					Description string   `json:"description"`
					Enum        []string `json:"enum,omitempty"`
				}{},
			},
		},
	}

	if fnDef.Parameters != nil && fnDef.Parameters.Properties != nil {
		for paramName, param := range fnDef.Parameters.Properties {
			tool.Function.Parameters.Properties[paramName] = struct {
				Type        string   `json:"type"`
				Description string   `json:"description"`
				Enum        []string `json:"enum,omitempty"`
			}{
				Type:        string(param.Type),
				Description: param.Description,
				// Note: Ollama's direct parameter struct doesn't have Enum.
				// This might need to be part of the description or handled differently if strict schema adherence is needed.
			}
		}
	}


	return tool
}
