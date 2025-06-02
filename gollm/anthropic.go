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
	"log"
	"os"
	"strings"

	anthropic "github.com/anthropics/anthropic-sdk-go"
)

const (
	defaultAnthropicModel    = "claude-3-haiku-20240307"
	DefaultMaxTokensToSample = 2048
)

// anthropicMessagesAPI defines the interface for Anthropic's messages service,
// allowing for mocking in tests.
type anthropicMessagesAPI interface {
	Create(context.Context, anthropic.MessagesRequest) (anthropic.MessagesResponse, error)
	CreateStream(context.Context, anthropic.MessagesRequest) (*anthropic.MessagesStream, error)
}

// AnthropicClient represents a client for the Anthropic API.
type AnthropicClient struct {
	messagesAPI  anthropicMessagesAPI // Interface for Anthropic messages service
	rawSDKClient *anthropic.Client    // Keep raw client if needed for other services
	defaultModel string
	apiKey       string
}

// anthropicChatSession represents a chat session with the Anthropic API.
type anthropicChatSession struct {
	client              *AnthropicClient // Reference to the parent client
	messagesAPI         anthropicMessagesAPI
	model               string
	systemPrompt        string
	messages            []anthropic.MessageParam
	tools               []anthropic.ToolDefinition
	functionDefinitions []*FunctionDefinition
	lastToolUseIDs      map[string]string // Tracks ToolName -> ToolUseID from assistant
}

// --- Client Factory and Initialization ---
func init() {
	if _, ok := providerFactories["anthropic"]; !ok {
		providerFactories["anthropic"] = newAnthropicClientFactory
	}
}

func newAnthropicClientFactory(opts *Options) (Client, error) {
	apiKey := opts.APIKey
	if apiKey == "" {
		apiKey = os.Getenv("ANTHROPIC_API_KEY")
	}
	if apiKey == "" {
		return nil, fmt.Errorf("Anthropic API key not set. Set ANTHROPIC_API_KEY or pass with --apikey")
	}
	// Pass API key via opts, model will be resolved in NewAnthropicClient
	internalOpts := &Options{APIKey: apiKey, Model: opts.Model}
	return NewAnthropicClient(internalOpts)
}

func NewAnthropicClient(opts *Options) (*AnthropicClient, error) {
	if opts.APIKey == "" {
		return nil, fmt.Errorf("Anthropic API key is required")
	}
	model := opts.Model
	if model == "" {
		model = os.Getenv("ANTHROPIC_MODEL")
	}
	if model == "" {
		model = defaultAnthropicModel
	}
	sdkClient, err := anthropic.NewClient(opts.APIKey)
	if err != nil {
		return nil, fmt.Errorf("failed to create Anthropic SDK client: %w", err)
	}
	return &AnthropicClient{
		messagesAPI:  sdkClient.Messages, // Assign the messages service to the interface
		rawSDKClient: sdkClient,
		defaultModel: model,
		apiKey:       opts.APIKey,
	}, nil
}

// --- AnthropicClient Methods (gollm.Client interface) ---
func (c *AnthropicClient) Close() error { return nil }

func (c *AnthropicClient) StartChat(ctx context.Context, systemPrompt string, model string) (Chat, error) {
	chatModel := model
	if chatModel == "" {
		chatModel = c.defaultModel
	}
	if c.messagesAPI == nil { // Check the interface, not the rawSDKClient for this
		return nil, fmt.Errorf("Anthropic messages API not initialized")
	}
	return &anthropicChatSession{
		client:           c,
		messagesAPI:      c.messagesAPI, // Pass the interface
		model:            chatModel,
		systemPrompt:     systemPrompt,
		messages:         make([]anthropic.MessageParam, 0),
		lastToolUseIDs:   make(map[string]string),
	}, nil
}

func (c *AnthropicClient) GenerateCompletion(ctx context.Context, req *CompletionRequest) (CompletionResponse, error) {
	if c.messagesAPI == nil {
		return nil, fmt.Errorf("SDK client not initialized")
	}
	modelName := req.Model
	if modelName == "" {
		modelName = c.defaultModel
	}
	anthropicMessages := []anthropic.MessageParam{anthropic.NewUserTextMessageParam(req.Prompt)}
	maxTokens := DefaultMaxTokensToSample
	if req.MaxTokens > 0 {
		maxTokens = req.MaxTokens
	}
	anthropicReq := anthropic.MessagesRequest{Model: modelName, Messages: anthropicMessages, MaxTokens: maxTokens}
	if req.Temperature != nil {
		anthropicReq.Temperature = (*float64)(req.Temperature)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	resp, err := c.messagesAPI.Create(ctx, anthropicReq) // Use interface
	if err != nil {
		return nil, fmt.Errorf("API request failed: %w", err)
	}
	if len(resp.Content) == 0 {
		return nil, fmt.Errorf("API returned no content")
	}
	var responseText string
	if textBlock, ok := resp.Content[0].(*anthropic.TextBlock); ok {
		responseText = textBlock.Text
	} else {
		return nil, fmt.Errorf("unexpected content block type: %T", resp.Content[0])
	}
	return &anthropicCompletionResponse{text: responseText}, nil
}

type anthropicCompletionResponse struct{ text string }

func (r *anthropicCompletionResponse) Text() string { return r.text }

func (c *AnthropicClient) SetResponseSchema(schema any) error {
	log.Println("Warning: SetResponseSchema not supported by Anthropic. Use Tool definitions.")
	return nil
}
func (c *AnthropicClient) ListModels(ctx context.Context) ([]ModelInfo, error) {
	models := []ModelInfo{
		{Name: "claude-3-opus-20240229", Description: "Most powerful.", MaxTokens: 200000},
		{Name: "claude-3-sonnet-20240229", Description: "Balance of intelligence/speed.", MaxTokens: 200000},
		{Name: "claude-3-haiku-20240307", Description: "Fastest, most compact.", MaxTokens: 200000, Default: true},
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return models, nil
}

func gollmMessagesToAnthropic(gollmMessages []*Message, lastToolUseIDs map[string]string) ([]anthropic.MessageParam, string, error) {
	var systemPrompt string
	anthropicMessages := make([]anthropic.MessageParam, 0, len(gollmMessages))
	for i, gMsg := range gollmMessages {
		if i == 0 && gMsg.Role == RoleSystem {
			var sb strings.Builder
			for _, part := range gMsg.Parts {
				if textProvider, ok := part.(TextProducer); ok {sb.WriteString(textProvider.Text())} else {
					return nil, "", fmt.Errorf("system message part is not TextProducer: %T", part)
				}
			}
			systemPrompt = sb.String(); continue
		}
		contentBlocks := make([]anthropic.ContentBlock, 0, len(gMsg.Parts))
		for _, part := range gMsg.Parts {
			if textProducer, ok := part.(TextProducer); ok && textProducer.Text() != "" {
				contentBlocks = append(contentBlocks, anthropic.NewTextBlock(textProducer.Text()))
			} else if fcProducer, ok := part.(FunctionCallProducer); ok && fcProducer.FunctionCall() != nil {
				fc := fcProducer.FunctionCall(); var inputMap map[string]any
				if err := json.Unmarshal([]byte(fc.Arguments), &inputMap); err != nil {
					log.Printf("Warning: Could not unmarshal function call args for %s: %v. Raw: %s. Sending as raw.", fc.Name, err, fc.Arguments)
					inputMap = map[string]any{"_raw_arguments": fc.Arguments}
				}
				toolUseID := fc.Name 
				contentBlocks = append(contentBlocks, anthropic.NewToolUseBlock(toolUseID, fc.Name, inputMap))
			} else if fcrProducer, ok := part.(FunctionCallResultProducer); ok && fcrProducer.FunctionCallResult() != nil {
				fcr := fcrProducer.FunctionCallResult(); toolUseID, idExists := lastToolUseIDs[fcr.ToolName]
				if !idExists {log.Printf("Warning: No ToolUseID for tool result %s. Using ToolName.", fcr.ToolName); toolUseID = fcr.ToolName}
				var resultStr string
				if strRes, ok := fcr.Result.(string); ok {resultStr = strRes} else {
					jsonRes, err := json.Marshal(fcr.Result)
					if err != nil {resultStr = `{"error": "failed to marshal tool result"}`; log.Printf("Error marshalling tool result for %s: %v", fcr.ToolName, err)} else {resultStr = string(jsonRes)}
				}
				contentBlocks = append(contentBlocks, anthropic.NewToolResultBlock(toolUseID, resultStr))
			}
		}
		if len(contentBlocks) == 0 {log.Printf("Warning: Gollm message at index %d resulted in no content blocks.", i); continue}
		var role anthropic.Role
		switch gMsg.Role {
		case RoleUser: role = anthropic.RoleUser
		case RoleAssistant: role = anthropic.RoleAssistant
		default: return nil, "", fmt.Errorf("unsupported gollm role: %s", gMsg.Role)
		}
		anthropicMessages = append(anthropicMessages, anthropic.Message{Role: role, Content: contentBlocks})
	}
	return anthropicMessages, systemPrompt, nil
}

func (c *AnthropicClient) GenerateContent(ctx context.Context, req *GenerateRequest) (*GenerateResponse, error) {
	if c.messagesAPI == nil {return nil, fmt.Errorf("Anthropic messages API not initialized")}
	anthropicMsgs, systemPrompt, err := gollmMessagesToAnthropic(req.Messages, make(map[string]string))
	if err != nil {return nil, fmt.Errorf("failed to convert gollm messages: %w", err)}
	modelName := c.defaultModel; if req.Model != "" {modelName = req.Model}
	anthropicReq := anthropic.MessagesRequest{Model: modelName, Messages: anthropicMsgs}
	if systemPrompt != "" {anthropicReq.System = systemPrompt}
	if req.MaxOutputTokens > 0 {anthropicReq.MaxTokens = req.MaxOutputTokens} else {anthropicReq.MaxTokens = DefaultMaxTokensToSample}
	if req.Temperature != nil {anthropicReq.Temperature = req.Temperature}
	if req.TopP != nil {anthropicReq.TopP = req.TopP}
	if req.TopK > 0 {anthropicReq.TopK = &req.TopK}

	if len(req.Tools) > 0 {
		anthropicTools := make([]anthropic.ToolDefinition, len(req.Tools))
		for i, toolDef := range req.Tools {
			var params map[string]any
			if toolDef.Parameters != nil {
				if pMap, ok := toolDef.Parameters.(map[string]any); ok {params = pMap} else {
					b, err := json.Marshal(toolDef.Parameters); if err != nil {return nil, fmt.Errorf("marshal params for %s: %w", toolDef.Name, err)}
					if err := json.Unmarshal(b, &params); err != nil {return nil, fmt.Errorf("unmarshal params for %s: %w", toolDef.Name, err)}
				}
			}
			anthropicTools[i] = anthropic.ToolDefinition{Name: toolDef.Name, Description: toolDef.Description, InputSchema: anthropic.NewToolInputSchema(params)}
		}
		anthropicReq.Tools = anthropicTools
	}
	if err := ctx.Err(); err != nil {return nil, err}
	apiResp, err := c.messagesAPI.Create(ctx, anthropicReq) // Use interface
	if err != nil {return nil, fmt.Errorf("Anthropic Messages.Create API call failed: %w", err)}
	gollmParts := make([]Part, 0, len(apiResp.Content))
	for _, block := range apiResp.Content {
		switch b := block.(type) {
		case *anthropic.TextBlock: gollmParts = append(gollmParts, &anthropicPartText{text: b.Text})
		case *anthropic.ToolUseBlock:
			inputJSON, _ := json.Marshal(b.Input)
			gollmParts = append(gollmParts, &anthropicPartFunctionCall{fc: FunctionCall{Name: b.Name, Arguments: string(inputJSON)}, toolUseID: b.ID})
		default: log.Printf("GenerateContent: Unknown Anthropic content block type: %T", b)
		}
	}
	var finishReason string; if apiResp.StopReason != nil {finishReason = string(*apiResp.StopReason)}
	usage := UsageData{}; if apiResp.Usage != nil {
		usage.PromptTokens = apiResp.Usage.InputTokens; usage.CompletionTokens = apiResp.Usage.OutputTokens
		usage.TotalTokens = apiResp.Usage.InputTokens + apiResp.Usage.OutputTokens
	}
	return &GenerateResponse{Candidates: []*Candidate{{Message: &Message{Role: RoleAssistant, Parts: gollmParts}, FinishReason: finishReason}}, Usage: usage}, nil
}

func (c *AnthropicClient) GenerateContentStream(ctx context.Context, req *GenerateRequest) (ChatResponseIterator, error) {
	if c.messagesAPI == nil {return nil, fmt.Errorf("Anthropic messages API not initialized")}
	anthropicMsgs, systemPrompt, err := gollmMessagesToAnthropic(req.Messages, make(map[string]string))
	if err != nil {return nil, fmt.Errorf("failed to convert gollm messages for streaming: %w", err)}
	modelName := c.defaultModel; if req.Model != "" {modelName = req.Model}
	anthropicReq := anthropic.MessagesRequest{Model: modelName, Messages: anthropicMsgs, Stream: true}
	if systemPrompt != "" {anthropicReq.System = systemPrompt}
	if req.MaxOutputTokens > 0 {anthropicReq.MaxTokens = req.MaxOutputTokens} else {anthropicReq.MaxTokens = DefaultMaxTokensToSample}
	if req.Temperature != nil {anthropicReq.Temperature = req.Temperature}
	if req.TopP != nil {anthropicReq.TopP = req.TopP}
	if req.TopK > 0 {anthropicReq.TopK = &req.TopK}

	if len(req.Tools) > 0 {
		anthropicTools := make([]anthropic.ToolDefinition, len(req.Tools))
		for i, toolDef := range req.Tools {
			var params map[string]any
			if toolDef.Parameters != nil {
				if pMap, ok := toolDef.Parameters.(map[string]any); ok {params = pMap} else {
					b, _ := json.Marshal(toolDef.Parameters); json.Unmarshal(b, &params)
				}
			}
			anthropicTools[i] = anthropic.ToolDefinition{Name: toolDef.Name, Description: toolDef.Description, InputSchema: anthropic.NewToolInputSchema(params)}
		}
		anthropicReq.Tools = anthropicTools
	}
	stream, err := c.messagesAPI.CreateStream(ctx, anthropicReq) // Use interface
	if err != nil {return nil, fmt.Errorf("Anthropic Messages.CreateStream API call failed: %w", err)}
	tempSession := &anthropicChatSession{
		client: c, messagesAPI: c.messagesAPI, model: modelName,
		messages: anthropicMsgs, lastToolUseIDs: make(map[string]string), systemPrompt: systemPrompt,
	}
	return &anthropicChatResponseIterator{
		ctx: ctx, cs: tempSession, stream: stream,
		fullAssistantMessage: anthropic.Message{Role: anthropic.RoleAssistant, Content: make([]anthropic.ContentBlock, 0)},
		activeTextBlocks: make(map[int]*strings.Builder), activeToolInputs: make(map[int]*strings.Builder),
		pendingParts: make([]Part, 0),
	}, nil
}
func (c *AnthropicClient) Name() string { return "anthropic" }

// --- anthropicChatSession Methods (gollm.Chat interface) ---
func (cs *anthropicChatSession) SetFunctionDefinitions(defs []*FunctionDefinition) error {
	cs.functionDefinitions = defs; cs.tools = make([]anthropic.ToolDefinition, len(defs))
	for i, def := range defs {
		var params map[string]any
		if def.Parameters != nil {
			if pMap, ok := def.Parameters.(map[string]any); ok {params = pMap} else {
				b, err := json.Marshal(def.Parameters); if err != nil {return fmt.Errorf("marshal params for %s: %w", def.Name, err)}
				if err := json.Unmarshal(b, &params); err != nil {return fmt.Errorf("unmarshal params for %s: %w", def.Name, err)}
			}
		}
		cs.tools[i] = anthropic.ToolDefinition{Name: def.Name, Description: def.Description, InputSchema: anthropic.NewToolInputSchema(params)}
	}
	return nil
}

func (cs *anthropicChatSession) processToMessageParams(contents ...any) error {
	for _, content := range contents {
		switch c := content.(type) {
		case string: cs.messages = append(cs.messages, anthropic.NewUserTextMessageParam(c))
		case *FunctionCallResult:
			var resultStr string
			if strContent, ok := c.Result.(string); ok {resultStr = strContent} else {
				jsonBytes, err := json.Marshal(c.Result)
				if err != nil {log.Printf("Warning: Marshal FCR content for %s: %v.", c.ToolName, err); resultStr = `{"error": "failed to marshal result"}`} else {resultStr = string(jsonBytes)}
			}
			toolUseID, ok := cs.lastToolUseIDs[c.ToolName]
			if !ok {
				log.Printf("Warning: No cached ToolUseID for ToolName '%s'. Searching history.", c.ToolName)
				var foundID string
				for i := len(cs.messages) - 1; i >= 0; i-- {
					if msg, okMsg := cs.messages[i].(anthropic.Message); okMsg && msg.Role == anthropic.RoleAssistant {
						for _, block := range msg.Content {
							if tub, okTub := block.(*anthropic.ToolUseBlock); okTub && tub.Name == c.ToolName {
								if idInCache, idExists := cs.lastToolUseIDs[tub.Name]; !idExists || idInCache != tub.ID {
									foundID = tub.ID; log.Printf("Found ToolUseID '%s' for '%s' in history.", foundID, c.ToolName); break
								}
							}
						}
						if foundID != "" {break}
					}
				}
				if foundID != "" {toolUseID = foundID} else {
					log.Printf("Error: Could not find ToolUseID for result '%s'. Using ToolName as ID (likely to fail).", c.ToolName)
					toolUseID = c.ToolName
				}
			} else {delete(cs.lastToolUseIDs, c.ToolName)}
			cs.messages = append(cs.messages, anthropic.NewToolResultParam(toolUseID, resultStr))
		default: return fmt.Errorf("unsupported content type: %T", c)
		}
	}
	return nil
}

func (cs *anthropicChatSession) Send(ctx context.Context, contents ...any) (ChatResponse, error) {
	if cs.messagesAPI == nil {return nil, fmt.Errorf("Anthropic messages API not initialized")}
	if err := cs.processToMessageParams(contents...); err != nil {return nil, err}
	req := anthropic.MessagesRequest{Model: cs.model, Messages: cs.messages, MaxTokens: DefaultMaxTokensToSample}
	if cs.systemPrompt != "" {req.System = cs.systemPrompt}
	if len(cs.tools) > 0 {req.Tools = cs.tools}
	if err := ctx.Err(); err != nil {return nil, err}
	resp, err := cs.messagesAPI.Create(ctx, req) // Use interface
	if err != nil {return nil, fmt.Errorf("Messages.Create failed: %w", err)}
	assistantMessage := anthropic.Message{Role: anthropic.RoleAssistant, Content: resp.Content}
	cs.messages = append(cs.messages, assistantMessage)
	for _, block := range resp.Content {
		if tub, ok := block.(*anthropic.ToolUseBlock); ok {cs.lastToolUseIDs[tub.Name] = tub.ID}
	}
	return cs.convertToGollmChatResponse(resp, nil), nil
}

func (cs *anthropicChatSession) convertToGollmChatResponse(resp *anthropic.MessagesResponse, streamStopReason *anthropic.MessageStopReason) ChatResponse {
	gollmParts := make([]Part, 0, len(resp.Content))
	for _, block := range resp.Content {
		switch b := block.(type) {
		case *anthropic.TextBlock: gollmParts = append(gollmParts, &anthropicPartText{text: b.Text})
		case *anthropic.ToolUseBlock:
			inputJSON, err := json.Marshal(b.Input); argsStr := string(inputJSON)
			if err != nil {log.Printf("Warning: Marshal tool input for %s: %v", b.Name, err); argsStr = fmt.Sprintf(`{"error":"marshal input: %v", "raw":"%+v"}`, err, b.Input)}
			gollmParts = append(gollmParts, &anthropicPartFunctionCall{fc: FunctionCall{Name: b.Name, Arguments: argsStr}, toolUseID: b.ID})
		default: log.Printf("Warning: Unknown Anthropic content block type: %T", b)
		}
	}
	var finishReason string
	if resp != nil && resp.StopReason != nil {finishReason = string(*resp.StopReason)} else if streamStopReason != nil {finishReason = string(*streamStopReason)}
	usage := UsageData{}; if resp != nil && resp.Usage != nil {
		usage.PromptTokens = resp.Usage.InputTokens; usage.CompletionTokens = resp.Usage.OutputTokens
		usage.TotalTokens = resp.Usage.InputTokens + resp.Usage.OutputTokens
	}
	return &anthropicChatResponse{candidates: []Candidate{&anthropicCandidate{parts: gollmParts, finishReason: finishReason}}, usage: usage}
}

type anthropicChatResponse struct{ candidates []Candidate; usage UsageData }
func (r *anthropicChatResponse) Candidates() []Candidate { return r.candidates }
func (r *anthropicChatResponse) Usage() UsageData      { return r.usage }

type anthropicCandidate struct{ parts []Part; finishReason string }
func (c *anthropicCandidate) Parts() []Part        { return c.parts }
func (c *anthropicCandidate) FinishReason() string { return c.finishReason }

type anthropicPartText struct{ text string }
func (p *anthropicPartText) Text() string            { return p.text }
func (p *anthropicPartText) FunctionCall() *FunctionCall { return nil }

type anthropicPartFunctionCall struct{ fc FunctionCall; toolUseID string }
func (p *anthropicPartFunctionCall) Text() string            { return "" }
func (p *anthropicPartFunctionCall) FunctionCall() *FunctionCall { return &p.fc }
func (p *anthropicPartFunctionCall) ToolUseID() string       { return p.toolUseID }

type anthropicChatResponseIterator struct {
	ctx context.Context; cs *anthropicChatSession; stream *anthropic.MessagesStream
	err error; fullAssistantMessage anthropic.Message; stopReason *anthropic.MessageStopReason
	usageData UsageData; activeTextBlocks map[int]*strings.Builder
	activeToolInputs map[int]*strings.Builder; pendingParts []Part
}

func (cs *anthropicChatSession) SendStream(ctx context.Context, contents ...any) (ChatResponseIterator, error) {
	if cs.messagesAPI == nil {return nil, fmt.Errorf("Anthropic messages API not initialized")}
	if err := cs.processToMessageParams(contents...); err != nil {return nil, err}
	req := anthropic.MessagesRequest{Model: cs.model, Messages: cs.messages, MaxTokens: DefaultMaxTokensToSample, Stream: true}
	if cs.systemPrompt != "" {req.System = cs.systemPrompt}
	if len(cs.tools) > 0 {req.Tools = cs.tools}
	stream, err := cs.messagesAPI.CreateStream(ctx, req) // Use interface
	if err != nil {return nil, fmt.Errorf("Messages.CreateStream failed: %w", err)}
	return &anthropicChatResponseIterator{
		ctx: ctx, cs: cs, stream: stream,
		fullAssistantMessage: anthropic.Message{Role: anthropic.RoleAssistant, Content: make([]anthropic.ContentBlock, 0)},
		activeTextBlocks: make(map[int]*strings.Builder), activeToolInputs: make(map[int]*strings.Builder),
		pendingParts: make([]Part, 0),
	}, nil
}

func (it *anthropicChatResponseIterator) Next() (ChatResponse, error) {
	if it.err != nil {return nil, it.err}
	it.pendingParts = make([]Part, 0)
	for {
		event, err := it.stream.Recv()
		if err != nil {
			if errors.Is(err, io.EOF) {
				if len(it.fullAssistantMessage.Content) > 0 || it.stopReason != nil {
					if it.cs.client != nil { // Check if it's a real session
						it.cs.messages = append(it.cs.messages, it.fullAssistantMessage)
						for _, block := range it.fullAssistantMessage.Content {
							if tub, ok := block.(*anthropic.ToolUseBlock); ok {it.cs.lastToolUseIDs[tub.Name] = tub.ID}
						}
					}
				}
				it.err = io.EOF
				var finalParts []Part; if len(it.pendingParts) > 0 {finalParts = it.pendingParts} else {finalParts = []Part{}}
				var fr string; if it.stopReason != nil {fr = string(*it.stopReason)}
				// Return final accumulated usage and stop reason, even if parts are empty for this last signal
				return &anthropicChatResponse{candidates: []Candidate{&anthropicCandidate{parts: finalParts, finishReason: fr}}, usage: it.usageData}, io.EOF
			}
			it.err = err; return nil, err
		}
		var currentResp ChatResponse
		switch e := event.(type) {
		case *anthropic.MessagesEventMessageStart:
			it.fullAssistantMessage.Role = e.Message.Role
			it.fullAssistantMessage.Content = make([]anthropic.ContentBlock, len(e.Message.Content))
			it.usageData.PromptTokens = e.Message.Usage.InputTokens
		case *anthropic.MessagesEventContentBlockStart:
			idx := e.Index
			for len(it.fullAssistantMessage.Content) <= idx {it.fullAssistantMessage.Content = append(it.fullAssistantMessage.Content, nil)}
			switch cb := e.ContentBlock.(type) {
			case *anthropic.TextBlock:
				it.fullAssistantMessage.Content[idx] = &anthropic.TextBlock{Type: anthropic.ContentTypeText, Text: ""}
				it.activeTextBlocks[idx] = &strings.Builder{}
			case *anthropic.ToolUseBlock:
				it.fullAssistantMessage.Content[idx] = cb; it.activeToolInputs[idx] = &strings.Builder{}
			}
		case *anthropic.MessagesEventContentBlockDelta:
			idx := e.Index
			if idx >= len(it.fullAssistantMessage.Content) || it.fullAssistantMessage.Content[idx] == nil {
				log.Printf("Stream: Delta for uninitialized/OOB idx %d.", idx)
				if _, ok := e.Delta.(*anthropic.TextDelta); ok {
					for len(it.fullAssistantMessage.Content) <= idx {it.fullAssistantMessage.Content = append(it.fullAssistantMessage.Content, nil)}
					it.fullAssistantMessage.Content[idx] = &anthropic.TextBlock{Type: anthropic.ContentTypeText, Text: ""}
					it.activeTextBlocks[idx] = &strings.Builder{}
				} else {continue}
			}
			switch delta := e.Delta.(type) {
			case *anthropic.TextDelta:
				if tb, ok := it.fullAssistantMessage.Content[idx].(*anthropic.TextBlock); ok {
					tb.Text += delta.Text; it.pendingParts = append(it.pendingParts, &anthropicPartText{text: delta.Text})
				}
			case *anthropic.InputJSONDelta:
				if builder, ok := it.activeToolInputs[idx]; ok {builder.WriteString(delta.PartialJSON)}
			}
		case *anthropic.MessagesEventContentBlockStop:
			idx := e.Index
			if idx >= len(it.fullAssistantMessage.Content) {continue}
			if toolUseBlock, ok := it.fullAssistantMessage.Content[idx].(*anthropic.ToolUseBlock); ok {
				if builder, ok := it.activeToolInputs[idx]; ok {
					fullInputJSON := builder.String(); var inputData map[string]any
					if err := json.Unmarshal([]byte(fullInputJSON), &inputData); err != nil {
						log.Printf("Stream: Error unmarshalling tool input JSON for %s: %v. Raw: %s", toolUseBlock.Name, err, fullInputJSON)
						toolUseBlock.Input = map[string]any{"error": "failed to unmarshal stream", "raw_json": fullInputJSON}
					} else {toolUseBlock.Input = inputData}
					it.pendingParts = append(it.pendingParts, &anthropicPartFunctionCall{
						fc: FunctionCall{Name: toolUseBlock.Name, Arguments: fullInputJSON}, toolUseID: toolUseBlock.ID,
					}); delete(it.activeToolInputs, idx)
				}
			}
		case *anthropic.MessagesEventMessageDelta:
			if e.Delta.StopReason != nil {it.stopReason = &e.Delta.StopReason}
			it.usageData.CompletionTokens = e.Usage.OutputTokens
		case *anthropic.MessagesEventMessageStop, *anthropic.MessagesEventPing: continue
		}
		if len(it.pendingParts) > 0 {
			currentResp = &anthropicChatResponse{
				candidates: []Candidate{&anthropicCandidate{parts: it.pendingParts, finishReason: ""}}, usage: it.usageData,
			}; return currentResp, nil
		}
	}
}
func (it *anthropicChatResponseIterator) Close() error { return it.err }

func (cs *anthropicChatSession) History() []*Message {
	history := make([]*Message, 0, len(cs.messages))
	for _, sdkMsgParam := range cs.messages {
		var gollmMsg Message; var parts []Part
		switch m := sdkMsgParam.(type) {
		case anthropic.Message:
			if m.Role == anthropic.RoleUser {gollmMsg.Role = RoleUser} else if m.Role == anthropic.RoleAssistant {gollmMsg.Role = RoleAssistant} else {log.Printf("Warn: Unknown role %s", m.Role); gollmMsg.Role = RoleUser}
			for _, block := range m.Content {parts = append(parts, cs.contentBlockToGollmPart(block)...)}
		case *anthropic.UserTextMessageParam:
			gollmMsg.Role = RoleUser; parts = append(parts, &anthropicPartText{text: m.Text})
		case *anthropic.ToolResultParam:
			gollmMsg.Role = RoleUser; var contentStrings []string
			for _, block := range m.Content {if tb, ok := block.(*anthropic.TextBlock); ok {contentStrings = append(contentStrings, tb.Text)}}
			parts = append(parts, &anthropicPartText{text: fmt.Sprintf("Tool Result (ID: %s): %s", m.ToolUseID, strings.Join(contentStrings, "\n"))})
		default: log.Printf("Warn: Unhandled MessageParam type %T in History()", sdkMsgParam); continue
		}
		gollmMsg.Parts = parts; history = append(history, &gollmMsg)
	}
	return history
}
func (cs *anthropicChatSession) contentBlockToGollmPart(block anthropic.ContentBlock) []Part {
	var parts []Part
	switch b := block.(type) {
	case *anthropic.TextBlock: parts = append(parts, &anthropicPartText{text: b.Text})
	case *anthropic.ToolUseBlock:
		inputJSON, _ := json.Marshal(b.Input)
		parts = append(parts, &anthropicPartFunctionCall{fc: FunctionCall{Name: b.Name, Arguments: string(inputJSON)}, toolUseID: b.ID})
	default: log.Printf("Warn: Unknown ContentBlock type %T in contentBlockToGollmPart", b)
	}
	return parts
}
func (cs *anthropicChatSession) ClearHistory() {
	cs.messages = make([]anthropic.MessageParam, 0); cs.lastToolUseIDs = make(map[string]string)
}
func (cs *anthropicChatSession) IsRetryableError(err error) bool {return DefaultIsRetryableError(err)}

var _ Client = (*AnthropicClient)(nil)
var _ Chat = (*anthropicChatSession)(nil)
var _ Part = (*anthropicPartText)(nil)
var _ Part = (*anthropicPartFunctionCall)(nil)
var _ Candidate = (*anthropicCandidate)(nil)
var _ ChatResponse = (*anthropicChatResponse)(nil)
var _ ChatResponseIterator = (*anthropicChatResponseIterator)(nil)
