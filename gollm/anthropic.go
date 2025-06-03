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
	anthropicoption "github.com/anthropics/anthropic-sdk-go/option"
	// "github.com/anthropics/anthropic-sdk-go/pkg/ssestream" // Temporarily removed
)

const (
	defaultAnthropicModel    = "claude-3-haiku-20240307"
	DefaultMaxTokensToSample = 2048
)

// anthropicMessagesAPI defines the interface for Anthropic's messages service,
// allowing for mocking in tests.
type anthropicMessagesAPI interface {
	New(context.Context, anthropic.MessageNewParams, ...anthropicoption.RequestOption) (*anthropic.Message, error)
	// NewStreaming(context.Context, anthropic.MessageNewParams, ...anthropicoption.RequestOption) (*ssestream.Stream[anthropic.MessageStreamEventUnion], error) // Temporarily removed
}

// AnthropicClient represents a client for the Anthropic API.
type AnthropicClient struct {
	messagesAPI  anthropicMessagesAPI 
	rawSDKClient *anthropic.Client    
	defaultModel string
	apiKey       string
}

// anthropicChatSession represents a chat session with the Anthropic API.
type anthropicChatSession struct {
	client              *AnthropicClient 
	messagesAPI         anthropicMessagesAPI
	model               string
	systemPrompt        string
	messages            []anthropic.MessageParam 
	tools               []anthropic.ToolParam    
	functionDefinitions []*FunctionDefinition
	lastToolUseIDs      map[string]string 
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
	sdkClient, err := anthropic.NewClient(anthropicoption.WithAPIKey(opts.APIKey)) 
	if err != nil {
		return nil, fmt.Errorf("failed to create Anthropic SDK client: %w", err)
	}
	return &AnthropicClient{
		messagesAPI:  sdkClient.Messages,
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
	if c.messagesAPI == nil {
		return nil, fmt.Errorf("Anthropic messages API not initialized")
	}
	return &anthropicChatSession{
		client:           c,
		messagesAPI:      c.messagesAPI,
		model:            chatModel,
		systemPrompt:     systemPrompt,
		messages:         make([]anthropic.MessageParam, 0),
		lastToolUseIDs:   make(map[string]string),
	}, nil
}

func (c *AnthropicClient) GenerateCompletion(ctx context.Context, req *CompletionRequest) (CompletionResponse, error) {
	if c.messagesAPI == nil {return nil, fmt.Errorf("SDK client not initialized")}
	modelName := req.Model; if modelName == "" {modelName = c.defaultModel}
	
	anthropicMessages := []anthropic.MessageParam{anthropic.NewUserMessage([]anthropic.ContentBlockParamUnion{
		anthropic.NewTextBlock(req.Prompt),
	})}
	
	maxTokens := DefaultMaxTokensToSample; if req.MaxTokens > 0 {maxTokens = req.MaxTokens}
	
	var tempOpt anthropic.Opt[float64]
	if req.Temperature != nil {
		tempOpt = anthropic.Float(*req.Temperature)
	}

	anthropicReq := anthropic.MessageNewParams{ 
		Model:       anthropic.Model(modelName), 
		Messages:    anthropicMessages,
		MaxTokens:   int64(maxTokens), 
		Temperature: tempOpt,
	}

	if err := ctx.Err(); err != nil {return nil, err}
	resp, err := c.messagesAPI.New(ctx, anthropicReq) 
	if err != nil {return nil, fmt.Errorf("API request failed: %w", err)}
	if len(resp.Content) == 0 {return nil, fmt.Errorf("API returned no content")}
	var responseText string
	if len(resp.Content) > 0 {
		firstBlock := resp.Content[0]
		if textBlock, ok := firstBlock.AsAny().(anthropic.TextBlock); ok {
			responseText = textBlock.Text
		} else {
			return nil, fmt.Errorf("unexpected first content block type: %T", firstBlock.AsAny())
		}
	} else {
		return nil, fmt.Errorf("API returned no content blocks")
	}
	return &anthropicCompletionResponse{text: responseText}, nil
}

type anthropicCompletionResponse struct{ text string }
func (r *anthropicCompletionResponse) Text() string { return r.text }

func (c *AnthropicClient) SetResponseSchema(schema *Schema) error { // Corrected to *Schema
	log.Println("Warning: SetResponseSchema not supported by Anthropic. Use Tool definitions.")
	return nil
}

func (c *AnthropicClient) ListModels(ctx context.Context) ([]string, error) {
	modelInfos := []ModelInfo{ 
		{Name: "claude-3-opus-20240229", Description: "Most powerful.", MaxTokens: 200000},
		{Name: "claude-3-sonnet-20240229", Description: "Balance of intelligence/speed.", MaxTokens: 200000},
		{Name: "claude-3-haiku-20240307", Description: "Fastest, most compact.", MaxTokens: 200000, Default: true},
	}
	if err := ctx.Err(); err != nil {return nil, err}
	names := make([]string, len(modelInfos))
	for i, mi := range modelInfos {
		names[i] = mi.Name
	}
	return names, nil
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
		contentBlocks := make([]anthropic.ContentBlockParamUnion, 0, len(gMsg.Parts)) 
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
				contentBlocks = append(contentBlocks, anthropic.NewToolResultBlock(toolUseID, resultStr, false)) 
			}
		}
		if len(contentBlocks) == 0 {log.Printf("Warning: Gollm message at index %d resulted in no content blocks.", i); continue}
		var role anthropic.MessageParamRole 
		switch gMsg.Role {
		case RoleUser: role = anthropic.MessageParamRoleUser
		case RoleAssistant: role = anthropic.MessageParamRoleAssistant
		default: return nil, "", fmt.Errorf("unsupported gollm role for message conversion: %s", gMsg.Role)
		}
		anthropicMessages = append(anthropicMessages, anthropic.MessageParam{Role: role, Content: contentBlocks})
	}
	return anthropicMessages, systemPrompt, nil
}

func (c *AnthropicClient) GenerateContent(ctx context.Context, req *GenerateRequest) (*GenerateResponse, error) {
	if c.messagesAPI == nil {return nil, fmt.Errorf("Anthropic messages API not initialized")}
	anthropicMsgs, systemPrompt, err := gollmMessagesToAnthropic(req.Messages, make(map[string]string))
	if err != nil {return nil, fmt.Errorf("failed to convert gollm messages: %w", err)}
	modelName := c.defaultModel; if req.Model != "" {modelName = req.Model}
	
	anthropicAPIReq := anthropic.MessageNewParams{ 
		Model:    anthropic.Model(modelName), 
		Messages: anthropicMsgs,
	}
	if systemPrompt != "" {anthropicAPIReq.System = []anthropic.TextBlockParam{{Text: systemPrompt}}}
	if req.MaxOutputTokens > 0 {anthropicAPIReq.MaxTokens = int64(req.MaxOutputTokens)} else {anthropicAPIReq.MaxTokens = DefaultMaxTokensToSample}
	if req.Temperature != nil {anthropicAPIReq.Temperature = anthropic.Float(*req.Temperature)}
	if req.TopP != nil {anthropicAPIReq.TopP = anthropic.Float(*req.TopP)}
	if req.TopK > 0 {anthropicAPIReq.TopK = anthropic.Int(int64(req.TopK))} 

	if len(req.Tools) > 0 {
		anthropicTools := make([]anthropic.ToolParam, len(req.Tools)) 
		for i, toolDef := range req.Tools {
			var params map[string]any
			if toolDef.Parameters != nil {
				if pMap, ok := toolDef.Parameters.(map[string]any); ok {params = pMap} else {
					b, err := json.Marshal(toolDef.Parameters); if err != nil {return nil, fmt.Errorf("marshal params for %s: %w", toolDef.Name, err)}
					if err := json.Unmarshal(b, &params); err != nil {return nil, fmt.Errorf("unmarshal params for %s: %w", toolDef.Name, err)}
				}
			}
			anthropicTools[i] = anthropic.ToolParam{ 
				Name: toolDef.Name, Description: anthropic.String(toolDef.Description), 
				InputSchema: anthropic.ToolInputSchemaParam{Type: anthropic.ToolInputSchemaParamTypeObject, Properties: params}, 
			}
		}
		toolUnionParams := make([]anthropic.ToolUnionParam, len(anthropicTools))
		for i, tp := range anthropicTools {
			toolUnionParams[i] = anthropic.ToolUnionParamOfTool(tp)
		}
		anthropicAPIReq.Tools = toolUnionParams
	}
	if err := ctx.Err(); err != nil {return nil, err}
	apiResp, err := c.messagesAPI.New(ctx, anthropicAPIReq) 
	if err != nil {return nil, fmt.Errorf("Anthropic Messages.Create API call failed: %w", err)}
	
	gollmParts := make([]Part, 0, len(apiResp.Content))
	for _, block := range apiResp.Content { 
		switch b := block.AsAny().(type) { 
		case anthropic.TextBlock: gollmParts = append(gollmParts, &anthropicPartText{PartBase{}, b.Text}) 
		case anthropic.ToolUseBlock:
			inputJSON, _ := json.Marshal(b.Input)
			gollmParts = append(gollmParts, &anthropicPartFunctionCall{PartBase{}, FunctionCall{Name: b.Name, Arguments: string(inputJSON)}, b.ID}) 
		default: log.Printf("GenerateContent: Unknown Anthropic content block type in response: %T", b)
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
	return nil, fmt.Errorf("GenerateContentStream is temporarily disabled due to SDK package resolution issues")
}
func (c *AnthropicClient) Name() string { return "anthropic" }

// --- anthropicChatSession Methods (gollm.Chat interface) ---
func (cs *anthropicChatSession) SetFunctionDefinitions(defs []*FunctionDefinition) error {
	cs.functionDefinitions = defs; cs.tools = make([]anthropic.ToolParam, len(defs)) 
	for i, def := range defs {
		var params map[string]any
		if def.Parameters != nil {
			if pMap, ok := def.Parameters.(map[string]any); ok {params = pMap} else {
				b, err := json.Marshal(def.Parameters); if err != nil {return fmt.Errorf("marshal params for %s: %w", def.Name, err)}
				if err := json.Unmarshal(b, &params); err != nil {return fmt.Errorf("unmarshal params for %s: %w", def.Name, err)}
			}
		}
		cs.tools[i] = anthropic.ToolParam{Name: def.Name, Description: anthropic.String(def.Description), InputSchema: anthropic.ToolInputSchemaParam{Type: anthropic.ToolInputSchemaParamTypeObject, Properties: params}}
	}
	return nil
}

func (cs *anthropicChatSession) processToMessageParams(contents ...any) error {
	for _, content := range contents {
		switch c := content.(type) {
		case string: cs.messages = append(cs.messages, anthropic.NewUserMessage([]anthropic.ContentBlockParamUnion{anthropic.NewTextBlock(c)}))
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
							if tub, okTub := block.AsAny().(anthropic.ToolUseBlock); okTub && tub.Name == c.ToolName { 
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
			cs.messages = append(cs.messages, anthropic.NewUserMessage([]anthropic.ContentBlockParamUnion{anthropic.NewToolResultBlock(toolUseID, resultStr, false)}))
		default: return fmt.Errorf("unsupported content type: %T", c)
		}
	}
	return nil
}

func (cs *anthropicChatSession) Send(ctx context.Context, contents ...any) (ChatResponse, error) {
	if cs.messagesAPI == nil {return nil, fmt.Errorf("Anthropic messages API not initialized")}
	if err := cs.processToMessageParams(contents...); err != nil {return nil, err}
	
	anthropicAPIReq := anthropic.MessageNewParams{ 
		Model:    anthropic.Model(cs.model), 
		Messages: cs.messages, 
		MaxTokens: DefaultMaxTokensToSample, 
	}
	if cs.systemPrompt != "" {anthropicAPIReq.System = []anthropic.TextBlockParam{{Text: cs.systemPrompt}}}
	if len(cs.tools) > 0 {
		toolUnionParams := make([]anthropic.ToolUnionParam, len(cs.tools))
		for i, tp := range cs.tools { toolUnionParams[i] = anthropic.ToolUnionParamOfTool(tp) }
		anthropicAPIReq.Tools = toolUnionParams
	}

	if err := ctx.Err(); err != nil {return nil, err}
	resp, err := cs.messagesAPI.New(ctx, anthropicAPIReq) 
	if err != nil {return nil, fmt.Errorf("Messages.Create failed: %w", err)}
	
	assistantMessage := anthropic.Message{Role: anthropic.RoleAssistant, Content: resp.Content} 
	cs.messages = append(cs.messages, assistantMessage) 
	for _, block := range resp.Content {
		if tub, ok := block.AsAny().(anthropic.ToolUseBlock); ok {cs.lastToolUseIDs[tub.Name] = tub.ID}
	}
	return cs.convertToGollmChatResponse(resp, nil), nil
}

func (cs *anthropicChatSession) convertToGollmChatResponse(resp *anthropic.Message, streamStopReason *anthropic.MessageStopReason) ChatResponse {
	gollmParts := make([]Part, 0, len(resp.Content))
	for _, block := range resp.Content { 
		switch b := block.AsAny().(type) { 
		case anthropic.TextBlock: gollmParts = append(gollmParts, &anthropicPartText{PartBase{}, b.Text})
		case anthropic.ToolUseBlock:
			inputJSON, err := json.Marshal(b.Input); argsStr := string(inputJSON)
			if err != nil {log.Printf("Warning: Marshal tool input for %s: %v", b.Name, err); argsStr = fmt.Sprintf(`{"error":"marshal input: %v", "raw":"%+v"}`, err, b.Input)}
			gollmParts = append(gollmParts, &anthropicPartFunctionCall{PartBase{}, FunctionCall{Name: b.Name, Arguments: argsStr}, b.ID})
		default: log.Printf("Warning: Unknown Anthropic content block type: %T", b)
		}
	}
	var finishReason string
	if resp.StopReason != nil {finishReason = string(*resp.StopReason)} else if streamStopReason != nil {finishReason = string(*streamStopReason)}
	usage := UsageData{}; if resp.Usage != nil { 
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
func (c *anthropicCandidate) String() string { 
	if len(c.parts) > 0 {
		if tp, ok := c.parts[0].(TextProducer); ok { 
			return tp.Text() 
		}
	}
	return ""
}

type anthropicPartText struct{ PartBase; text string } 
func (p *anthropicPartText) Text() string            { return p.text }
func (p *anthropicPartText) FunctionCall() *FunctionCall { return nil }
func (p *anthropicPartText) FunctionCallResult() *FunctionCallResult { return nil }


type anthropicPartFunctionCall struct{ PartBase; fc FunctionCall; toolUseID string } 
func (p *anthropicPartFunctionCall) Text() string            { return "" }
func (p *anthropicPartFunctionCall) FunctionCall() *FunctionCall { return &p.fc }
func (p *anthropicPartFunctionCall) FunctionCallResult() *FunctionCallResult { return nil }


type anthropicChatResponseIterator struct {
	ctx context.Context; cs *anthropicChatSession; // stream *ssestream.Stream[anthropic.MessageStreamEventUnion] // Temporarily removed
	err error; fullAssistantMessage anthropic.Message; stopReason *anthropic.MessageStopReason 
	usageData UsageData; activeTextBlocks map[int]*strings.Builder
	activeToolInputs map[int]*strings.Builder; pendingParts []Part
}

// SendStream ensures the method name matches the Chat interface.
func (cs *anthropicChatSession) SendStream(ctx context.Context, contents ...any) (ChatResponseIterator, error) {
	return nil, fmt.Errorf("SendStream is temporarily disabled due to SDK package resolution issues")
}

func (it *anthropicChatResponseIterator) Next() (ChatResponse, error) {
	if it.err == nil { 
	    it.err = fmt.Errorf("Next() called on a disabled stream iterator due to SDK package issues")
	}
	return nil, it.err
}
func (it *anthropicChatResponseIterator) Close() error { return it.err }

func (cs *anthropicChatSession) History() []*Message {
	history := make([]*Message, 0, len(cs.messages))
	for _, sdkMsgParam := range cs.messages {
		var gollmMsg Message; var parts []Part
		currentMsg := sdkMsgParam.(anthropic.Message) 

		if currentMsg.Role == anthropic.RoleUser {gollmMsg.Role = RoleUser} else if currentMsg.Role == anthropic.RoleAssistant {gollmMsg.Role = RoleAssistant} else {log.Printf("Warn: Unknown role %s", currentMsg.Role); gollmMsg.Role = RoleUser}
		for _, block := range currentMsg.Content {parts = append(parts, cs.contentBlockToGollmPart(block)...)}
		
		gollmMsg.Parts = parts; history = append(history, &gollmMsg)
	}
	return history
}
func (cs *anthropicChatSession) contentBlockToGollmPart(block anthropic.ContentBlockUnion) []Part {
	var parts []Part
	switch b := block.AsAny().(type) { 
	case anthropic.TextBlock: parts = append(parts, &anthropicPartText{PartBase{}, b.Text})
	case anthropic.ToolUseBlock:
		inputJSON, _ := json.Marshal(b.Input) 
		parts = append(parts, &anthropicPartFunctionCall{PartBase{}, FunctionCall{Name: b.Name, Arguments: string(inputJSON)}, b.ID})
	case anthropic.ToolResultBlock: 
		parts = append(parts, &anthropicPartText{PartBase{}, fmt.Sprintf("Tool Result (ID: %s): %s", b.ToolUseID, string(b.Content))})
	default: log.Printf("Warn: Unknown ContentBlockUnion type %T in contentBlockToGollmPart", b)
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
