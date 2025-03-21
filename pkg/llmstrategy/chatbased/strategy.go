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

package chatbased

import (
	"context"
	_ "embed"
	"encoding/json"
	"fmt"
	"html/template"
	"os"
	"sort"
	"strings"
	"time"

	"github.com/GoogleCloudPlatform/kubectl-ai/gollm"
	"github.com/GoogleCloudPlatform/kubectl-ai/pkg/journal"
	"github.com/GoogleCloudPlatform/kubectl-ai/pkg/llmstrategy"
	"github.com/GoogleCloudPlatform/kubectl-ai/pkg/tools"
	"github.com/GoogleCloudPlatform/kubectl-ai/pkg/ui"
	"k8s.io/klog/v2"
)

//go:embed chatbased_systemprompt_template_default.txt
var defaultSystemPromptChatAgent string

type Strategy struct {
	LLM gollm.Client

	// PromptTemplateFile allows specifying a custom template file
	PromptTemplateFile string

	// Recorder captures events for diagnostics
	Recorder journal.Recorder

	RemoveWorkDir bool

	MaxIterations int

	Kubeconfig          string
	AsksForConfirmation bool

	Tools tools.Tools

	EnableToolUseShim bool
}

type Conversation struct {
	strategy *Strategy

	// Recorder captures events for diagnostics
	recorder journal.Recorder

	UI      ui.UI
	llmChat gollm.Chat

	workDir string
}

func (s *Strategy) NewConversation(ctx context.Context, u ui.UI) (llmstrategy.Conversation, error) {
	log := klog.FromContext(ctx)

	// Create a temporary working directory
	workDir, err := os.MkdirTemp("", "agent-workdir-*")
	if err != nil {
		log.Error(err, "Failed to create temporary working directory")
		return nil, err
	}

	log.Info("Created temporary working directory", "workDir", workDir)

	systemPrompt, err := s.generatePrompt(ctx, defaultSystemPromptChatAgent, PromptData{
		Tools:             s.Tools,
		EnableToolUseShim: s.EnableToolUseShim,
	})
	if err != nil {
		log.Error(err, "Failed to generate system prompt")
		return nil, err
	}

	// Start a new chat session
	llmChat := s.LLM.StartChat(systemPrompt)

	if !s.EnableToolUseShim {
		var functionDefinitions []*gollm.FunctionDefinition
		for _, tool := range s.Tools.AllTools() {
			functionDefinitions = append(functionDefinitions, tool.FunctionDefinition())
		}
		// Sort function definitions to help KV cache reuse
		sort.Slice(functionDefinitions, func(i, j int) bool {
			return functionDefinitions[i].Name < functionDefinitions[j].Name
		})
		if err := llmChat.SetFunctionDefinitions(functionDefinitions); err != nil {
			return nil, fmt.Errorf("setting function definitions: %w", err)
		}
	}

	return &Conversation{
		strategy: s,
		recorder: s.Recorder,
		UI:       u,
		llmChat:  llmChat,
		workDir:  workDir,
	}, nil
}

func (c *Conversation) Close() error {
	if c.workDir != "" {
		if c.strategy.RemoveWorkDir {
			if err := os.RemoveAll(c.workDir); err != nil {
				klog.Warningf("error cleaning up directory %q: %v", c.workDir, err)
			}
		}
	}
	return nil
}

// RunOneRound executes a chat-based agentic loop with the LLM using function calling.
func (a *Conversation) RunOneRound(ctx context.Context, query string) error {
	log := klog.FromContext(ctx)
	log.Info("Starting chat loop for query:", "query", query)

	// currChatContent tracks chat content that needs to be sent
	// to the LLM in each iteration of  the agentic loop below
	var currChatContent []any

	// Set the initial message to start the conversation
	currChatContent = []any{query} //fmt.Sprintf("can you help me with query: %q", query)}

	currentIteration := 0
	maxIterations := a.strategy.MaxIterations

	for currentIteration < maxIterations {
		log.Info("Starting iteration", "iteration", currentIteration)

		a.recorder.Write(ctx, &journal.Event{
			Timestamp: time.Now(),
			Action:    "llm-chat",
			Payload:   []any{currChatContent},
		})

		response, err := a.llmChat.Send(ctx, currChatContent...)
		if err != nil {
			log.Error(err, "Error sending initial message")
			return err
		}

		a.recorder.Write(ctx, &journal.Event{
			Timestamp: time.Now(),
			Action:    "llm-response",
			Payload:   response,
		})

		currChatContent = nil

		if len(response.Candidates()) == 0 {
			log.Error(nil, "No candidates in response")
			return fmt.Errorf("no candidates in LLM response")
		}

		candidate := response.Candidates()[0]

		// Process each part of the response
		// only applicable is not using tooluse shim
		var functionCalls []gollm.FunctionCall

		for _, part := range candidate.Parts() {
			// Check if it's a text response
			if text, ok := part.AsText(); ok {
				log.Info("text response", "text", text)
				textResponse := text

				if a.strategy.EnableToolUseShim {
					reActResp, err := parseReActResponse(textResponse)
					if err != nil {
						log.Error(err, "Error parsing ReAct response")
						a.UI.RenderOutput(ctx, fmt.Sprintf("\nSorry, Couldn't complete the task. LLM error %v\n", err), ui.Foreground(ui.ColorRed))
						return err
					}

					if reActResp.Answer != "" {
						a.UI.RenderOutput(ctx, reActResp.Answer, ui.RenderMarkdown())
						return nil
					}
					// Handle action
					if reActResp.Action != nil {
						currChatContent = append(currChatContent, reActResp.Thought)
						currChatContent = append(currChatContent, reActResp.Action.Reason)
						// Sanitize and prepare action
						reActResp.Action.Command = strings.TrimSpace(reActResp.Action.Command)

						functionCallName := reActResp.Action.Name
						functionCallArgs, err := toMap(reActResp.Action)
						if err != nil {
							log.Error(err, "Error converting action to map")
							return err
						}
						delete(functionCallArgs, "name") // passed separately
						// TODO(droot): Hack: deleting fields from args because I don't like the output from the toolCall.prettyPrint. Pl. fix me.
						delete(functionCallArgs, "reason")
						delete(functionCallArgs, "modifies_resource")

						toolCall, err := a.strategy.Tools.ParseToolInvocation(ctx, functionCallName, functionCallArgs)
						if err != nil {
							return fmt.Errorf("building tool call: %w", err)
						}

						a.UI.RenderOutput(ctx, reActResp.Action.Reason, ui.RenderMarkdown())
						// Display action details
						s := toolCall.PrettyPrint()
						a.UI.RenderOutput(ctx, fmt.Sprintf("  Running: %s\n", s), ui.Foreground(ui.ColorGreen))

						if a.strategy.AsksForConfirmation && reActResp.Action.ModifiesResource == "yes" {
							confirm := a.UI.AskForConfirmation(ctx, "  Are you sure you want to run this command (Y/n)?")
							if !confirm {
								a.UI.RenderOutput(ctx, "Sure.\n", ui.RenderMarkdown())
								return nil
							}
						}

						ctx := journal.ContextWithRecorder(ctx, a.recorder)

						output, err := toolCall.InvokeTool(ctx, tools.InvokeToolOptions{
							WorkDir: a.workDir,
						})
						if err != nil {
							return fmt.Errorf("executing action: %w", err)
						}

						observation := fmt.Sprintf("Result of running %q:\n%s", reActResp.Action.Command, output)
						currChatContent = append(currChatContent, observation)
					}
					continue
				} else {
					// If we have a text response, render it
					if textResponse != "" {
						a.UI.RenderOutput(ctx, textResponse, ui.RenderMarkdown())
					}
				}
			}

			if !a.strategy.EnableToolUseShim {
				continue
			}

			// Check if it's a function call
			if calls, ok := part.AsFunctionCalls(); ok && len(calls) > 0 {
				log.Info("function calls", "calls", calls)
				functionCalls = append(functionCalls, calls...)

				// TODO(droot): Run all function calls in parallel
				// (may have to specify in the prompt to make these function calls independent)
				for _, call := range calls {
					toolCall, err := a.strategy.Tools.ParseToolInvocation(ctx, call.Name, call.Arguments)
					if err != nil {
						return fmt.Errorf("building tool call: %w", err)
					}

					s := toolCall.PrettyPrint()
					a.UI.RenderOutput(ctx, fmt.Sprintf("  Running: %s\n", s), ui.Foreground(ui.ColorGreen))
					if a.strategy.AsksForConfirmation && call.Arguments["modifies_resource"] == "no" {
						confirm := a.UI.AskForConfirmation(ctx, "  Are you sure you want to run this command (Y/n)? ")
						if !confirm {
							a.UI.RenderOutput(ctx, "Sure.\n", ui.RenderMarkdown())
							return nil
						}
					}

					ctx := journal.ContextWithRecorder(ctx, a.recorder)
					output, err := toolCall.InvokeTool(ctx, tools.InvokeToolOptions{
						WorkDir: a.workDir,
					})
					if err != nil {
						return fmt.Errorf("executing action: %w", err)
					}

					result, err := toResult(output)
					if err != nil {
						return err
					}

					currChatContent = append(currChatContent, gollm.FunctionCallResult{
						Name:   call.Name,
						Result: result,
					})

				}
			}
		}

		// If no function calls were made, we're done
		if len(functionCalls) == 0 && !a.strategy.EnableToolUseShim {
			return nil
		}

		currentIteration++
	}

	// If we've reached the maximum number of iterations
	log.Info("Max iterations reached", "iterations", maxIterations)
	a.UI.RenderOutput(ctx, fmt.Sprintf("\nSorry, couldn't complete the task after %d iterations.\n", maxIterations), ui.Foreground(ui.ColorRed))
	return fmt.Errorf("max iterations reached")
}

// toResult converts an arbitrary result to a map[string]any
func toResult(v any) (map[string]any, error) {
	b, err := json.Marshal(v)
	if err != nil {
		return nil, fmt.Errorf("converting result to json: %w", err)
	}

	m := make(map[string]any)
	if err := json.Unmarshal(b, &m); err != nil {
		return nil, fmt.Errorf("converting json result to map: %w", err)
	}
	return m, nil
}

// generateFromTemplate generates a prompt for LLM. It uses the prompt from the provides template file or default.
func (a *Strategy) generatePrompt(_ context.Context, defaultPromptTemplate string, data PromptData) (string, error) {
	promptTemplate := defaultPromptTemplate
	if a.PromptTemplateFile != "" {
		content, err := os.ReadFile(a.PromptTemplateFile)
		if err != nil {
			return "", fmt.Errorf("error reading template file: %v", err)
		}
		promptTemplate = string(content)
	}

	tmpl, err := template.New("promptTemplate").Parse(promptTemplate)
	if err != nil {
		return "", fmt.Errorf("building template for prompt: %w", err)
	}

	var result strings.Builder
	err = tmpl.Execute(&result, &data)
	if err != nil {
		return "", fmt.Errorf("evaluating template for prompt: %w", err)
	}
	return result.String(), nil
}

// PromptData represents the structure of the data to be filled into the template.
type PromptData struct {
	Query string
	Tools tools.Tools

	EnableToolUseShim bool
}

func (a *PromptData) ToolsAsJSON() string {
	var toolDefinitions []*gollm.FunctionDefinition

	for _, tool := range a.Tools.AllTools() {
		toolDefinitions = append(toolDefinitions, tool.FunctionDefinition())
	}

	json, err := json.MarshalIndent(toolDefinitions, "", "  ")
	if err != nil {
		return ""
	}
	return string(json)
}

func (a *PromptData) ToolNames() string {
	return strings.Join(a.Tools.Names(), ", ")
}

type ReActResponse struct {
	Thought string  `json:"thought"`
	Answer  string  `json:"answer,omitempty"`
	Action  *Action `json:"action,omitempty"`
}

type Action struct {
	Name             string `json:"name"`
	Reason           string `json:"reason"`
	Command          string `json:"command"`
	ModifiesResource string `json:"modifies_resource"`
}

// parseReActResponse parses the LLM response into a ReActResponse struct
// This function assumes the input contains exactly one JSON code block
// formatted with ```json and ``` markers. The JSON block is expected to
// contain a valid ReActResponse object.
func parseReActResponse(input string) (*ReActResponse, error) {
	cleaned := strings.TrimSpace(input)

	const jsonBlockMarker = "```json"
	first := strings.Index(cleaned, jsonBlockMarker)
	last := strings.LastIndex(cleaned, "```")
	if first == -1 || last == -1 {
		return nil, fmt.Errorf("no JSON code block found in %q", cleaned)
	}
	cleaned = cleaned[first+len(jsonBlockMarker) : last]

	cleaned = strings.ReplaceAll(cleaned, "\n", "")
	cleaned = strings.TrimSpace(cleaned)

	var reActResp ReActResponse
	if err := json.Unmarshal([]byte(cleaned), &reActResp); err != nil {
		return nil, fmt.Errorf("parsing JSON %q: %w", cleaned, err)
	}
	return &reActResp, nil
}

// toMap converts the value to a map, going via JSON
func toMap(v any) (map[string]any, error) {
	j, err := json.Marshal(v)
	if err != nil {
		return nil, fmt.Errorf("converting %T to json: %w", v, err)
	}
	m := make(map[string]any)
	if err := json.Unmarshal(j, &m); err != nil {
		return nil, fmt.Errorf("converting json to map: %w", err)
	}
	return m, nil
}
