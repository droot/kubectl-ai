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

package react

import (
	"context"
	_ "embed"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"text/template"
	"time"

	"github.com/GoogleCloudPlatform/kubectl-ai/gollm"
	"github.com/GoogleCloudPlatform/kubectl-ai/pkg/journal"
	"github.com/GoogleCloudPlatform/kubectl-ai/pkg/llmstrategy"
	"github.com/GoogleCloudPlatform/kubectl-ai/pkg/tools"
	"github.com/GoogleCloudPlatform/kubectl-ai/pkg/ui"
	"k8s.io/klog/v2"
)

//go:embed react_systemprompt_template_default.txt
var defaultSystemPromptTemplate string

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
}

type Conversation struct {
	strategy *Strategy

	recorder journal.Recorder

	UI      ui.UI
	llmChat gollm.Chat

	MaxIterations int

	workDir string
}

func (s *Strategy) NewConversation(ctx context.Context, userInterface ui.UI) (llmstrategy.Conversation, error) {
	log := klog.FromContext(ctx)

	// Create a temporary working directory
	workDir, err := os.MkdirTemp("", "agent-workdir-*")
	if err != nil {
		return nil, fmt.Errorf("creating temporary working directory: %w", err)
	}

	log.Info("Created temporary working directory", "workDir", workDir)
	data := PromptData{
		Tools: s.Tools,
	}

	systemPrompt, err := s.generatePrompt(ctx, defaultSystemPromptTemplate, data)
	if err != nil {
		log.Error(err, "Failed to generate system prompt")
		return nil, err
	}

	llmChat := s.LLM.StartChat(systemPrompt)

	return &Conversation{
		strategy:      s,
		workDir:       workDir,
		recorder:      s.Recorder,
		UI:            userInterface,
		llmChat:       llmChat,
		MaxIterations: s.MaxIterations,
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
	log.Info("Executing query:", "query", query)

	var currChatContent []any

	currChatContent = []any{query}

	u := a.UI

	currentIteration := 0
	// Main execution loop
	for currentIteration < a.MaxIterations {
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

		if len(response.Candidates()) == 0 {
			log.Error(nil, "No candidates in response")
			return fmt.Errorf("no candidates in LLM response")
		}

		candidate := response.Candidates()[0]

		for _, part := range candidate.Parts() {
			text, ok := part.AsText()
			if !ok {
				continue
			}
			if text == "" {
				log.Info("empty text response")
				continue
			}

			log.Info("text response", "text", text)
			textResponse := text
			reActResp, err := parseReActResponse(textResponse)
			if err != nil {
				log.Error(err, "Error parsing ReAct response")
				u.RenderOutput(ctx, fmt.Sprintf("\nSorry, Couldn't complete the task. LLM error %v\n", err), ui.Foreground(ui.ColorRed))
				return err
			}

			if reActResp.Answer != "" {
				u.RenderOutput(ctx, reActResp.Answer, ui.RenderMarkdown())
				return nil
			}
			// Handle action
			if reActResp.Action != nil {
				currChatContent = append(currChatContent, reActResp.Thought)
				currChatContent = append(currChatContent, reActResp.Action.Reason)
				// Sanitize and prepare action
				reActResp.Action.Command = sanitizeToolInput(reActResp.Action.Command)

				functionCallName := reActResp.Action.Name
				functionCallArgs, err := toMap(reActResp.Action)
				if err != nil {
					return err
				}
				delete(functionCallArgs, "name") // passed separately
				delete(functionCallArgs, "reason")
				delete(functionCallArgs, "modifies_resource")

				toolCall, err := a.strategy.Tools.ParseToolInvocation(ctx, functionCallName, functionCallArgs)
				if err != nil {
					return fmt.Errorf("building tool call: %w", err)
				}

				u.RenderOutput(ctx, reActResp.Action.Reason, ui.RenderMarkdown())
				// Display action details
				s := toolCall.PrettyPrint()
				u.RenderOutput(ctx, fmt.Sprintf("  Running: %s\n", s), ui.Foreground(ui.ColorGreen))

				if a.strategy.AsksForConfirmation && reActResp.Action.ModifiesResource == "yes" {
					confirm := u.AskForConfirmation(ctx, "  Are you sure you want to run this command (Y/n)?")
					if !confirm {
						u.RenderOutput(ctx, "Sure.\n", ui.RenderMarkdown())
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
		}
		currentIteration++
	}

	// Handle max iterations reached
	log.Info("Max iterations reached", "iterations", currentIteration)
	u.RenderOutput(ctx, fmt.Sprintf("\nSorry, Couldn't complete the task after %d attempts.\n", a.MaxIterations), ui.Foreground(ui.ColorRed))
	return a.recordError(ctx, fmt.Errorf("max iterations reached"))
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

// executeAction handles the execution of a single action
func (a *Conversation) executeAction(ctx context.Context, tool *tools.ToolCall, workDir string) (string, error) {
	ctx = journal.ContextWithRecorder(ctx, a.recorder)

	output, err := tool.InvokeTool(ctx, tools.InvokeToolOptions{
		WorkDir: a.workDir,
	})
	if err != nil {
		return "", err
	}

	switch output := output.(type) {
	case string:
		return output, nil
	default:
		b, err := json.Marshal(output)
		if err != nil {
			return "", fmt.Errorf("converting output to json: %w", err)
		}
		return string(b), nil
	}
}

func sanitizeToolInput(input string) string {
	return strings.TrimSpace(input)
}

func (a *Conversation) recordError(ctx context.Context, err error) error {
	return a.recorder.Write(ctx, &journal.Event{
		Timestamp: time.Now(),
		Action:    "error",
		Payload:   err.Error(),
	})
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

// PromptData represents the structure of the data to be filled into the template.
type PromptData struct {
	Query string
	Tools tools.Tools
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
