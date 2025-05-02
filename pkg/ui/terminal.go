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

package ui

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"github.com/GoogleCloudPlatform/kubectl-ai/pkg/journal"
	"github.com/charmbracelet/glamour"
	"github.com/chzyer/readline"
	"k8s.io/klog/v2"
)

type TerminalUI struct {
	journal          journal.Recorder
	markdownRenderer *glamour.TermRenderer

	subscription io.Closer

	// currentBlock is the block we are rendering
	currentBlock Block
	// currentBlockText is text of the currentBlock that we have already rendered to the screen
	currentBlockText string

	rl *readline.Instance

	// This is useful in cases where stdin is already been used for providing the input to the agent (caller in this case)
	// in such cases, stdin is already consumed and closed and reading input results in IO error.
	// In such cases, we open /dev/tty and use it for taking input.
	useTTYForInput bool
}

var _ UI = &TerminalUI{}

func NewTerminalUI(doc *Document, journal journal.Recorder, useTTYForInput bool) (*TerminalUI, error) {
	mdRenderer, err := glamour.NewTermRenderer(
		glamour.WithAutoStyle(),
		glamour.WithPreservedNewLines(),
		glamour.WithEmoji(),
	)
	if err != nil {
		return nil, fmt.Errorf("error initializing the markdown renderer: %w", err)
	}
	u := &TerminalUI{markdownRenderer: mdRenderer, journal: journal, useTTYForInput: useTTYForInput}

	subscription := doc.AddSubscription(u)
	u.subscription = subscription

	var reader io.ReadCloser
	var writer io.Writer = os.Stdout // Default writer
	if useTTYForInput {
		// Stdin was used for piped data, open the terminal directly
		tty, err := os.OpenFile("/dev/tty", os.O_RDWR, 0)
		if err != nil {
			return nil, fmt.Errorf("error opening /dev/tty: %w", err)
		}
		// defer tty.Close() // Don't close tty here, readline needs it
		reader = tty
		// writer = tty // Use tty for writer as well
		// writer = os.Stdout
	} else {
		reader = os.Stdin
		// writer remains os.Stdout
	}

	u.rl, err = readline.NewEx(&readline.Config{
		Prompt:      "> ",
		Stdin:       reader,
		Stdout:      writer, // Use tty or os.Stdout
		HistoryFile: filepath.Join(os.TempDir(), "kubectl-ai-history"),
	})
	if err != nil {
		return nil, fmt.Errorf("error initializing readline: %w", err)
	}
	return u, nil
}

func (u *TerminalUI) Close() error {
	var errs []error
	if u.subscription != nil {
		if err := u.subscription.Close(); err != nil {
			errs = append(errs, err)
		} else {
			u.subscription = nil
		}
	}
	err := u.rl.Close()
	if err != nil {
		errs = append(errs, err)
	}
	return errors.Join(errs...)
}

func (u *TerminalUI) DocumentChanged(doc *Document, block Block) {
	blockIndex := doc.IndexOf(block)

	if blockIndex != doc.NumBlocks()-1 {
		klog.Warningf("update to blocks other than the last block is not supported in terminal mode")
		return
	}

	if u.currentBlock != block {
		u.currentBlock = block
		if u.currentBlockText != "" {
			fmt.Printf("\n")
		}
		u.currentBlockText = ""
	}

	text := ""
	streaming := false

	var styleOptions []StyleOption
	switch block := block.(type) {
	case *ErrorBlock:
		styleOptions = append(styleOptions, Foreground(ColorRed))
		text = block.Text()
	case *FunctionCallRequestBlock:
		styleOptions = append(styleOptions, Foreground(ColorGreen))
		text = block.Text()
	case *AgentTextBlock:
		styleOptions = append(styleOptions, RenderMarkdown())
		if block.Color != "" {
			styleOptions = append(styleOptions, Foreground(block.Color))
		}
		text = block.Text()
		streaming = block.Streaming()
	case *InputTextBlock:
		// u.rl.SetPrompt("> ")
		block.Observable().Set(">", nil)
		// fmt.Printf("> ")
		query, err := u.rl.Readline()
		if err != nil {
			block.Observable().Set("", err)
			if err == readline.ErrInterrupt || err == io.EOF {
				return
			} else {
				fmt.Printf("error reading input: %v\n", err)
			}
			return
		}
		block.Observable().Set(query, nil)
		return
		// var reader *bufio.Reader
		// if u.useTTYForInput {
		// 	// Stdin was used for piped data, open the terminal directly
		// 	tty, err := os.OpenFile("/dev/tty", os.O_RDWR, 0)
		// 	if err != nil {
		// 		block.Observable().Set("", err)
		// 		return
		// 	}
		// 	defer tty.Close()
		// 	reader = bufio.NewReader(tty)
		// } else {
		// 	reader = bufio.NewReader(os.Stdin)
		// }
		// query, err := reader.ReadString('\n')
		// if err != nil {
		// 	block.Observable().Set("", err)
		// } else {
		// 	block.Observable().Set(query, nil)
		// }
		// return

	case *InputOptionBlock:
		fmt.Printf("%s\n", block.Prompt)
		// var reader *bufio.Reader
		// if u.useTTYForInput {
		// 	tty, err := os.OpenFile("/dev/tty", os.O_RDWR, 0)
		// 	if err != nil {
		// 		block.Observable().Set("", err)
		// 		return
		// 	}
		// 	defer tty.Close()
		// 	reader = bufio.NewReader(tty)
		// } else {
		// 	reader = bufio.NewReader(os.Stdin)
		// }

		u.rl.SetPrompt("  Enter your choice (number): ")

		for {
			// fmt.Print("  Enter your choice (number): ")
			var response string
			response, err := u.rl.Readline()
			if err != nil {
				block.Observable().Set("", err)
				break
			}

			choice := strings.TrimSpace(response)

			if slices.Contains(block.Options, choice) {
				block.Observable().Set(choice, nil)
				break
			}

			// If not returned, the choice was invalid
			fmt.Printf("  Invalid choice. Please enter one of: %s\n", strings.Join(block.Options, ", "))
			// u.rl.Refresh() // Removed
			continue
		}
		return
	}

	computedStyle := &style{}
	for _, opt := range styleOptions {
		opt(computedStyle)
	}

	if streaming && computedStyle.renderMarkdown {
		// Because we can't render markdown incrementally,
		// we "hold back" the text if we are streaming markdown until streaming is done
		text = ""
	}

	printText := text

	if computedStyle.renderMarkdown && printText != "" {
		out, err := u.markdownRenderer.Render(printText)
		if err != nil {
			klog.Errorf("Error rendering markdown: %v", err)
		} else {
			printText = out
		}
	}

	if u.currentBlockText != "" {
		if strings.HasPrefix(text, u.currentBlockText) {
			printText = strings.TrimPrefix(printText, u.currentBlockText)
		} else {
			klog.Warningf("text did not match text already rendered; text %q; currentBlockText %q", text, u.currentBlockText)
		}
	}
	u.currentBlockText = text

	reset := ""
	switch computedStyle.foreground {
	case ColorRed:
		fmt.Printf("\033[31m")
		reset += "\033[0m"
	case ColorGreen:
		fmt.Printf("\033[32m")
		reset += "\033[0m"
	case ColorWhite:
		fmt.Printf("\033[37m")
		reset += "\033[0m"

	case "":
	default:
		klog.Info("foreground color not supported by TerminalUI", "color", computedStyle.foreground)
	}

	fmt.Printf("%s%s", printText, reset)
}

func (u *TerminalUI) RenderOutput(ctx context.Context, s string, styleOptions ...StyleOption) {
	log := klog.FromContext(ctx)

	u.journal.Write(ctx, &journal.Event{
		Action: journal.ActionUIRender,
		Payload: map[string]any{
			"text": s,
		},
	})

	computedStyle := &style{}
	for _, opt := range styleOptions {
		opt(computedStyle)
	}

	if computedStyle.renderMarkdown {
		out, err := u.markdownRenderer.Render(s)
		if err != nil {
			log.Error(err, "Error rendering markdown")
		}
		s = out
	}

	reset := ""
	switch computedStyle.foreground {
	case ColorRed:
		fmt.Printf("\033[31m")
		reset += "\033[0m"
	case ColorGreen:
		fmt.Printf("\033[32m")
		reset += "\033[0m"
	case ColorWhite:
		fmt.Printf("\033[37m")
		reset += "\033[0m"

	case "":
	default:
		log.Info("foreground color not supported by TerminalUI", "color", computedStyle.foreground)
	}

	fmt.Printf("%s%s", s, reset)
}

func (u *TerminalUI) ClearScreen() {
	readline.ClearScreen(os.Stdout)
}
