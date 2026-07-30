package llm

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math/rand"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/ml"
	"golang.org/x/sync/semaphore"
)

type oilServerRunner struct {
	port      int
	cmd       *exec.Cmd
	done      chan struct{}
	doneErr   error
	client    *http.Client
	modelPath string
	options   api.Options
	status    *StatusWriter
	sem       *semaphore.Weighted

	memoryMu sync.RWMutex
	memTotal uint64
	memGPU   uint64

	loadStart    time.Time
	loadTracking atomic.Bool
	loadActivity atomic.Int64

	gpus []ml.DeviceInfo
}

func (s *oilServerRunner) ModelPath() string {
	return s.modelPath
}

func (s *oilServerRunner) Pid() int {
	if s.cmd != nil && s.cmd.Process != nil {
		return s.cmd.Process.Pid
	}
	return 0
}

func (s *oilServerRunner) GetPort() int {
	return s.port
}

func (s *oilServerRunner) HasExited() bool {
	return s.cmd != nil && s.cmd.ProcessState != nil && s.cmd.ProcessState.ExitCode() >= 0
}

func (s *oilServerRunner) ContextLength() int {
	if s.options.NumCtx > 0 {
		return s.options.NumCtx
	}
	return 2048
}

func (s *oilServerRunner) httpClient() *http.Client {
	if s.client != nil {
		return s.client
	}
	return defaultLlamaServerHTTPClient
}

func (s *oilServerRunner) Ping(ctx context.Context) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, fmt.Sprintf("http://127.0.0.1:%d/health", s.port), nil)
	if err != nil {
		return err
	}
	resp, err := s.httpClient().Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	return nil
}

func (s *oilServerRunner) WaitUntilRunning(ctx context.Context) error {
	start := time.Now()
	timeout := envconfig.LoadTimeout()
	for {
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, fmt.Sprintf("http://127.0.0.1:%d/health", s.port), nil)
		if err != nil {
			return err
		}
		resp, err := s.httpClient().Do(req)
		if err == nil {
			resp.Body.Close()
			if resp.StatusCode == 200 {
				slog.Info("oil-server ready", "port", s.port, "took", time.Since(start))
				return nil
			}
		}
		if time.Since(start) > timeout {
			return fmt.Errorf("oil-server did not start within %v", timeout)
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(200 * time.Millisecond):
		}
	}
}

func (s *oilServerRunner) Load(ctx context.Context, systemInfo ml.SystemInfo, gpus []ml.DeviceInfo, _ bool) ([]ml.DeviceID, error) {
	slog.Info("loading model via oil-server", "model", s.modelPath)

	if err := s.WaitUntilRunning(ctx); err != nil {
		return nil, err
	}

	s.memoryMu.Lock()
	s.memTotal = 0
	s.memGPU = 0
	s.memoryMu.Unlock()

	deviceIDs := make([]ml.DeviceID, len(gpus))
	for i, g := range gpus {
		deviceIDs[i] = g.DeviceID
	}
	return deviceIDs, nil
}

func (s *oilServerRunner) Completion(ctx context.Context, req CompletionRequest, fn func(CompletionResponse)) error {
	slog.Debug("oil-server completion request", "prompt_len", len(req.Prompt))

	if req.Options == nil {
		opts := api.DefaultOptions()
		req.Options = &opts
	}

	if err := s.sem.Acquire(ctx, 1); err != nil {
		return err
	}
	defer s.sem.Release(1)

	if err := s.Ping(ctx); err != nil {
		return fmt.Errorf("oil-server not ready: %w", err)
	}

	body := map[string]any{
		"prompt":      req.Prompt,
		"max_tokens":  boundedNumPredict(req.Options.NumPredict, s.options.NumCtx),
		"temperature": req.Options.Temperature,
		"top_k":       req.Options.TopK,
		"top_p":       req.Options.TopP,
		"stream":      true,
	}
	if req.Options.RepeatPenalty > 0 {
		body["repeat_penalty"] = req.Options.RepeatPenalty
	}

	buffer := &bytes.Buffer{}
	enc := json.NewEncoder(buffer)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(body); err != nil {
		return fmt.Errorf("failed to marshal oil-server request: %v", err)
	}

	endpoint := fmt.Sprintf("http://127.0.0.1:%d/api/generate", s.port)
	serverReq, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, buffer)
	if err != nil {
		return fmt.Errorf("error creating oil-server request: %v", err)
	}
	serverReq.Header.Set("Content-Type", "application/json")

	res, err := s.httpClient().Do(serverReq)
	if err != nil {
		return fmt.Errorf("oil-server completion error: %w", err)
	}
	defer res.Body.Close()

	if res.StatusCode >= 400 {
		bodyBytes, _ := io.ReadAll(res.Body)
		return fmt.Errorf("oil-server error (status %d): %s", res.StatusCode, string(bodyBytes))
	}

	scanner := bufio.NewScanner(res.Body)
	buf := make([]byte, 0, 64*1024)
	scanner.Buffer(buf, 8*format.MegaByte)

	var content strings.Builder
	var finalResp CompletionResponse
	var hasFinalResp bool
	promptEvalCount := 0
	evalCount := 0

	for scanner.Scan() {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
			line := scanner.Bytes()
			if len(line) == 0 {
				continue
			}
			evt, ok := bytes.CutPrefix(line, []byte("data: "))
			if !ok {
				continue
			}

			var data map[string]any
			if err := json.Unmarshal(evt, &data); err != nil {
				continue
			}

			if token, ok := data["token"].(string); ok && token != "" {
				content.WriteString(token)
				fn(CompletionResponse{Content: token})
			}

			if done, _ := data["done"].(bool); done {
				if count, _ := data["count"].(float64); count > 0 {
					evalCount = int(count)
				}
				if promptCount, _ := data["prompt_tokens"].(float64); promptCount > 0 {
					promptEvalCount = int(promptCount)
				}
				hasFinalResp = true
				break
			}
		}
		if hasFinalResp {
			break
		}
	}

	if hasFinalResp {
		for scanner.Scan() {
		}
		finalResp = CompletionResponse{
			Content:         content.String(),
			Done:            true,
			DoneReason:      DoneReasonStop,
			PromptEvalCount: promptEvalCount,
			EvalCount:       evalCount,
		}
		fn(finalResp)
		return nil
	}

	return scanner.Err()
}

func (s *oilServerRunner) Chat(ctx context.Context, req ChatRequest, fn func(ChatResponse)) error {
	slog.Debug("oil-server chat request", "messages", len(req.Messages))

	if len(req.Messages) == 0 {
		return errors.New("no messages in chat request")
	}

	lastMsg := req.Messages[len(req.Messages)-1]
	prompt := lastMsg.Content

	completionReq := CompletionRequest{
		Prompt:  prompt,
		Options: req.Options,
		Format:  req.Format,
	}

	return s.Completion(ctx, completionReq, func(cr CompletionResponse) {
		fn(ChatResponse{
			Message: api.Message{
				Role:    "assistant",
				Content: cr.Content,
			},
			Done:               cr.Done,
			DoneReason:         cr.DoneReason,
			PromptEvalCount:    cr.PromptEvalCount,
			PromptEvalDuration: cr.PromptEvalDuration,
			EvalCount:          cr.EvalCount,
			EvalDuration:       cr.EvalDuration,
		})
	})
}

func (s *oilServerRunner) ApplyChatTemplate(ctx context.Context, req ChatRequest) (string, error) {
	if len(req.Messages) == 0 {
		return "", nil
	}
	lastMsg := req.Messages[len(req.Messages)-1]
	return lastMsg.Content, nil
}

func (s *oilServerRunner) Embedding(ctx context.Context, input string) ([]float32, int, error) {
	return nil, 0, errors.New("OIL models do not support embedding endpoints; use /api/generate instead")
}

func (s *oilServerRunner) Tokenize(ctx context.Context, content string) ([]int, error) {
	body := map[string]string{"content": content}
	buffer := &bytes.Buffer{}
	if err := json.NewEncoder(buffer).Encode(body); err != nil {
		return nil, err
	}

	endpoint := fmt.Sprintf("http://127.0.0.1:%d/api/tokenize", s.port)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, buffer)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := s.httpClient().Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var result struct {
		Tokens []int `json:"tokens"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}
	return result.Tokens, nil
}

func (s *oilServerRunner) Detokenize(ctx context.Context, tokens []int) (string, error) {
	body := map[string][]int{"tokens": tokens}
	buffer := &bytes.Buffer{}
	if err := json.NewEncoder(buffer).Encode(body); err != nil {
		return "", err
	}

	endpoint := fmt.Sprintf("http://127.0.0.1:%d/api/detokenize", s.port)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, buffer)
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := s.httpClient().Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	var result struct {
		Content string `json:"content"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", err
	}
	return result.Content, nil
}

func (s *oilServerRunner) Close() error {
	slog.Info("stopping oil-server", "model", s.modelPath)
	if s.cmd != nil && s.cmd.Process != nil {
		if err := s.cmd.Process.Kill(); err != nil {
			return err
		}
	}
	if s.done != nil {
		<-s.done
	}
	return nil
}

func (s *oilServerRunner) MemorySize() (total, vram uint64) {
	s.memoryMu.RLock()
	defer s.memoryMu.RUnlock()
	return s.memTotal, s.memGPU
}

func (s *oilServerRunner) VRAMByGPU(id ml.DeviceID) uint64 {
	return 0
}

func (s *oilServerRunner) GetDeviceInfos(ctx context.Context) []ml.DeviceInfo {
	return s.gpus
}

// NewOilServer creates a new oil-server runner for the given OIL model.
func NewOilServer(gpus []ml.DeviceInfo, modelPath string, opts api.Options, numParallel int) (LlamaServer, error) {
	exe, err := findOilServer()
	if err != nil {
		return nil, err
	}

	port := 0
	if a, err := net.ResolveTCPAddr("tcp", "localhost:0"); err == nil {
		if l, err := net.ListenTCP("tcp", a); err == nil {
			port = l.Addr().(*net.TCPAddr).Port
			l.Close()
		}
	}
	if port == 0 {
		port = rand.Intn(65535-49152) + 49152
	}

	params := []string{
		"--model", modelPath,
		"--port", strconv.Itoa(port),
		"--max-tokens", strconv.Itoa(opts.NumCtx),
	}
	if opts.NumThread > 0 {
		params = append(params, "--workers", strconv.Itoa(opts.NumThread))
	}
	if opts.Temperature > 0 {
		params = append(params, "--temperature", strconv.FormatFloat(float64(opts.Temperature), 'f', 2, 32))
	}
	if opts.TopK > 0 {
		params = append(params, "--top-k", strconv.Itoa(opts.TopK))
	}
	if opts.TopP > 0 {
		params = append(params, "--top-p", strconv.FormatFloat(float64(opts.TopP), 'f', 2, 32))
	}

	status := NewStatusWriter(os.Stderr)

	cmd := exec.Command(exe, params...)
	cmd.Stdout = status
	cmd.Stderr = status
	cmd.SysProcAttr = LlamaServerSysProcAttr
	cmd.Env = os.Environ()

	slog.Info("starting oil-server", "cmd", cmd)

	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("error starting oil-server: %w", err)
	}

	s := &oilServerRunner{
		client:    newLlamaServerHTTPClient(),
		status:    status,
		options:   opts,
		modelPath: modelPath,
		gpus:      gpus,
		sem:       semaphore.NewWeighted(int64(numParallel)),
	}
	s.cmd = cmd
	s.port = port
	s.done = make(chan struct{})
	s.loadStart = time.Now()

	go func(cmd *exec.Cmd, done chan struct{}) {
		err := cmd.Wait()
		s.doneErr = err
		if err != nil && s.status.LastError() != "" {
			s.doneErr = errors.New(s.status.LastError())
		}
		close(done)
	}(s.cmd, s.done)

	return s, nil
}

// findOilServer locates the oil-server executable.
func findOilServer() (string, error) {
	name := "oil_server"
	if runtime.GOOS == "windows" {
		name += ".exe"
	}

	candidates := []string{name}
	exe, err := os.Executable()
	if err == nil {
		dir := filepath.Dir(exe)
		candidates = append(candidates, filepath.Join(dir, name))
		// Check ../lib/ollama/ for packaged layouts
		candidates = append(candidates, filepath.Join(dir, "..", "lib", "ollama", name))
		// Check dist/ directory for local builds
		candidates = append(candidates, filepath.Join(dir, "dist", runtime.GOOS+"-"+runtime.GOARCH, name))
		if runtime.GOOS == "windows" {
			candidates = append(candidates, filepath.Join(dir, "dist", runtime.GOOS+"_"+runtime.GOARCH, name))
		}
	}
	// Check common build directories
	wd, _ := os.Getwd()
	if wd != "" {
		candidates = append(candidates, filepath.Join(wd, "build", "tools", name))
		candidates = append(candidates, filepath.Join(wd, "build", "oil_server", name))
	}

	for _, path := range candidates {
		if _, err := os.Stat(path); err == nil {
			return path, nil
		}
	}

	return "", fmt.Errorf("oil_server binary not found (checked: %s)", strings.Join(candidates, ", "))
}

// IsOilFile returns true if the given path has a .oil extension.
func IsOilFile(path string) bool {
	return strings.EqualFold(filepath.Ext(path), ".oil")
}
