// Command jef-server serves the /v1/systemone contract from Go.
//
// It exists so a deployment can be one static binary with a small footprint
// instead of a Python process carrying torch. It answers identically to the
// Python server -- the maths, the calibration artifact and the hashing test
// backbone are all held to golden fixtures produced by the Python side -- so
// the two can sit behind one load balancer.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/cyber-security-dev-dep-mitake-com-tw/jef/jef-go/internal/backbone"
	"github.com/cyber-security-dev-dep-mitake-com-tw/jef/jef-go/internal/calib"
	"github.com/cyber-security-dev-dep-mitake-com-tw/jef/jef-go/internal/contract"
	"github.com/cyber-security-dev-dep-mitake-com-tw/jef/jef-go/internal/engine"
	"github.com/cyber-security-dev-dep-mitake-com-tw/jef/jef-go/internal/head"
)

const version = "0.1.0"

type config struct {
	addr            string
	backboneSpec    string
	backboneDir     string
	ortLibrary      string
	headPath        string
	calibrationPath string
	modelName       string
	threads         int
	maxQuestions    int
	maxStateBytes   int
	alpha           float64
}

func loadConfig() config {
	c := config{
		addr:            envStr("JEF_ADDR", ":8080"),
		backboneSpec:    envStr("JEF_BACKBONE", "hashing"),
		backboneDir:     envStr("JEF_BACKBONE_DIR", ""),
		ortLibrary:      envStr("JEF_ORT_LIBRARY", ""),
		headPath:        envStr("JEF_HEAD_PATH", ""),
		calibrationPath: envStr("JEF_CALIBRATION_PATH", ""),
		modelName:       envStr("JEF_MODEL_NAME", ""),
		threads:         envInt("JEF_THREADS", 0),
		maxQuestions:    envInt("JEF_MAX_QUESTIONS", 256),
		maxStateBytes:   envInt("JEF_MAX_STATE_BYTES", 4<<20),
		alpha:           envFloat("JEF_ALPHA", 0.10),
	}
	flag.StringVar(&c.addr, "addr", c.addr, "listen address")
	flag.StringVar(&c.backboneSpec, "backbone", c.backboneSpec, "'hashing' or 'onnx'")
	flag.StringVar(&c.backboneDir, "backbone-dir", c.backboneDir, "exported backbone directory")
	flag.StringVar(&c.headPath, "head", c.headPath, "head.npz")
	flag.StringVar(&c.calibrationPath, "calibration", c.calibrationPath, "calibration.json")
	flag.IntVar(&c.threads, "threads", c.threads, "intra-op threads")
	flag.Parse()
	return c
}

// metrics are the counters that make the shared-state guarantee observable.
//
// stateEncodes/requests must stay at 1.0 no matter how many questions a request
// carries. A rising ratio means shared-state encoding has regressed into
// per-question encoding, which is the one failure that would leave the project
// without a reason to exist.
type metrics struct {
	requests     atomic.Int64
	errors       atomic.Int64
	questions    atomic.Int64
	stateEncodes atomic.Int64
}

type server struct {
	cfg     config
	engine  *engine.Engine
	metrics metrics
	log     *slog.Logger
}

func main() {
	logger := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelInfo}))
	cfg := loadConfig()

	eng, err := buildEngine(cfg, logger)
	if err != nil {
		logger.Error("startup failed", "error", err)
		os.Exit(1)
	}
	defer eng.Backbone.Close()

	srv := &server{cfg: cfg, engine: eng, log: logger}

	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/systemone", srv.handleSystemOne)
	mux.HandleFunc("GET /v1/models", srv.handleModels)
	mux.HandleFunc("GET /v1/limits", srv.handleLimits)
	mux.HandleFunc("GET /healthz", srv.handleHealth)
	mux.HandleFunc("GET /metrics", srv.handleMetrics)

	httpServer := &http.Server{
		Addr:              cfg.addr,
		Handler:           mux,
		ReadHeaderTimeout: 10 * time.Second,
		// Generous: a cold CPU encode of a long state is not fast, and cutting
		// it off would look like a model failure rather than a timeout.
		ReadTimeout:  120 * time.Second,
		WriteTimeout: 120 * time.Second,
		IdleTimeout:  120 * time.Second,
	}

	go func() {
		logger.Info("listening",
			"addr", cfg.addr,
			"model", eng.ModelName,
			"calibrated", eng.Calibrator.IsFitted(),
		)
		if err := httpServer.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			logger.Error("serve failed", "error", err)
			os.Exit(1)
		}
	}()

	stop := make(chan os.Signal, 1)
	signal.Notify(stop, os.Interrupt, syscall.SIGTERM)
	<-stop

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	if err := httpServer.Shutdown(ctx); err != nil {
		logger.Error("shutdown", "error", err)
	}
}

func buildEngine(cfg config, logger *slog.Logger) (*engine.Engine, error) {
	var bb backbone.Backbone
	var err error

	switch cfg.backboneSpec {
	case "hashing", "":
		logger.Warn("JEF_BACKBONE=hashing: deterministic but semantically meaningless vectors; set 'onnx' with JEF_BACKBONE_DIR for anything real")
		bb = backbone.NewHashing(0)
	case "onnx":
		if cfg.backboneDir == "" {
			return nil, errors.New("backbone 'onnx' needs JEF_BACKBONE_DIR (see: python -m jef_train.export_onnx)")
		}
		bb, err = backbone.NewONNX(backbone.ONNXOptions{
			Dir:         cfg.backboneDir,
			LibraryPath: cfg.ortLibrary,
			Threads:     cfg.threads,
		})
		if err != nil {
			return nil, err
		}
	default:
		return nil, fmt.Errorf("unknown backbone %q (expected 'hashing' or 'onnx')", cfg.backboneSpec)
	}

	var h engine.Head = head.ZeroShot{}
	headName := "zeroshot"
	if cfg.headPath != "" {
		bilinear, err := head.LoadBilinear(cfg.headPath)
		if err != nil {
			return nil, err
		}
		if bilinear.Dim() != bb.Dim() {
			return nil, fmt.Errorf(
				"head expects dim %d but backbone %q has dim %d -- these artifacts do not belong together",
				bilinear.Dim(), bb.Name(), bb.Dim())
		}
		h, headName = bilinear, bilinear.Name
	}

	calibrator := calib.New()
	if cfg.calibrationPath != "" {
		if calibrator, err = calib.Load(cfg.calibrationPath); err != nil {
			return nil, err
		}
	}
	if !calibrator.IsFitted() {
		logger.Warn("no calibration loaded: probabilities are uncalibrated, P(correct) is unavailable, and scene gates that depend on it cannot fire")
	}

	return engine.New(bb, h, calibrator, cfg.modelName, headName), nil
}

// -- handlers --------------------------------------------------------------- //

func (s *server) handleSystemOne(w http.ResponseWriter, r *http.Request) {
	body := http.MaxBytesReader(w, r.Body, int64(s.cfg.maxStateBytes))
	var req contract.Request
	if err := json.NewDecoder(body).Decode(&req); err != nil {
		s.fail(w, http.StatusUnprocessableEntity, "invalid_request", err.Error())
		return
	}
	if len(req.Questions) == 0 {
		s.fail(w, http.StatusUnprocessableEntity, "invalid_request", "at least one question is required")
		return
	}
	if len(req.Questions) > s.cfg.maxQuestions {
		s.fail(w, http.StatusRequestEntityTooLarge, "too_many_questions",
			fmt.Sprintf("%d questions exceeds the limit of %d", len(req.Questions), s.cfg.maxQuestions))
		return
	}

	stateText, err := renderState(req.State)
	if err != nil {
		s.fail(w, http.StatusUnprocessableEntity, "invalid_state", err.Error())
		return
	}

	before := s.engine.StateEncodes()
	resp, err := s.engine.Evaluate(stateText, req.Questions, s.cfg.alpha)
	if err != nil {
		// Contract violations and runtime failures are both 422 here: the
		// former is by far the common case, and the message names the question.
		s.fail(w, http.StatusUnprocessableEntity, "invalid_question", err.Error())
		return
	}

	s.metrics.stateEncodes.Add(s.engine.StateEncodes() - before)
	s.metrics.requests.Add(1)
	s.metrics.questions.Add(int64(len(req.Questions)))
	s.writeJSON(w, http.StatusOK, resp)
}

func (s *server) handleModels(w http.ResponseWriter, _ *http.Request) {
	s.writeJSON(w, http.StatusOK, map[string]any{
		"data": []map[string]any{{
			"id":             s.engine.ModelName,
			"object":         "model",
			"backbone":       s.engine.Backbone.Name(),
			"head":           s.engine.HeadName,
			"calibrated":     s.engine.Calibrator.IsFitted(),
			"question_types": []string{"choice", "score", "noul", "boolean"},
			"runtime":        "go",
		}},
	})
}

func (s *server) handleLimits(w http.ResponseWriter, _ *http.Request) {
	s.writeJSON(w, http.StatusOK, map[string]any{
		"max_questions_per_request": s.cfg.maxQuestions,
		"max_state_bytes":           s.cfg.maxStateBytes,
	})
}

func (s *server) handleHealth(w http.ResponseWriter, _ *http.Request) {
	s.writeJSON(w, http.StatusOK, map[string]any{
		"status":     "ok",
		"model":      s.engine.ModelName,
		"calibrated": s.engine.Calibrator.IsFitted(),
		// Surfaced so a deployment cannot quietly serve meaningless vectors.
		"test_backbone": s.engine.Backbone.Name() == "hashing",
		"runtime":       "go",
		"version":       version,
	})
}

func (s *server) handleMetrics(w http.ResponseWriter, _ *http.Request) {
	var b strings.Builder
	writeCounter(&b, "jef_requests_total", "Evaluation requests", s.metrics.requests.Load())
	writeCounter(&b, "jef_errors_total", "Rejected requests", s.metrics.errors.Load())
	writeCounter(&b, "jef_questions_total", "Questions evaluated", s.metrics.questions.Load())
	writeCounter(&b, "jef_state_encodes_total",
		"Full state encodes. Divided by jef_requests_total this must stay at 1.0; above it, shared-state encoding has regressed.",
		s.metrics.stateEncodes.Load())
	w.Header().Set("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
	_, _ = w.Write([]byte(b.String()))
}

func writeCounter(b *strings.Builder, name, help string, value int64) {
	fmt.Fprintf(b, "# HELP %s %s\n# TYPE %s counter\n%s %d\n", name, help, name, name, value)
}

// -- helpers ---------------------------------------------------------------- //

// renderState flattens an accepted state shape to the text the backbone sees.
//
// A JSON array is one shared state, not a batch: the same distinction TypeSafe
// draws, and rendering it as a single document preserves it. Indentation and
// key order match Python's json.dumps(indent=2) so both servers read the same
// characters for the same request.
func renderState(raw json.RawMessage) (string, error) {
	if len(raw) == 0 {
		return "", errors.New("state is required")
	}
	var asString string
	if err := json.Unmarshal(raw, &asString); err == nil {
		return asString, nil
	}

	var probe any
	if err := json.Unmarshal(raw, &probe); err != nil {
		return "", fmt.Errorf("state is not valid JSON: %w", err)
	}
	switch probe.(type) {
	case map[string]any, []any:
		var buf strings.Builder
		enc := json.NewEncoder(&buf)
		enc.SetIndent("", "  ")
		enc.SetEscapeHTML(false)
		if err := enc.Encode(probe); err != nil {
			return "", fmt.Errorf("rendering state: %w", err)
		}
		return strings.TrimRight(buf.String(), "\n"), nil
	default:
		return "", fmt.Errorf("state must be a string, object or array, got %T", probe)
	}
}

func (s *server) writeJSON(w http.ResponseWriter, status int, payload any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	enc := json.NewEncoder(w)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(payload); err != nil {
		s.log.Error("writing response", "error", err)
	}
}

func (s *server) fail(w http.ResponseWriter, status int, code, message string) {
	s.metrics.errors.Add(1)
	s.writeJSON(w, status, map[string]any{
		"error": map[string]string{"code": code, "message": message},
	})
}

func envStr(key, fallback string) string {
	if v, ok := os.LookupEnv(key); ok && v != "" {
		return v
	}
	return fallback
}

func envInt(key string, fallback int) int {
	if v, ok := os.LookupEnv(key); ok && v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return fallback
}

func envFloat(key string, fallback float64) float64 {
	if v, ok := os.LookupEnv(key); ok && v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			return f
		}
	}
	return fallback
}
