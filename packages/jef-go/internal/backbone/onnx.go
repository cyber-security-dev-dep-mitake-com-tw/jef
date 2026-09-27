package backbone

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"

	"github.com/sugarme/tokenizer"
	"github.com/sugarme/tokenizer/pretrained"
	ort "github.com/yalue/onnxruntime_go"
)

// Meta describes an exported backbone directory, written by
// `python -m jef_train.export_onnx`.
type Meta struct {
	ModelID               string   `json:"model_id"`
	Opset                 int      `json:"opset"`
	HiddenSize            int      `json:"hidden_size"`
	MaxPositionEmbeddings int      `json:"max_position_embeddings"`
	TokenizerFiles        []string `json:"tokenizer_files"`
	Note                  string   `json:"note"`
}

// ONNX runs the exported encoder through ONNX Runtime.
//
// The tokenizer comes from the exported directory, not from a reimplementation.
// A tokenizer mismatch is the quietest possible failure: every request
// succeeds, every answer is subtly wrong, and nothing in the logs says so.
type ONNX struct {
	meta      Meta
	tokenizer *tokenizer.Tokenizer
	session   *ort.DynamicAdvancedSession

	maxStateTokens int
	maxQueryTokens int

	// ONNX Runtime sessions are not documented as goroutine-safe for concurrent
	// Run, and the server serves concurrently.
	mu sync.Mutex
}

var (
	ortOnce sync.Once
	ortErr  error
)

// InitONNXRuntime loads the shared library once per process.
//
// The path is explicit because there is no portable default: the Docker image
// installs it to a known location, while a developer machine usually has it
// inside the Python onnxruntime wheel.
func InitONNXRuntime(libraryPath string) error {
	ortOnce.Do(func() {
		if libraryPath != "" {
			ort.SetSharedLibraryPath(libraryPath)
		}
		ortErr = ort.InitializeEnvironment()
	})
	return ortErr
}

// ONNXOptions configures the ONNX backbone.
type ONNXOptions struct {
	// Dir holds backbone.onnx, backbone.json and the tokenizer files.
	Dir string
	// LibraryPath is libonnxruntime.{so,dylib}. Empty uses the system default.
	LibraryPath string
	// Threads is the intra-op thread count. The deployment target is a CPU-only
	// box, so this is the main throughput lever; leave headroom below the CPU
	// limit rather than matching it.
	Threads int
	// MaxStateTokens defaults to the model's position limit.
	MaxStateTokens int
	// MaxQueryTokens bounds the short question/option encodes.
	MaxQueryTokens int
}

// NewONNX loads an exported backbone directory.
func NewONNX(opts ONNXOptions) (*ONNX, error) {
	if err := InitONNXRuntime(opts.LibraryPath); err != nil {
		return nil, fmt.Errorf("jef: initialising ONNX Runtime: %w", err)
	}

	metaRaw, err := os.ReadFile(filepath.Join(opts.Dir, "backbone.json"))
	if err != nil {
		return nil, fmt.Errorf("jef: reading backbone metadata: %w", err)
	}
	var meta Meta
	if err := json.Unmarshal(metaRaw, &meta); err != nil {
		return nil, fmt.Errorf("jef: parsing backbone metadata: %w", err)
	}
	if meta.HiddenSize <= 0 {
		return nil, fmt.Errorf("jef: backbone metadata has no hidden_size")
	}

	tokenizerPath := filepath.Join(opts.Dir, "tokenizer.json")
	tk, err := pretrained.FromFile(tokenizerPath)
	if err != nil {
		return nil, fmt.Errorf("jef: loading tokenizer %s: %w", tokenizerPath, err)
	}

	sessionOpts, err := ort.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("jef: creating session options: %w", err)
	}
	defer sessionOpts.Destroy()
	if opts.Threads > 0 {
		if err := sessionOpts.SetIntraOpNumThreads(opts.Threads); err != nil {
			return nil, fmt.Errorf("jef: setting intra-op threads: %w", err)
		}
	}

	modelPath := filepath.Join(opts.Dir, "backbone.onnx")
	session, err := ort.NewDynamicAdvancedSession(
		modelPath,
		[]string{"input_ids", "attention_mask"},
		[]string{"last_hidden_state"},
		sessionOpts,
	)
	if err != nil {
		return nil, fmt.Errorf("jef: loading %s: %w", modelPath, err)
	}

	maxState := opts.MaxStateTokens
	if maxState <= 0 {
		maxState = meta.MaxPositionEmbeddings
	}
	if maxState <= 0 {
		maxState = 8192
	}
	maxQuery := opts.MaxQueryTokens
	if maxQuery <= 0 {
		maxQuery = 256
	}

	return &ONNX{
		meta:           meta,
		tokenizer:      tk,
		session:        session,
		maxStateTokens: maxState,
		maxQueryTokens: maxQuery,
	}, nil
}

// Name reports the exported model id.
func (o *ONNX) Name() string { return o.meta.ModelID }

// Dim is the encoder hidden size.
func (o *ONNX) Dim() int { return o.meta.HiddenSize }

// Close releases the session.
func (o *ONNX) Close() error {
	if o.session != nil {
		o.session.Destroy()
		o.session = nil
	}
	return nil
}

// CountTokens reports the tokenised length, for usage accounting only.
func (o *ONNX) CountTokens(text string) int {
	ids, err := o.encodeIDs(text, o.maxStateTokens)
	if err != nil {
		return 0
	}
	return len(ids)
}

func (o *ONNX) encodeIDs(text string, limit int) ([]int64, error) {
	enc, err := o.tokenizer.EncodeSingle(text, true)
	if err != nil {
		return nil, fmt.Errorf("jef: tokenising: %w", err)
	}
	ids := enc.Ids
	if len(ids) > limit {
		ids = ids[:limit]
	}
	out := make([]int64, len(ids))
	for i, id := range ids {
		out[i] = int64(id)
	}
	if len(out) == 0 {
		// An empty sequence is not a valid model input; a single pad token
		// keeps the shape legal and the answer meaningless-but-finite.
		out = []int64{0}
	}
	return out, nil
}

// EncodeState runs the encoder once and keeps token-level hidden states.
//
// This is the expensive call, and the whole design exists so it happens once
// per request rather than once per question.
func (o *ONNX) EncodeState(text string) (Encoding, error) {
	ids, err := o.encodeIDs(text, o.maxStateTokens)
	if err != nil {
		return Encoding{}, err
	}
	hidden, err := o.forward([][]int64{ids})
	if err != nil {
		return Encoding{}, err
	}
	mask := make([]float64, len(ids))
	for i := range mask {
		mask[i] = 1
	}
	return Encoding{Hidden: hidden[0], Mask: mask, NTokens: len(ids)}, nil
}

// EncodeQueries batches the short question/option texts into one pass.
func (o *ONNX) EncodeQueries(texts []string) ([][]float64, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	batch := make([][]int64, len(texts))
	for i, text := range texts {
		ids, err := o.encodeIDs(text, o.maxQueryTokens)
		if err != nil {
			return nil, err
		}
		batch[i] = ids
	}

	hidden, err := o.forward(batch)
	if err != nil {
		return nil, err
	}

	out := make([][]float64, len(texts))
	for i := range texts {
		// Mean-pool over real tokens only, then L2 -- matching
		// jef_core.backends.mmbert.encode_queries.
		pooled := make([]float64, o.Dim())
		real := len(batch[i])
		for t := 0; t < real; t++ {
			for d := range pooled {
				pooled[d] += hidden[i][t][d]
			}
		}
		for d := range pooled {
			pooled[d] /= float64(real)
		}
		out[i] = l2(pooled)
	}
	return out, nil
}

// forward runs one padded batch and returns (batch, sequence, dim).
func (o *ONNX) forward(batch [][]int64) ([][][]float64, error) {
	maxLen := 0
	for _, ids := range batch {
		if len(ids) > maxLen {
			maxLen = len(ids)
		}
	}

	flatIDs := make([]int64, len(batch)*maxLen)
	flatMask := make([]int64, len(batch)*maxLen)
	for b, ids := range batch {
		copy(flatIDs[b*maxLen:], ids)
		for t := range ids {
			flatMask[b*maxLen+t] = 1
		}
	}

	shape := ort.NewShape(int64(len(batch)), int64(maxLen))
	idTensor, err := ort.NewTensor(shape, flatIDs)
	if err != nil {
		return nil, fmt.Errorf("jef: building input_ids tensor: %w", err)
	}
	defer idTensor.Destroy()

	maskTensor, err := ort.NewTensor(shape, flatMask)
	if err != nil {
		return nil, fmt.Errorf("jef: building attention_mask tensor: %w", err)
	}
	defer maskTensor.Destroy()

	outShape := ort.NewShape(int64(len(batch)), int64(maxLen), int64(o.Dim()))
	outTensor, err := ort.NewEmptyTensor[float32](outShape)
	if err != nil {
		return nil, fmt.Errorf("jef: allocating output tensor: %w", err)
	}
	defer outTensor.Destroy()

	o.mu.Lock()
	err = o.session.Run([]ort.Value{idTensor, maskTensor}, []ort.Value{outTensor})
	o.mu.Unlock()
	if err != nil {
		return nil, fmt.Errorf("jef: running the encoder: %w", err)
	}

	raw := outTensor.GetData()
	out := make([][][]float64, len(batch))
	dim := o.Dim()
	for b := range batch {
		// Only the real tokens are kept: padding must never reach pooling.
		real := len(batch[b])
		rows := make([][]float64, real)
		for t := 0; t < real; t++ {
			row := make([]float64, dim)
			base := (b*maxLen + t) * dim
			for d := 0; d < dim; d++ {
				row[d] = float64(raw[base+d])
			}
			rows[t] = row
		}
		out[b] = rows
	}
	return out, nil
}
