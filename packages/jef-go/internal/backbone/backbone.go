// Package backbone runs the frozen encoder.
//
// Two implementations: an ONNX Runtime one for production, and a deterministic
// hashing stub that lets the whole server -- contract, head, calibration,
// routing -- be exercised without a 300MB model. The stub is the Go twin of
// jef_core.backends.hashing and carries the same warning: its vectors are
// stable, not meaningful, and no number produced with it should be published.
package backbone

import (
	"crypto/sha256"
	"encoding/binary"
	"math"
	"strconv"
)

// Encoding is a state read once and reused by every question in the request.
type Encoding struct {
	// Hidden is (tokens, dim) contextual vectors.
	Hidden [][]float64
	// Mask is 1 for real tokens, 0 for padding.
	Mask []float64
	// NTokens counts real tokens, for usage reporting.
	NTokens int
}

// Backbone is a frozen text encoder.
//
// EncodeState keeps token-level hidden states because every question attends
// over them independently; EncodeQueries only needs pooled vectors.
type Backbone interface {
	Name() string
	Dim() int
	EncodeState(text string) (Encoding, error)
	EncodeQueries(texts []string) ([][]float64, error)
	CountTokens(text string) int
	Close() error
}

const (
	// hashingTokenChars chunks by character rather than whitespace. Splitting
	// Chinese on spaces collapses a whole alert into one token, and zh-TW is a
	// first-class language here.
	hashingTokenChars = 4
	hashingMaxTokens  = 2048
)

// Hashing is the deterministic, dependency-free stub.
type Hashing struct{ dim int }

// NewHashing returns the stub backbone. dim <= 0 uses the default 256.
func NewHashing(dim int) *Hashing {
	if dim <= 0 {
		dim = 256
	}
	return &Hashing{dim: dim}
}

// Name identifies the backbone in /healthz and in model ids.
func (h *Hashing) Name() string { return "hashing" }

// Dim is the vector width.
func (h *Hashing) Dim() int { return h.dim }

// Close is a no-op; the stub holds no resources.
func (h *Hashing) Close() error { return nil }

// EncodeState hashes the state into stable per-token vectors.
func (h *Hashing) EncodeState(text string) (Encoding, error) {
	tokens := pseudoTokens(text)
	hidden := make([][]float64, len(tokens))
	mask := make([]float64, len(tokens))
	for i, tok := range tokens {
		hidden[i] = hashVector(tok, h.dim)
		mask[i] = 1
	}
	return Encoding{Hidden: hidden, Mask: mask, NTokens: len(tokens)}, nil
}

// EncodeQueries pools each short query text into one vector.
func (h *Hashing) EncodeQueries(texts []string) ([][]float64, error) {
	out := make([][]float64, len(texts))
	for i, text := range texts {
		tokens := pseudoTokens(text)
		sum := make([]float64, h.dim)
		for _, tok := range tokens {
			v := hashVector(tok, h.dim)
			for d := range sum {
				sum[d] += v[d]
			}
		}
		for d := range sum {
			sum[d] = float64(float32(sum[d] / float64(len(tokens))))
		}
		out[i] = l2(sum)
	}
	return out, nil
}

// CountTokens reports pseudo-token count for usage accounting.
func (h *Hashing) CountTokens(text string) int { return len(pseudoTokens(text)) }

func pseudoTokens(text string) []string {
	runes := []rune(text)
	if len(runes) == 0 {
		return []string{""}
	}
	out := make([]string, 0, len(runes)/hashingTokenChars+1)
	for i := 0; i < len(runes) && len(out) < hashingMaxTokens; i += hashingTokenChars {
		end := i + hashingTokenChars
		if end > len(runes) {
			end = len(runes)
		}
		out = append(out, string(runes[i:end]))
	}
	return out
}

// hashVector reproduces jef_core.backends.hashing._hash_vec exactly.
//
// Bit-for-bit, deliberately: with both stubs agreeing, the same acceptance
// suite can be pointed at the Python server or the Go one and expect the same
// answers. That turns "the two implementations are equivalent" from a claim
// into something the conformance tests actually check, end to end, without
// needing a 300MB model in CI.
//
// The seed is the token, a NUL, and the counter as a *decimal string* -- not a
// byte -- because that is what Python's f-string produces, and a single byte
// would silently diverge past counter 9.
func hashVector(token string, dim int) []float64 {
	need := dim * 4
	buf := make([]byte, 0, need+sha256.Size)
	for counter := 0; len(buf) < need; counter++ {
		sum := sha256.Sum256([]byte(token + "\x00" + strconv.Itoa(counter)))
		buf = append(buf, sum[:]...)
	}
	out := make([]float64, dim)
	for i := range out {
		raw := binary.LittleEndian.Uint32(buf[i*4:])
		out[i] = float64(raw)/float64(math.MaxUint32)*2 - 1
	}
	// Python stores these as float32; matching the precision keeps the two
	// stubs comparable at float32 epsilon rather than drifting apart.
	normalized := l2(out)
	for i, v := range normalized {
		normalized[i] = float64(float32(v))
	}
	return normalized
}

func l2(v []float64) []float64 {
	var sum float64
	for _, x := range v {
		sum += x * x
	}
	norm := math.Sqrt(sum)
	if norm == 0 {
		return v
	}
	out := make([]float64, len(v))
	for i, x := range v {
		out[i] = x / norm
	}
	return out
}
