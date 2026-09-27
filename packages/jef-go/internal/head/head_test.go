package head_test

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/cyber-security-dev-dep-mitake-com-tw/jef/jef-go/internal/head"
)

// Pooling and the head are float32 in Python and float64 here, so the tolerance
// is float32 epsilon rather than 1e-9. Anything looser would hide a real
// divergence; anything tighter would fail on representation alone.
const tolerance = 1e-5

type goldenFile struct {
	Head *struct {
		Dim       int         `json:"dim"`
		Rank      int         `json:"rank"`
		Hidden    [][]float64 `json:"hidden"`
		Mask      []float64   `json:"mask"`
		HeadScale float64     `json:"head_scale"`
		HeadBias  float64     `json:"head_bias"`
		Cases     []struct {
			NOptions               int         `json:"n_options"`
			Queries                [][]float64 `json:"queries"`
			ExpectedPooled         [][]float64 `json:"expected_pooled"`
			ExpectedBilinearLogits []float64   `json:"expected_bilinear_logits"`
			ExpectedZeroShotLogits []float64   `json:"expected_zeroshot_logits"`
		} `json:"cases"`
	} `json:"head"`
}

func load(t *testing.T) goldenFile {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", "..", "testdata", "golden.json"))
	if err != nil {
		t.Fatalf("golden fixtures missing (regenerate with `python -m jef_train.golden`): %v", err)
	}
	var g goldenFile
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("golden fixtures unreadable: %v", err)
	}
	if g.Head == nil {
		t.Fatal("golden fixtures carry no head cases")
	}
	return g
}

func headPath() string { return filepath.Join("..", "..", "testdata", "head.npz") }

func TestNPZLoadsThePythonHead(t *testing.T) {
	g := load(t)
	h, err := head.LoadBilinear(headPath())
	if err != nil {
		t.Fatalf("loading head.npz: %v", err)
	}
	if h.Dim() != g.Head.Dim {
		t.Errorf("dim: got %d want %d", h.Dim(), g.Head.Dim)
	}
	if h.Rank() != g.Head.Rank {
		t.Errorf("rank: got %d want %d", h.Rank(), g.Head.Rank)
	}
	if math.Abs(h.Scale-g.Head.HeadScale) > tolerance {
		t.Errorf("scale: got %v want %v", h.Scale, g.Head.HeadScale)
	}
	if math.Abs(h.Bias-g.Head.HeadBias) > tolerance {
		t.Errorf("bias: got %v want %v", h.Bias, g.Head.HeadBias)
	}
}

func TestAttentionPoolMatchesPython(t *testing.T) {
	g := load(t)
	for _, c := range g.Head.Cases {
		pooled, err := head.AttentionPool(g.Head.Hidden, g.Head.Mask, c.Queries)
		if err != nil {
			t.Fatalf("n=%d: %v", c.NOptions, err)
		}
		if len(pooled) != len(c.ExpectedPooled) {
			t.Fatalf("n=%d: got %d pooled vectors, want %d", c.NOptions, len(pooled), len(c.ExpectedPooled))
		}
		for i := range pooled {
			for d := range pooled[i] {
				if math.Abs(pooled[i][d]-c.ExpectedPooled[i][d]) > tolerance {
					t.Fatalf("n=%d option %d dim %d: got %.9g want %.9g",
						c.NOptions, i, d, pooled[i][d], c.ExpectedPooled[i][d])
				}
			}
		}
	}
}

func TestMaskedTokensDoNotContribute(t *testing.T) {
	// The fixture masks its last five tokens. Replacing them with garbage must
	// not move the pooled output at all -- if it does, padding is leaking into
	// every answer.
	g := load(t)
	c := g.Head.Cases[0]

	clean, err := head.AttentionPool(g.Head.Hidden, g.Head.Mask, c.Queries)
	if err != nil {
		t.Fatal(err)
	}

	poisoned := make([][]float64, len(g.Head.Hidden))
	for i, row := range g.Head.Hidden {
		poisoned[i] = append([]float64(nil), row...)
	}
	for i := len(poisoned) - 5; i < len(poisoned); i++ {
		for d := range poisoned[i] {
			poisoned[i][d] = 1e6
		}
	}

	dirty, err := head.AttentionPool(poisoned, g.Head.Mask, c.Queries)
	if err != nil {
		t.Fatal(err)
	}
	for i := range clean {
		for d := range clean[i] {
			if math.Abs(clean[i][d]-dirty[i][d]) > 1e-12 {
				t.Fatalf("masked token changed pooled output at option %d dim %d", i, d)
			}
		}
	}
}

func TestBilinearLogitsMatchPython(t *testing.T) {
	g := load(t)
	h, err := head.LoadBilinear(headPath())
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range g.Head.Cases {
		pooled, err := head.AttentionPool(g.Head.Hidden, g.Head.Mask, c.Queries)
		if err != nil {
			t.Fatal(err)
		}
		logits, err := h.Logits(pooled, c.Queries)
		if err != nil {
			t.Fatalf("n=%d: %v", c.NOptions, err)
		}
		for i := range logits {
			if math.Abs(logits[i]-c.ExpectedBilinearLogits[i]) > tolerance {
				t.Errorf("n=%d option %d: got %.9g want %.9g",
					c.NOptions, i, logits[i], c.ExpectedBilinearLogits[i])
			}
		}
	}
}

func TestZeroShotLogitsMatchPython(t *testing.T) {
	g := load(t)
	for _, c := range g.Head.Cases {
		pooled, err := head.AttentionPool(g.Head.Hidden, g.Head.Mask, c.Queries)
		if err != nil {
			t.Fatal(err)
		}
		logits, err := head.ZeroShot{}.Logits(pooled, c.Queries)
		if err != nil {
			t.Fatal(err)
		}
		for i := range logits {
			if math.Abs(logits[i]-c.ExpectedZeroShotLogits[i]) > tolerance {
				t.Errorf("n=%d option %d: got %.9g want %.9g",
					c.NOptions, i, logits[i], c.ExpectedZeroShotLogits[i])
			}
		}
	}
}

func TestHeadRejectsDimensionMismatch(t *testing.T) {
	h, err := head.LoadBilinear(headPath())
	if err != nil {
		t.Fatal(err)
	}
	wrong := [][]float64{make([]float64, h.Dim()+8)}
	if _, err := h.Logits(wrong, wrong); err == nil {
		t.Error("a head/backbone dimension mismatch must fail loudly")
	}
}

func TestNPZRejectsAMissingFile(t *testing.T) {
	if _, err := head.LoadBilinear(filepath.Join(t.TempDir(), "nope.npz")); err == nil {
		t.Error("loading a missing head should fail")
	}
}
