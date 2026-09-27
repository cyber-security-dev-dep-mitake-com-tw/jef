package backbone_test

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/cyber-security-dev-dep-mitake-com-tw/jef/jef-go/internal/backbone"
)

// float32 epsilon: Python stores these vectors as float32 and Go computes in
// float64 before rounding to float32 at the same points.
const tolerance = 1e-6

type goldenFile struct {
	Hashing struct {
		Dim   int `json:"dim"`
		Cases []struct {
			Text               string    `json:"text"`
			NTokens            int       `json:"n_tokens"`
			ExpectedStateFirst []float64 `json:"expected_state_first"`
			ExpectedStateLast  []float64 `json:"expected_state_last"`
			ExpectedQuery      []float64 `json:"expected_query"`
		} `json:"cases"`
	} `json:"hashing_backbone"`
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
	if len(g.Hashing.Cases) == 0 {
		t.Fatal("golden fixtures carry no hashing cases")
	}
	return g
}

// TestHashingStubMatchesPythonExactly is what makes the conformance story real.
//
// With both stubs producing the same vectors, the same acceptance suite can be
// pointed at the Python server or the Go one and expect identical answers --
// without a 300MB model in CI. If this drifts, "the two implementations are
// equivalent" goes back to being a claim.
func TestHashingStubMatchesPythonExactly(t *testing.T) {
	g := load(t)
	bb := backbone.NewHashing(g.Hashing.Dim)

	for _, c := range g.Hashing.Cases {
		enc, err := bb.EncodeState(c.Text)
		if err != nil {
			t.Fatalf("%q: %v", c.Text, err)
		}
		if enc.NTokens != c.NTokens {
			t.Errorf("%q: got %d tokens, want %d", c.Text, enc.NTokens, c.NTokens)
			continue
		}

		compare(t, c.Text+" [state first]", enc.Hidden[0], c.ExpectedStateFirst)
		compare(t, c.Text+" [state last]", enc.Hidden[len(enc.Hidden)-1], c.ExpectedStateLast)

		queries, err := bb.EncodeQueries([]string{c.Text})
		if err != nil {
			t.Fatalf("%q: %v", c.Text, err)
		}
		compare(t, c.Text+" [query]", queries[0], c.ExpectedQuery)
	}
}

func compare(t *testing.T, label string, got, want []float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Errorf("%s: got %d dims, want %d", label, len(got), len(want))
		return
	}
	for i := range got {
		if math.Abs(got[i]-want[i]) > tolerance {
			t.Errorf("%s: dim %d got %.9g want %.9g", label, i, got[i], want[i])
			return
		}
	}
}

func TestHashingIsDeterministic(t *testing.T) {
	bb := backbone.NewHashing(64)
	a, _ := bb.EncodeState("同樣的文字")
	b, _ := bb.EncodeState("同樣的文字")
	for i := range a.Hidden {
		for d := range a.Hidden[i] {
			if a.Hidden[i][d] != b.Hidden[i][d] {
				t.Fatalf("the stub must be deterministic; differed at token %d dim %d", i, d)
			}
		}
	}
}

func TestChineseIsNotCollapsedIntoOneToken(t *testing.T) {
	// Whitespace splitting would make this a single token. zh-TW is a
	// first-class language here, so chunking is by character.
	bb := backbone.NewHashing(32)
	enc, err := bb.EncodeState("付款服務連續三天失敗無空白字元")
	if err != nil {
		t.Fatal(err)
	}
	if enc.NTokens < 2 {
		t.Errorf("Chinese text collapsed into %d token(s)", enc.NTokens)
	}
}

func TestEmptyTextStillProducesOneToken(t *testing.T) {
	bb := backbone.NewHashing(16)
	enc, err := bb.EncodeState("")
	if err != nil {
		t.Fatal(err)
	}
	if enc.NTokens != 1 || len(enc.Hidden) != 1 {
		t.Errorf("empty state should yield exactly one token, got %d", enc.NTokens)
	}
}

func TestVectorsAreUnitLength(t *testing.T) {
	bb := backbone.NewHashing(128)
	enc, _ := bb.EncodeState("付款失敗")
	for i, v := range enc.Hidden {
		var sum float64
		for _, x := range v {
			sum += x * x
		}
		if math.Abs(math.Sqrt(sum)-1) > 1e-5 {
			t.Errorf("token %d has norm %v, want 1", i, math.Sqrt(sum))
		}
	}
}
