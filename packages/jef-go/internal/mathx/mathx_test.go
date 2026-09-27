package mathx_test

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/cyber-security-dev-dep-mitake-com-tw/jef/jef-go/internal/mathx"
)

// tolerance is tight on purpose. These are the same formulas over the same
// float64s; anything looser would hide a real divergence between the Python and
// Go serving paths rather than catch it.
const tolerance = 1e-9

type golden struct {
	Softmax []struct {
		Logits      []float64 `json:"logits"`
		Temperature float64   `json:"temperature"`
		Expected    []float64 `json:"expected"`
	} `json:"softmax"`
	Confidence []struct {
		Probabilities []float64 `json:"probabilities"`
		Expected      float64   `json:"expected"`
	} `json:"confidence"`
	ScoreExpectation []struct {
		Probabilities []float64 `json:"probabilities"`
		Expected      float64   `json:"expected"`
	} `json:"score_expectation"`
	Brier []struct {
		Probabilities []float64 `json:"probabilities"`
		Correct       int       `json:"correct"`
		Expected      float64   `json:"expected"`
	} `json:"brier"`
}

func loadGolden(t *testing.T) golden {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", "..", "testdata", "golden.json"))
	if err != nil {
		t.Fatalf("golden fixtures missing (regenerate with `python -m jef_train.golden`): %v", err)
	}
	var g golden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("golden fixtures unreadable: %v", err)
	}
	return g
}

func TestSoftmaxMatchesPython(t *testing.T) {
	g := loadGolden(t)
	if len(g.Softmax) == 0 {
		t.Fatal("no softmax fixtures")
	}
	for i, c := range g.Softmax {
		got, err := mathx.Softmax(c.Logits, c.Temperature)
		if err != nil {
			t.Fatalf("case %d: %v", i, err)
		}
		for j := range got {
			if math.Abs(got[j]-c.Expected[j]) > tolerance {
				t.Errorf("case %d (logits=%v T=%v) index %d: got %.17g want %.17g",
					i, c.Logits, c.Temperature, j, got[j], c.Expected[j])
			}
		}
	}
}

func TestSoftmaxSurvivesOverflow(t *testing.T) {
	// Without max-subtraction this is Inf/Inf. A NaN reaching a gate condition
	// is a silent wrong decision, not a crash.
	got, err := mathx.Softmax([]float64{1000, -1000}, 1.0)
	if err != nil {
		t.Fatal(err)
	}
	var sum float64
	for _, p := range got {
		if math.IsNaN(p) || math.IsInf(p, 0) {
			t.Fatalf("non-finite probability: %v", got)
		}
		sum += p
	}
	if math.Abs(sum-1.0) > tolerance {
		t.Errorf("probabilities sum to %v, want 1", sum)
	}
}

func TestSoftmaxRejectsNonPositiveTemperature(t *testing.T) {
	for _, temp := range []float64{0, -1} {
		if _, err := mathx.Softmax([]float64{1, 2}, temp); err == nil {
			t.Errorf("temperature %v should be rejected", temp)
		}
	}
}

func TestConfidenceMatchesPython(t *testing.T) {
	for i, c := range loadGolden(t).Confidence {
		got, err := mathx.Confidence(c.Probabilities)
		if err != nil {
			t.Fatalf("case %d: %v", i, err)
		}
		if math.Abs(got-c.Expected) > tolerance {
			t.Errorf("case %d (%v): got %.17g want %.17g", i, c.Probabilities, got, c.Expected)
		}
	}
}

func TestConfidenceNeedsTwoOptions(t *testing.T) {
	if _, err := mathx.Confidence([]float64{1.0}); err == nil {
		t.Error("a single-option question should be rejected")
	}
}

func TestScoreExpectationMatchesPython(t *testing.T) {
	for i, c := range loadGolden(t).ScoreExpectation {
		got, err := mathx.ScoreExpectation(c.Probabilities)
		if err != nil {
			t.Fatalf("case %d: %v", i, err)
		}
		if math.Abs(got-c.Expected) > tolerance {
			t.Errorf("case %d (%v): got %.17g want %.17g", i, c.Probabilities, got, c.Expected)
		}
	}
}

func TestScoreLandsBetweenLevels(t *testing.T) {
	got, err := mathx.ScoreExpectation([]float64{0, 0.5, 0.5, 0})
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(got-1.5) > tolerance {
		t.Errorf("mass split between levels 1 and 2 should score 1.5, got %v", got)
	}
}

func TestBrierMatchesPython(t *testing.T) {
	for i, c := range loadGolden(t).Brier {
		if got := mathx.BrierScore(c.Probabilities, c.Correct); math.Abs(got-c.Expected) > tolerance {
			t.Errorf("case %d: got %.17g want %.17g", i, got, c.Expected)
		}
	}
}

func TestArgMaxTiesGoToTheLowestIndex(t *testing.T) {
	// numpy.argmax does this. Differing would make Python and Go pick different
	// options for a perfectly balanced distribution.
	if got := mathx.ArgMax([]float64{0.25, 0.25, 0.25, 0.25}); got != 0 {
		t.Errorf("got %d, want 0", got)
	}
	if got := mathx.ArgMax([]float64{0.1, 0.4, 0.4, 0.1}); got != 1 {
		t.Errorf("got %d, want 1", got)
	}
}
