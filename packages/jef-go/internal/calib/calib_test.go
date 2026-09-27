package calib_test

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/cyber-security-dev-dep-mitake-com-tw/jef/jef-go/internal/calib"
	"github.com/cyber-security-dev-dep-mitake-com-tw/jef/jef-go/internal/mathx"
)

const tolerance = 1e-9

type goldenFile struct {
	Calibrator  calib.Calibrator `json:"calibrator"`
	Calibration []struct {
		Kind                  string    `json:"kind"`
		NOptions              int       `json:"n_options"`
		Logits                []float64 `json:"logits"`
		ExpectedTemperature   float64   `json:"expected_temperature"`
		ExpectedProbs         []float64 `json:"expected_probabilities"`
		ExpectedConfidence    float64   `json:"expected_confidence"`
		ExpectedPCorrect      *float64  `json:"expected_p_correct"`
		ExpectedPredictionSet []int     `json:"expected_prediction_set_010"`
	} `json:"calibration"`
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
	return g
}

func TestCalibrationMatchesPythonEndToEnd(t *testing.T) {
	g := load(t)
	if len(g.Calibration) == 0 {
		t.Fatal("no calibration fixtures")
	}
	c := &g.Calibrator

	for i, tc := range g.Calibration {
		if got := c.Temperature(tc.Kind, tc.NOptions); math.Abs(got-tc.ExpectedTemperature) > tolerance {
			t.Errorf("case %d temperature: got %.17g want %.17g", i, got, tc.ExpectedTemperature)
		}

		probs, err := c.Apply(tc.Logits, tc.Kind, tc.NOptions)
		if err != nil {
			t.Fatalf("case %d: %v", i, err)
		}
		for j := range probs {
			if math.Abs(probs[j]-tc.ExpectedProbs[j]) > tolerance {
				t.Errorf("case %d prob[%d]: got %.17g want %.17g", i, j, probs[j], tc.ExpectedProbs[j])
			}
		}

		conf, err := mathx.Confidence(probs)
		if err != nil {
			t.Fatalf("case %d: %v", i, err)
		}
		if math.Abs(conf-tc.ExpectedConfidence) > tolerance {
			t.Errorf("case %d confidence: got %.17g want %.17g", i, conf, tc.ExpectedConfidence)
		}

		gotP, ok := c.PCorrect(conf, tc.Kind, tc.NOptions)
		switch {
		case tc.ExpectedPCorrect == nil && ok:
			t.Errorf("case %d: Go produced P(correct)=%v where Python had none", i, gotP)
		case tc.ExpectedPCorrect != nil && !ok:
			t.Errorf("case %d: Go has no P(correct) where Python had %v", i, *tc.ExpectedPCorrect)
		case tc.ExpectedPCorrect != nil && math.Abs(gotP-*tc.ExpectedPCorrect) > tolerance:
			t.Errorf("case %d p_correct: got %.17g want %.17g", i, gotP, *tc.ExpectedPCorrect)
		}

		set := c.PredictionSet(probs, tc.Kind, tc.NOptions, 0.10)
		if !reflect.DeepEqual(set, tc.ExpectedPredictionSet) {
			t.Errorf("case %d prediction set: got %v want %v", i, set, tc.ExpectedPredictionSet)
		}
	}
}

func TestUnfittedCalibratorIsIdentityAndExcludesNothing(t *testing.T) {
	c := calib.New()
	if c.IsFitted() {
		t.Error("a fresh calibrator must not claim to be fitted")
	}
	if got := c.Temperature("choice", 5); got != 1.0 {
		t.Errorf("unfitted temperature should be identity, got %v", got)
	}
	if _, ok := c.PCorrect(0.9, "choice", 5); ok {
		t.Error("unfitted calibrator must not invent a P(correct)")
	}
	// Excluding nothing is the only honest set without a quantile -- and it
	// stops `set_size <= 2` automating on an uncalibrated server.
	set := c.PredictionSet([]float64{0.9, 0.05, 0.03, 0.01, 0.01}, "choice", 5, 0.10)
	if len(set) != 5 {
		t.Errorf("uncalibrated prediction set should hold every option, got %v", set)
	}
}

func TestPredictionSetIsNeverEmpty(t *testing.T) {
	g := load(t)
	c := &g.Calibrator
	// A distribution so flat nothing clears the threshold still yields the argmax:
	// some option was chosen, so the set cannot be empty.
	flat := []float64{0.2, 0.2, 0.2, 0.2, 0.2}
	if got := c.PredictionSet(flat, "choice", 5, 0.10); len(got) == 0 {
		t.Error("prediction set must never be empty")
	}
}

func TestPCorrectIsClampedToTheFittedRange(t *testing.T) {
	g := load(t)
	c := &g.Calibrator
	for _, v := range []float64{-1, 0, 0.5, 1, 2} {
		p, ok := c.PCorrect(v, "noul", 2)
		if !ok {
			t.Skip("fixture has no correctness map for noul:2")
		}
		if p < 0 || p > 1 {
			t.Errorf("P(correct)=%v for confidence %v is outside [0,1]", p, v)
		}
	}
}

func TestLoadRejectsAMissingFile(t *testing.T) {
	if _, err := calib.Load(filepath.Join(t.TempDir(), "nope.json")); err == nil {
		t.Error("loading a missing calibration should fail loudly")
	}
}

func TestLoadRoundTripsThePythonArtifact(t *testing.T) {
	g := load(t)
	path := filepath.Join(t.TempDir(), "calibration.json")
	raw, err := json.Marshal(g.Calibrator)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, raw, 0o600); err != nil {
		t.Fatal(err)
	}
	loaded, err := calib.Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if !loaded.IsFitted() {
		t.Error("round-tripped calibrator lost its buckets")
	}
	if len(loaded.BucketNames()) != len(g.Calibrator.Buckets) {
		t.Errorf("bucket count changed: %d vs %d", len(loaded.BucketNames()), len(g.Calibrator.Buckets))
	}
}
