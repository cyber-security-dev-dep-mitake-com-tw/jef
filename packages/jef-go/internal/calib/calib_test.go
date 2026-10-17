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

func TestPredictionSetFailsClosedWhenNothingClearsTheThreshold(t *testing.T) {
	// Parity with jef_core.calibration since PR 4: an empty keep set is the full
	// set, never the argmax. A singleton here would read as "safe to automate".
	c := &calib.Calibrator{Schema: calib.Schema, Buckets: map[string]calib.Bucket{
		"choice:2": {Temperature: 1, Conformal: map[string]float64{"0.1000": 0.05}},
	}}
	if got := c.PredictionSet([]float64{0.6, 0.4}, "choice", 2, 0.10); !reflect.DeepEqual(got, []int{0, 1}) {
		t.Errorf("empty keep set must become the full set, got %v", got)
	}
	if got := c.PredictionSet([]float64{0.99, 0.01}, "choice", 2, 0.10); !reflect.DeepEqual(got, []int{0}) {
		t.Errorf("a probability that clears the threshold yields the singleton, got %v", got)
	}
}

func TestTaskScopedBucketsNeverBorrow(t *testing.T) {
	// POL-002: task A is calibrated; task B shares the shape and gets nothing.
	c := &calib.Calibrator{Schema: calib.Schema, Buckets: map[string]calib.Bucket{
		"cve.user_interaction.v1|noul:2": {
			Temperature:    1.3,
			TaskID:         "cve.user_interaction.v1",
			Conformal:      map[string]float64{"0.1000": 0.4},
			CorrectnessMap: [][]float64{{0.1, 0.5}, {0.9, 0.95}},
		},
	}}
	if got := c.Status("cve.user_interaction.v1", "noul", 2, 0.10); got != "validated" {
		t.Errorf("own task: want validated, got %s", got)
	}
	if got := c.Status("soar_zh.false_positive.v1", "noul", 2, 0.10); got != "missing" {
		t.Errorf("other task of the same shape: want missing, got %s", got)
	}
	if got := c.Status("", "noul", 2, 0.10); got != "missing" {
		t.Errorf("shape-only question: want missing, got %s", got)
	}
	if got := c.TemperatureFor("soar_zh.false_positive.v1", "noul", 2); got != 1.0 {
		t.Errorf("other task must get identity temperature, got %v", got)
	}
	if _, ok := c.PCorrectFor("soar_zh.false_positive.v1", 0.9, "noul", 2); ok {
		t.Error("other task must not get a P(correct)")
	}
	if got := c.PredictionSetFor("soar_zh.false_positive.v1", []float64{0.99, 0.01}, "noul", 2, 0.10); len(got) != 2 {
		t.Errorf("other task must get the full set, got %v", got)
	}
}

func TestLegacyAndIncompatibleStatus(t *testing.T) {
	g := load(t)
	legacy := g.Calibrator
	if legacy.Schema == 0 {
		legacy.Schema = 1 // the golden fixture predates the schema field
	}
	if !legacy.Legacy() {
		t.Fatal("a schema-1 artifact must report Legacy()")
	}
	if got := legacy.Status("", "choice", 5, 0.10); got != "legacy" {
		t.Errorf("legacy status: got %s", got)
	}
	if _, ok := legacy.PCorrect(0.9, "choice", 5); !ok {
		t.Error("legacy numbers are still shown for analysis")
	}

	c := &calib.Calibrator{Schema: calib.Schema, Artifact: calib.Artifact{ModelSHA256: "sha256:aaaa"}, Buckets: map[string]calib.Bucket{
		"t.v1|choice:2": {Temperature: 2, Conformal: map[string]float64{"0.1000": 0.4}, CorrectnessMap: [][]float64{{0, 0.5}, {1, 0.9}}},
	}}
	c.RuntimeModelSHA256 = "sha256:aaaa"
	if got := c.Status("t.v1", "choice", 2, 0.10); got != "validated" {
		t.Errorf("matching digest: got %s", got)
	}
	c.RuntimeModelSHA256 = "sha256:bbbb"
	if got := c.Status("t.v1", "choice", 2, 0.10); got != "incompatible" {
		t.Errorf("mismatched digest: got %s", got)
	}
	if got := c.TemperatureFor("t.v1", "choice", 2); got != 1.0 {
		t.Errorf("incompatible artifact must apply identity temperature, got %v", got)
	}
	if _, ok := c.PCorrectFor("t.v1", 0.9, "choice", 2); ok {
		t.Error("incompatible artifact must not give a P(correct)")
	}
}

func TestSchemaTwoFileLoads(t *testing.T) {
	path := filepath.Join(t.TempDir(), "calibration.json")
	raw := `{"calibration_schema": 2, "backbone": "b", "head": "h", "notes": "",
	  "artifact": {"model_sha256": "sha256:abc", "language": "en", "sample_count": 3},
	  "buckets": {"cve.severity.v1|score:5": {"temperature": 1.1, "conformal": {"0.1000": 0.3},
	    "correctness_map": [[0.2, 0.4], [0.9, 0.9]], "n_samples": 3, "task_id": "cve.severity.v1", "metrics": {"ece": 0.1}}}}`
	if err := os.WriteFile(path, []byte(raw), 0o600); err != nil {
		t.Fatal(err)
	}
	c, err := calib.Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if c.Legacy() || c.Artifact.ModelSHA256 != "sha256:abc" {
		t.Errorf("schema-2 provenance not read: %+v", c.Artifact)
	}
	if got := c.Status("cve.severity.v1", "score", 5, 0.10); got != "validated" {
		t.Errorf("want validated, got %s", got)
	}
	if got := c.Status("", "score", 5, 0.10); got != "missing" {
		t.Errorf("shape-only lookup must miss a task bucket, got %s", got)
	}
	if _, err := calib.Load(writeJSON(t, `{"calibration_schema": 99, "buckets": {}}`)); err == nil {
		t.Error("a newer schema must be refused")
	}
}

func writeJSON(t *testing.T, raw string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "c.json")
	if err := os.WriteFile(path, []byte(raw), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
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
