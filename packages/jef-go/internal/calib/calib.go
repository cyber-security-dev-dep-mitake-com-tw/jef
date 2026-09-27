// Package calib reads the calibration artifact the Python trainer produces and
// applies it identically.
//
// The artifact is plain JSON on purpose: a serving path that needed pickle, or
// a Python interpreter, could not ship as a single static Go binary. Everything
// the Go server needs to turn logits into calibrated probabilities, a
// prediction set with a coverage guarantee, and an honest P(correct) is in that
// one file.
package calib

import (
	"encoding/json"
	"fmt"
	"os"
	"sort"

	"github.com/cyber-security-dev-dep-mitake-com-tw/jef/jef-go/internal/mathx"
)

// Bucket holds the fitted parameters for one (kind, n_options) pair.
type Bucket struct {
	Temperature float64 `json:"temperature"`
	// Conformal maps a stringified alpha ("0.1000") to its nonconformity quantile.
	Conformal map[string]float64 `json:"conformal"`
	// CorrectnessMap is the isotonic confidence -> P(correct) curve, as
	// [][confidence, p_correct] sorted by confidence. Empty when the calibration
	// split had too few points in this bucket to fit one honestly.
	CorrectnessMap [][]float64 `json:"correctness_map"`
	NSamples       int         `json:"n_samples"`
}

// Calibrator is the Go counterpart of jef_core.calibration.Calibrator.
type Calibrator struct {
	Buckets  map[string]Bucket `json:"buckets"`
	Backbone string            `json:"backbone"`
	Head     string            `json:"head"`
	Notes    string            `json:"notes"`
}

// New returns an unfitted calibrator: identity temperature, no P(correct), and
// a prediction set that excludes nothing.
func New() *Calibrator {
	return &Calibrator{Buckets: map[string]Bucket{}}
}

// Load reads a calibration.json produced by `python -m jef_train.fit`.
func Load(path string) (*Calibrator, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("jef: reading calibration %s: %w", path, err)
	}
	c := New()
	if err := json.Unmarshal(raw, c); err != nil {
		return nil, fmt.Errorf("jef: parsing calibration %s: %w", path, err)
	}
	if c.Buckets == nil {
		c.Buckets = map[string]Bucket{}
	}
	return c, nil
}

// Key is the bucket identity: question kind plus option count. A 2-option noul
// and a 2-option choice are separate buckets because their sharpness profiles
// differ even at identical width.
func Key(kind string, nOptions int) string {
	return fmt.Sprintf("%s:%d", kind, nOptions)
}

// IsFitted reports whether any calibration was loaded.
func (c *Calibrator) IsFitted() bool { return len(c.Buckets) > 0 }

// Temperature returns the fitted temperature, or 1.0 (identity) if unfitted.
func (c *Calibrator) Temperature(kind string, nOptions int) float64 {
	if b, ok := c.Buckets[Key(kind, nOptions)]; ok {
		return b.Temperature
	}
	return 1.0
}

// Apply turns logits into calibrated probabilities.
func (c *Calibrator) Apply(logits []float64, kind string, nOptions int) ([]float64, error) {
	return mathx.Softmax(logits, c.Temperature(kind, nOptions))
}

// PCorrect maps Jev's confidence statistic onto the probability that the top
// answer is right.
//
// The second return value is false when no correctness map was fitted. That is
// the honest answer, and a gate must be able to tell it apart from a low
// probability: "we cannot say how reliable this is" and "this is unreliable"
// lead to different decisions.
func (c *Calibrator) PCorrect(confidence float64, kind string, nOptions int) (float64, bool) {
	b, ok := c.Buckets[Key(kind, nOptions)]
	if !ok || len(b.CorrectnessMap) == 0 {
		return 0, false
	}
	points := b.CorrectnessMap
	if confidence <= points[0][0] {
		return points[0][1], true
	}
	last := points[len(points)-1]
	if confidence >= last[0] {
		return last[1], true
	}
	for i := 0; i+1 < len(points); i++ {
		x0, y0 := points[i][0], points[i][1]
		x1, y1 := points[i+1][0], points[i+1][1]
		if confidence >= x0 && confidence <= x1 {
			if x1 == x0 {
				return y1, true
			}
			ratio := (confidence - x0) / (x1 - x0)
			return y0 + ratio*(y1-y0), true
		}
	}
	return last[1], true
}

// PredictionSet returns the option indices whose probability clears the
// conformal threshold for this miscoverage level.
//
// Without a fitted quantile it returns every index. That is the only set that
// honestly covers at any level, and it matters: returning an empty or singleton
// set would let a gate written as `set_size <= 2` fire on an uncalibrated
// server and automate a decision nobody could stand behind.
func (c *Calibrator) PredictionSet(probabilities []float64, kind string, nOptions int, alpha float64) []int {
	b, ok := c.Buckets[Key(kind, nOptions)]
	if !ok {
		return allIndices(len(probabilities))
	}
	qhat, ok := b.Conformal[alphaKey(alpha)]
	if !ok {
		return allIndices(len(probabilities))
	}

	keep := make([]int, 0, len(probabilities))
	for i, p := range probabilities {
		if p >= 1.0-qhat {
			keep = append(keep, i)
		}
	}
	if len(keep) == 0 {
		// An empty set is never the right answer: some option was chosen.
		return []int{mathx.ArgMax(probabilities)}
	}
	return keep
}

// HasCorrectnessMap reports whether P(correct) is available for a bucket.
func (c *Calibrator) HasCorrectnessMap(kind string, nOptions int) bool {
	b, ok := c.Buckets[Key(kind, nOptions)]
	return ok && len(b.CorrectnessMap) > 0
}

// BucketNames lists the fitted buckets, sorted, for /healthz and /v1/models.
func (c *Calibrator) BucketNames() []string {
	names := make([]string, 0, len(c.Buckets))
	for k := range c.Buckets {
		names = append(names, k)
	}
	sort.Strings(names)
	return names
}

func alphaKey(alpha float64) string { return fmt.Sprintf("%.4f", alpha) }

func allIndices(n int) []int {
	out := make([]int, n)
	for i := range out {
		out[i] = i
	}
	return out
}
