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

// Bucket holds the fitted parameters for one (task, kind, n_options) identity.
type Bucket struct {
	Temperature float64 `json:"temperature"`
	// TaskID is the decision this bucket was fitted for; empty for a
	// shape-scoped bucket from a schema-1 artifact.
	TaskID string `json:"task_id,omitempty"`
	// Conformal maps a stringified alpha ("0.1000") to its nonconformity quantile.
	Conformal map[string]float64 `json:"conformal"`
	// CorrectnessMap is the isotonic confidence -> P(correct) curve, as
	// [][confidence, p_correct] sorted by confidence. Empty when the calibration
	// split had too few points in this bucket to fit one honestly.
	CorrectnessMap [][]float64 `json:"correctness_map"`
	NSamples       int         `json:"n_samples"`
}

// Artifact is the provenance block a schema-2 file carries.
type Artifact struct {
	ModelSHA256   string `json:"model_sha256"`
	DatasetDigest string `json:"dataset_digest"`
	Language      string `json:"language"`
	SampleCount   int    `json:"sample_count"`
}

// Schema is the artifact format this code understands. A file without the
// field is schema 1: shape-keyed buckets and no provenance. It still loads,
// and Status reports every bucket as "legacy".
const Schema = 2

// Calibrator is the Go counterpart of jef_core.calibration.Calibrator.
type Calibrator struct {
	Schema   int               `json:"calibration_schema"`
	Artifact Artifact          `json:"artifact"`
	Buckets  map[string]Bucket `json:"buckets"`
	Backbone string            `json:"backbone"`
	Head     string            `json:"head"`
	Notes    string            `json:"notes"`
	// RuntimeModelSHA256 is the digest of the head actually loaded, set by the
	// server. Compared with Artifact.ModelSHA256 for "incompatible".
	RuntimeModelSHA256 string `json:"-"`
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
	if c.Schema == 0 {
		c.Schema = 1
	}
	if c.Schema > Schema {
		return nil, fmt.Errorf("jef: calibration %s is schema %d, newer than this build understands (%d)", path, c.Schema, Schema)
	}
	return c, nil
}

// Key is the shape-only bucket identity: question kind plus option count.
// Kept for schema-1 artifacts and for questions that declare no task.
func Key(kind string, nOptions int) string {
	return fmt.Sprintf("%s:%d", kind, nOptions)
}

// KeyFor is the full bucket identity. With a task it reads
// "cve.user_interaction.v1|noul:2"; without, it is Key. The two never alias:
// a question that names its task is answered only by a bucket fitted for that
// task, and a shape-only question only by a shape-only bucket.
func KeyFor(taskID, kind string, nOptions int) string {
	if taskID == "" {
		return Key(kind, nOptions)
	}
	return taskID + "|" + Key(kind, nOptions)
}

// IsFitted reports whether any calibration was loaded.
func (c *Calibrator) IsFitted() bool { return len(c.Buckets) > 0 }

// Legacy reports a schema-1 artifact: no task identity, no provenance. Its
// numbers are still applied for display, but Status never says "validated".
func (c *Calibrator) Legacy() bool { return c.Schema < Schema }

// Incompatible reports that the artifact names a model digest and the loaded
// head has a different one. Numbers fitted for another model are noise, so
// every lookup below behaves as unfitted.
func (c *Calibrator) Incompatible() bool {
	return c.Artifact.ModelSHA256 != "" && c.RuntimeModelSHA256 != "" &&
		c.Artifact.ModelSHA256 != c.RuntimeModelSHA256
}

func (c *Calibrator) bucket(taskID, kind string, nOptions int) (Bucket, bool) {
	if c.Incompatible() {
		return Bucket{}, false
	}
	b, ok := c.Buckets[KeyFor(taskID, kind, nOptions)]
	return b, ok
}

// Status mirrors jef_core.calibration.Calibrator.status: one of validated,
// insufficient, missing, legacy, incompatible. Only validated may unlock
// automation on the Python side; the Go server reports it and never acts.
func (c *Calibrator) Status(taskID, kind string, nOptions int, alpha float64) string {
	if c.Legacy() {
		return "legacy"
	}
	if c.Incompatible() {
		return "incompatible"
	}
	b, ok := c.Buckets[KeyFor(taskID, kind, nOptions)]
	if !ok {
		return "missing"
	}
	if _, has := b.Conformal[alphaKey(alpha)]; !has || len(b.CorrectnessMap) == 0 {
		return "insufficient"
	}
	return "validated"
}

// Temperature returns the fitted temperature, or 1.0 (identity) if unfitted.
func (c *Calibrator) Temperature(kind string, nOptions int) float64 {
	return c.TemperatureFor("", kind, nOptions)
}

// TemperatureFor is Temperature for a task-scoped question.
func (c *Calibrator) TemperatureFor(taskID, kind string, nOptions int) float64 {
	if b, ok := c.bucket(taskID, kind, nOptions); ok {
		return b.Temperature
	}
	return 1.0
}

// Apply turns logits into calibrated probabilities.
func (c *Calibrator) Apply(logits []float64, kind string, nOptions int) ([]float64, error) {
	return c.ApplyFor("", logits, kind, nOptions)
}

// ApplyFor is Apply for a task-scoped question.
func (c *Calibrator) ApplyFor(taskID string, logits []float64, kind string, nOptions int) ([]float64, error) {
	return mathx.Softmax(logits, c.TemperatureFor(taskID, kind, nOptions))
}

// PCorrect maps Jev's confidence statistic onto the probability that the top
// answer is right.
//
// The second return value is false when no correctness map was fitted. That is
// the honest answer, and a gate must be able to tell it apart from a low
// probability: "we cannot say how reliable this is" and "this is unreliable"
// lead to different decisions.
func (c *Calibrator) PCorrect(confidence float64, kind string, nOptions int) (float64, bool) {
	return c.PCorrectFor("", confidence, kind, nOptions)
}

// PCorrectFor is PCorrect for a task-scoped question.
func (c *Calibrator) PCorrectFor(taskID string, confidence float64, kind string, nOptions int) (float64, bool) {
	b, ok := c.bucket(taskID, kind, nOptions)
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
// It fails closed, exactly as the Python side does since PR 4: without a
// bucket, without a quantile for this alpha, or when no probability clears the
// threshold, it returns every index. That is the only set whose coverage is
// guaranteed without a quantile, and it matters: an empty or singleton set
// would let a gate written as `set_size <= 2` fire on an uncalibrated server
// and automate a decision nobody could stand behind. An earlier version
// returned the argmax on an empty keep set, which dressed an uncalibrated
// guess up as a certified singleton.
func (c *Calibrator) PredictionSet(probabilities []float64, kind string, nOptions int, alpha float64) []int {
	return c.PredictionSetFor("", probabilities, kind, nOptions, alpha)
}

// PredictionSetFor is PredictionSet for a task-scoped question.
func (c *Calibrator) PredictionSetFor(taskID string, probabilities []float64, kind string, nOptions int, alpha float64) []int {
	b, ok := c.bucket(taskID, kind, nOptions)
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
		return allIndices(len(probabilities))
	}
	return keep
}

// HasCorrectnessMap reports whether P(correct) is available for a bucket.
func (c *Calibrator) HasCorrectnessMap(kind string, nOptions int) bool {
	return c.HasCorrectnessMapFor("", kind, nOptions)
}

// HasCorrectnessMapFor is HasCorrectnessMap for a task-scoped question.
func (c *Calibrator) HasCorrectnessMapFor(taskID, kind string, nOptions int) bool {
	b, ok := c.bucket(taskID, kind, nOptions)
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
