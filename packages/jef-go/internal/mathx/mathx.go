// Package mathx is the Go side of JEF's probability maths.
//
// Every function here has a Python counterpart in jef_core.mathx, and the two
// must agree: a deployment can serve the Python implementation and the Go one
// behind the same load balancer, and an answer that depends on which binary
// took the request is not an answer. The parity is not assumed -- Python emits
// golden fixtures in testdata/ and the Go tests assert against them.
package mathx

import (
	"errors"
	"math"
)

var (
	// ErrTooFewOptions mirrors Python's ValueError for degenerate questions.
	ErrTooFewOptions = errors.New("jef: need at least 2 options")
	// ErrBadTemperature mirrors Python's rejection of non-positive temperature.
	ErrBadTemperature = errors.New("jef: temperature must be > 0")
)

// Softmax converts logits to a distribution, with optional temperature scaling.
//
// Temperature > 1 flattens, < 1 sharpens. It is the single scalar fitted during
// calibration. The max-subtraction is not an optimisation: without it a logit
// around 1000 overflows to +Inf and the whole distribution becomes NaN.
func Softmax(logits []float64, temperature float64) ([]float64, error) {
	if temperature <= 0 {
		return nil, ErrBadTemperature
	}
	if len(logits) == 0 {
		return nil, ErrTooFewOptions
	}

	scaled := make([]float64, len(logits))
	maxLogit := math.Inf(-1)
	for i, v := range logits {
		scaled[i] = v / temperature
		if scaled[i] > maxLogit {
			maxLogit = scaled[i]
		}
	}

	out := make([]float64, len(logits))
	var total float64
	for i, v := range scaled {
		out[i] = math.Exp(v - maxLogit)
		total += out[i]
	}

	if total == 0 || math.IsInf(total, 0) || math.IsNaN(total) {
		// Degenerate input: a uniform distribution is wrong but finite, which
		// beats propagating NaN into a gate condition.
		uniform := 1.0 / float64(len(logits))
		for i := range out {
			out[i] = uniform
		}
		return out, nil
	}
	for i := range out {
		out[i] /= total
	}
	return out, nil
}

// Confidence is Jev's statistic: (n*peak - 1) / (n - 1).
//
// It measures how peaked the distribution is, not how likely the answer is to
// be correct. Use calib.PCorrect for the latter; a gate threshold set on this
// number alone means "the model was decisive", which is not the same thing.
func Confidence(probabilities []float64) (float64, error) {
	n := len(probabilities)
	if n < 2 {
		return 0, ErrTooFewOptions
	}
	peak := 0.0
	for _, p := range probabilities {
		if p > peak {
			peak = p
		}
	}
	c := (float64(n)*peak - 1.0) / (float64(n) - 1.0)
	return math.Min(1, math.Max(0, c)), nil
}

// ScoreExpectation is the continuous score: sum(i * p_i) over ordered levels.
//
// This is why a score answer may land between levels rather than snapping to
// one, and why level order is load-bearing: reversing the levels inverts the
// scale silently.
func ScoreExpectation(probabilities []float64) (float64, error) {
	if len(probabilities) < 2 {
		return 0, ErrTooFewOptions
	}
	var sum float64
	for i, p := range probabilities {
		sum += float64(i) * p
	}
	return sum, nil
}

// ArgMax returns the index of the largest value, ties going to the lowest index
// -- matching numpy.argmax, so Python and Go pick the same option.
func ArgMax(values []float64) int {
	best, bestIdx := math.Inf(-1), 0
	for i, v := range values {
		if v > best {
			best, bestIdx = v, i
		}
	}
	return bestIdx
}

// BrierScore is the multiclass Brier score for one prediction (lower is better).
func BrierScore(probabilities []float64, correct int) float64 {
	var sum float64
	for i, p := range probabilities {
		target := 0.0
		if i == correct {
			target = 1.0
		}
		sum += (p - target) * (p - target)
	}
	return sum
}
