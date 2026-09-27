package contract

import (
	"bytes"
	"io"
	"math"
)

func newReader(b []byte) io.Reader { return bytes.NewReader(b) }

// round applies half-to-even at the requested decimal place.
//
// Python's round() rounds on the decimal representation rather than by
// multiply-round-divide, so the two can differ in the last representable bit
// for values sitting exactly on a tie. At six decimals over a probability that
// is far below anything a gate threshold can distinguish, and the golden
// fixtures assert the distributions themselves rather than their rendering.
func round(v float64, decimals int) float64 {
	factor := math.Pow(10, float64(decimals))
	return math.RoundToEven(v*factor) / factor
}
