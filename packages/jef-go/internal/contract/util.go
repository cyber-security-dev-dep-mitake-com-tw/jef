package contract

import (
	"bytes"
	"io"
	"strconv"
)

func newReader(b []byte) io.Reader { return bytes.NewReader(b) }

// round matches Python's round(x, n) exactly.
//
// The obvious implementation -- math.RoundToEven(v*1e6)/1e6 -- rounds a value
// that has *already* lost precision to the multiplication, so it disagrees with
// Python on exact ties. That surfaced as a real divergence between the two
// servers: 0.369918 from Python against 0.369917 from Go for the same
// distribution. One part in a million is far below anything a gate threshold
// can see, but "the two implementations answer identically" should be true
// rather than nearly true, and a claim that is nearly true is the kind that
// quietly stops being true.
//
// strconv.FormatFloat performs correctly-rounded decimal conversion on the
// exact binary value, which is the same thing CPython's round does.
func round(v float64, decimals int) float64 {
	rounded, err := strconv.ParseFloat(strconv.FormatFloat(v, 'f', decimals, 64), 64)
	if err != nil {
		return v
	}
	return rounded
}
