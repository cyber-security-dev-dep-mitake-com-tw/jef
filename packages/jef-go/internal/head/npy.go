// Minimal .npy / .npz reader.
//
// The trained head is a .npz -- a zip of .npy arrays -- because that is what
// numpy writes and what the Python side loads. Rather than introduce a second
// artifact format for Go, or a conversion step that could silently drift, the
// Go server reads the same file. Only what JEF actually stores is supported:
// little-endian float32/float64, C-contiguous, 1-D or 2-D, no pickled objects.
package head

import (
	"archive/zip"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"regexp"
	"strconv"
	"strings"
)

// Array is a decoded .npy payload, flattened, with its shape.
type Array struct {
	Shape []int
	Data  []float64
}

// At indexes a 2-D array.
func (a Array) At(row, col int) float64 { return a.Data[row*a.Shape[1]+col] }

// Rows and Cols describe a 2-D array.
func (a Array) Rows() int { return a.Shape[0] }
func (a Array) Cols() int {
	if len(a.Shape) < 2 {
		return 1
	}
	return a.Shape[1]
}

// Scalar returns the single value of a 0-D or 1-element array.
func (a Array) Scalar() (float64, error) {
	if len(a.Data) != 1 {
		return 0, fmt.Errorf("jef: expected a scalar, got shape %v", a.Shape)
	}
	return a.Data[0], nil
}

// errUnsupportedDtype marks a member ReadNPZ may skip rather than fail on.
var errUnsupportedDtype = errors.New("jef: unsupported .npy dtype")

var (
	descrRe = regexp.MustCompile(`'descr':\s*'([^']+)'`)
	orderRe = regexp.MustCompile(`'fortran_order':\s*(True|False)`)
	shapeRe = regexp.MustCompile(`'shape':\s*\(([^)]*)\)`)
)

// ReadNPY decodes one .npy stream.
func ReadNPY(r io.Reader) (Array, error) {
	var magic [6]byte
	if _, err := io.ReadFull(r, magic[:]); err != nil {
		return Array{}, fmt.Errorf("jef: reading npy magic: %w", err)
	}
	if string(magic[:]) != "\x93NUMPY" {
		return Array{}, fmt.Errorf("jef: not a .npy file (magic %q)", magic)
	}

	var version [2]byte
	if _, err := io.ReadFull(r, version[:]); err != nil {
		return Array{}, fmt.Errorf("jef: reading npy version: %w", err)
	}

	var headerLen int
	switch version[0] {
	case 1:
		var n uint16
		if err := binary.Read(r, binary.LittleEndian, &n); err != nil {
			return Array{}, fmt.Errorf("jef: reading npy header length: %w", err)
		}
		headerLen = int(n)
	case 2, 3:
		var n uint32
		if err := binary.Read(r, binary.LittleEndian, &n); err != nil {
			return Array{}, fmt.Errorf("jef: reading npy header length: %w", err)
		}
		headerLen = int(n)
	default:
		return Array{}, fmt.Errorf("jef: unsupported .npy version %d.%d", version[0], version[1])
	}

	header := make([]byte, headerLen)
	if _, err := io.ReadFull(r, header); err != nil {
		return Array{}, fmt.Errorf("jef: reading npy header: %w", err)
	}
	meta := string(header)

	descr := descrRe.FindStringSubmatch(meta)
	if descr == nil {
		return Array{}, fmt.Errorf("jef: npy header has no dtype: %q", meta)
	}
	if m := orderRe.FindStringSubmatch(meta); m != nil && m[1] == "True" {
		// JEF never writes Fortran order; if one appears, the file did not come
		// from this pipeline and silently transposing it would be worse.
		return Array{}, fmt.Errorf("jef: Fortran-ordered arrays are not supported")
	}

	shape, count, err := parseShape(meta)
	if err != nil {
		return Array{}, err
	}

	data, err := readTyped(r, descr[1], count)
	if err != nil {
		return Array{}, err
	}
	return Array{Shape: shape, Data: data}, nil
}

func parseShape(meta string) ([]int, int, error) {
	m := shapeRe.FindStringSubmatch(meta)
	if m == nil {
		return nil, 0, fmt.Errorf("jef: npy header has no shape: %q", meta)
	}
	var shape []int
	count := 1
	for _, part := range strings.Split(m[1], ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		dim, err := strconv.Atoi(part)
		if err != nil {
			return nil, 0, fmt.Errorf("jef: bad npy shape %q: %w", m[1], err)
		}
		shape = append(shape, dim)
		count *= dim
	}
	if len(shape) == 0 { // 0-d array, e.g. a saved scalar
		shape = []int{1}
	}
	return shape, count, nil
}

func readTyped(r io.Reader, descr string, count int) ([]float64, error) {
	out := make([]float64, count)
	switch descr {
	case "<f4", "|f4", "=f4":
		buf := make([]byte, 4*count)
		if _, err := io.ReadFull(r, buf); err != nil {
			return nil, fmt.Errorf("jef: reading float32 payload: %w", err)
		}
		for i := range out {
			out[i] = float64(math.Float32frombits(binary.LittleEndian.Uint32(buf[i*4:])))
		}
	case "<f8", "|f8", "=f8":
		buf := make([]byte, 8*count)
		if _, err := io.ReadFull(r, buf); err != nil {
			return nil, fmt.Errorf("jef: reading float64 payload: %w", err)
		}
		for i := range out {
			out[i] = math.Float64frombits(binary.LittleEndian.Uint64(buf[i*8:]))
		}
	default:
		return nil, fmt.Errorf("%w: %q (JEF reads float32/float64)", errUnsupportedDtype, descr)
	}
	return out, nil
}

// ReadNPZ decodes the numeric arrays in a .npz archive, keyed by member name
// without the .npy suffix.
//
// Members whose dtype is not numeric are returned as skipped rather than
// decoded or quietly dropped. A head.npz legitimately carries a unicode `name`
// array, and failing the whole file over it would be wrong -- but so would
// dropping a member without saying so, since only the caller knows which
// arrays it actually needs.
func ReadNPZ(path string) (arrays map[string]Array, skipped []string, err error) {
	zr, err := zip.OpenReader(path)
	if err != nil {
		return nil, nil, fmt.Errorf("jef: opening %s: %w", path, err)
	}
	defer zr.Close()

	arrays = make(map[string]Array, len(zr.File))
	for _, f := range zr.File {
		name := strings.TrimSuffix(f.Name, ".npy")
		rc, openErr := f.Open()
		if openErr != nil {
			return nil, nil, fmt.Errorf("jef: opening %s in %s: %w", f.Name, path, openErr)
		}
		arr, decodeErr := ReadNPY(rc)
		rc.Close()
		if decodeErr != nil {
			if errors.Is(decodeErr, errUnsupportedDtype) {
				skipped = append(skipped, name)
				continue
			}
			return nil, nil, fmt.Errorf("jef: decoding %s in %s: %w", f.Name, path, decodeErr)
		}
		arrays[name] = arr
	}
	return arrays, skipped, nil
}
