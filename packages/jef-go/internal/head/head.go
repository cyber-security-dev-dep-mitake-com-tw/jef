package head

import (
	"fmt"
	"math"
)

// Bilinear is the trained decision head: the only part of JEF that is learned.
//
//	logit_j = cos(ctx_j @ U, q_j @ V) * scale + bias
//
// U and V are a few million parameters because the backbone is frozen, which is
// what makes CPU-only training viable in the first place.
type Bilinear struct {
	U     Array
	V     Array
	Scale float64
	Bias  float64
	Name  string
}

// LoadBilinear reads a head.npz written by jef_core.head.BilinearHead.save.
func LoadBilinear(path string) (*Bilinear, error) {
	arrays, skipped, err := ReadNPZ(path)
	if err != nil {
		return nil, err
	}
	// `name` is stored as a unicode array and is not needed here; the numeric
	// members this head requires are validated immediately below.
	_ = skipped
	u, ok := arrays["u"]
	if !ok {
		return nil, fmt.Errorf("jef: %s has no 'u' array", path)
	}
	v, ok := arrays["v"]
	if !ok {
		return nil, fmt.Errorf("jef: %s has no 'v' array", path)
	}
	if u.Rows() != v.Rows() || u.Cols() != v.Cols() {
		return nil, fmt.Errorf("jef: U %v and V %v must share a shape", u.Shape, v.Shape)
	}

	h := &Bilinear{U: u, V: v, Scale: 1, Bias: 0, Name: "bilinear"}
	if s, ok := arrays["scale"]; ok {
		if val, err := s.Scalar(); err == nil {
			h.Scale = val
		}
	}
	if b, ok := arrays["bias"]; ok {
		if val, err := b.Scalar(); err == nil {
			h.Bias = val
		}
	}
	return h, nil
}

// Dim is the backbone hidden size this head expects.
func (h *Bilinear) Dim() int { return h.U.Rows() }

// Rank is the low-rank factor width.
func (h *Bilinear) Rank() int { return h.U.Cols() }

// Logits scores every option against one shared state encoding.
//
// ctx and queries are both (nOptions, dim): ctx comes from AttentionPool, which
// takes no trainable parameters, so the Python trainer can precompute it and
// this function is the whole of inference once the backbone has run.
func (h *Bilinear) Logits(ctx, queries [][]float64) ([]float64, error) {
	if len(ctx) != len(queries) {
		return nil, fmt.Errorf("jef: %d context vectors for %d queries", len(ctx), len(queries))
	}
	out := make([]float64, len(ctx))
	for i := range ctx {
		if len(ctx[i]) != h.Dim() {
			return nil, fmt.Errorf(
				"jef: head expects dim %d, got %d -- head and backbone do not match",
				h.Dim(), len(ctx[i]),
			)
		}
		a := normalize(project(ctx[i], h.U))
		b := normalize(project(queries[i], h.V))
		var dot float64
		for r := range a {
			dot += a[r] * b[r]
		}
		out[i] = dot*h.Scale + h.Bias
	}
	return out, nil
}

// ZeroShot is the untrained baseline: cosine similarity between the pooled
// context and the option query. Shipped so a server without a trained head
// still answers, loudly flagged as uncalibrated.
type ZeroShot struct{ Scale float64 }

// Logits scores options by cosine similarity.
func (z ZeroShot) Logits(ctx, queries [][]float64) ([]float64, error) {
	if len(ctx) != len(queries) {
		return nil, fmt.Errorf("jef: %d context vectors for %d queries", len(ctx), len(queries))
	}
	scale := z.Scale
	if scale == 0 {
		scale = 10.0
	}
	out := make([]float64, len(ctx))
	for i := range ctx {
		a, b := normalize(ctx[i]), normalize(queries[i])
		var dot float64
		for j := range a {
			dot += a[j] * b[j]
		}
		out[i] = dot * scale
	}
	return out, nil
}

// AttentionPool pools the shared state encoding once per option query.
//
// This function takes no trainable parameters, which is the property the whole
// training pipeline rests on: the Python side caches its output for the corpus
// and never runs the backbone again. Changing it invalidates every cached
// feature and every trained head, in both languages.
func AttentionPool(hidden [][]float64, mask []float64, queries [][]float64) ([][]float64, error) {
	if len(hidden) == 0 {
		return nil, fmt.Errorf("jef: empty state encoding")
	}
	if len(mask) != len(hidden) {
		return nil, fmt.Errorf("jef: mask has %d entries for %d tokens", len(mask), len(hidden))
	}
	dim := len(hidden[0])
	inverseSqrtDim := 1.0 / math.Sqrt(float64(dim))

	out := make([][]float64, len(queries))
	scores := make([]float64, len(hidden))
	for qi, q := range queries {
		if len(q) != dim {
			return nil, fmt.Errorf("jef: query dim %d does not match state dim %d", len(q), dim)
		}

		maxScore := math.Inf(-1)
		for t, tok := range hidden {
			if mask[t] <= 0 {
				scores[t] = math.Inf(-1)
				continue
			}
			var dot float64
			for d := range q {
				dot += q[d] * tok[d]
			}
			scores[t] = dot * inverseSqrtDim
			if scores[t] > maxScore {
				maxScore = scores[t]
			}
		}

		var total float64
		weights := make([]float64, len(hidden))
		for t := range scores {
			if math.IsInf(scores[t], -1) {
				continue
			}
			weights[t] = math.Exp(scores[t] - maxScore)
			total += weights[t]
		}
		if total < 1e-12 {
			total = 1e-12
		}

		pooled := make([]float64, dim)
		for t, w := range weights {
			if w == 0 {
				continue
			}
			w /= total
			for d := range pooled {
				pooled[d] += w * hidden[t][d]
			}
		}
		out[qi] = pooled
	}
	return out, nil
}

func project(v []float64, m Array) []float64 {
	cols := m.Cols()
	out := make([]float64, cols)
	for r, x := range v {
		if x == 0 {
			continue
		}
		base := r * cols
		for c := 0; c < cols; c++ {
			out[c] += x * m.Data[base+c]
		}
	}
	return out
}

func normalize(v []float64) []float64 {
	var sum float64
	for _, x := range v {
		sum += x * x
	}
	norm := math.Sqrt(sum)
	if norm < 1e-9 {
		norm = 1e-9
	}
	out := make([]float64, len(v))
	for i, x := range v {
		out[i] = x / norm
	}
	return out
}
