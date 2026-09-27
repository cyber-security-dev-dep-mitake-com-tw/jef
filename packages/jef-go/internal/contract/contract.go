// Package contract is the /v1/systemone wire format.
//
// It exists in Go as well as Python because the two servers must be
// indistinguishable to a client: the same request, the same JSON shape back,
// including the detail that trips people up -- TypeSafe spells the yes/no
// primitive `noul` and returns it under that key, while the Vercel AI SDK
// spells it `boolean` and returns `probability`. JEF accepts both and answers
// in whichever spelling was asked.
package contract

import (
	"encoding/json"
	"fmt"
	"sort"
)

// Question kinds, normalised. The wire accepts "noul" and "boolean"; both
// normalise to KindNoul, and the original spelling is remembered for the reply.
const (
	KindChoice = "choice"
	KindScore  = "score"
	KindNoul   = "noul"
)

// Question is one typed question off the wire.
//
// Criteria is deliberately json.RawMessage: a choice question's criteria is an
// object, a score question's is an array, and a noul question's is an object
// with reserved keys. Decoding is deferred to Normalize, which knows the type.
type Question struct {
	Type         string          `json:"type"`
	Instructions string          `json:"instructions"`
	Criteria     json.RawMessage `json:"criteria,omitempty"`
}

// Request is the POST /v1/systemone body.
type Request struct {
	State     json.RawMessage     `json:"state"`
	Questions map[string]Question `json:"questions"`
	Model     string              `json:"model,omitempty"`
}

// Normalized reduces any question to "score these N options against the state",
// which is what lets one head serve all three primitives.
type Normalized struct {
	ID           string
	Kind         string
	Instructions string
	OptionKeys   []string
	OptionLabels []string
	// Dialect is the spelling the caller used for a noul question, so the reply
	// can use the same one.
	Dialect string
}

// NOptions is the calibration bucket width.
func (n Normalized) NOptions() int { return len(n.OptionKeys) }

// Normalize validates a wire question and reduces it.
func Normalize(id string, q Question) (Normalized, error) {
	switch q.Type {
	case KindChoice:
		keys, labels, err := decodeChoiceCriteria(q.Criteria)
		if err != nil {
			return Normalized{}, fmt.Errorf("question %q: %w", id, err)
		}
		if len(keys) < 2 {
			return Normalized{}, fmt.Errorf("question %q: choice needs >= 2 options", id)
		}
		if q.Instructions == "" {
			return Normalized{}, fmt.Errorf("question %q: instructions must not be empty", id)
		}
		return Normalized{ID: id, Kind: KindChoice, Instructions: q.Instructions,
			OptionKeys: keys, OptionLabels: labels}, nil

	case KindScore:
		var levels []string
		if err := json.Unmarshal(q.Criteria, &levels); err != nil {
			return Normalized{}, fmt.Errorf("question %q: score criteria must be an array of levels", id)
		}
		if len(levels) < 2 {
			return Normalized{}, fmt.Errorf("question %q: score needs >= 2 ordered levels", id)
		}
		if hasDuplicate(levels) {
			return Normalized{}, fmt.Errorf("question %q: score levels must be distinct", id)
		}
		if q.Instructions == "" {
			return Normalized{}, fmt.Errorf("question %q: instructions must not be empty", id)
		}
		return Normalized{ID: id, Kind: KindScore, Instructions: q.Instructions,
			OptionKeys: levels, OptionLabels: append([]string(nil), levels...)}, nil

	case KindNoul, "boolean":
		if q.Instructions == "" {
			return Normalized{}, fmt.Errorf("question %q: instructions must not be empty", id)
		}
		labels := []string{"", ""}
		if len(q.Criteria) > 0 {
			var c struct {
				True  string `json:"true"`
				False string `json:"false"`
			}
			if err := json.Unmarshal(q.Criteria, &c); err != nil {
				return Normalized{}, fmt.Errorf("question %q: noul criteria must be an object with true/false", id)
			}
			labels = []string{c.False, c.True}
		}
		// Order is always (false, true) so index 1 is P(true) everywhere
		// downstream, in both languages.
		return Normalized{ID: id, Kind: KindNoul, Instructions: q.Instructions,
			OptionKeys: []string{"false", "true"}, OptionLabels: labels, Dialect: q.Type}, nil

	default:
		return Normalized{}, fmt.Errorf(
			"question %q: unknown type %q (JEF answers choice, score and noul/boolean; it does not generate prose)",
			id, q.Type)
	}
}

// decodeChoiceCriteria preserves the JSON key order.
//
// Go maps are unordered, and option order decides which index each option gets
// in the distribution. Reading the raw JSON keeps the caller's order, so the
// Go server reports probabilities in the same order the Python one does.
func decodeChoiceCriteria(raw json.RawMessage) ([]string, []string, error) {
	if len(raw) == 0 {
		return nil, nil, fmt.Errorf("choice criteria are required")
	}
	dec := json.NewDecoder(newReader(raw))
	tok, err := dec.Token()
	if err != nil || tok != json.Delim('{') {
		return nil, nil, fmt.Errorf("choice criteria must be an object")
	}

	var keys, labels []string
	for dec.More() {
		keyTok, err := dec.Token()
		if err != nil {
			return nil, nil, fmt.Errorf("choice criteria: %w", err)
		}
		key, ok := keyTok.(string)
		if !ok {
			return nil, nil, fmt.Errorf("choice criteria keys must be strings")
		}
		var value any
		if err := dec.Decode(&value); err != nil {
			return nil, nil, fmt.Errorf("choice criteria: %w", err)
		}
		label := ""
		if s, ok := value.(string); ok {
			label = s
		}
		keys = append(keys, key)
		labels = append(labels, label)
	}
	return keys, labels, nil
}

func hasDuplicate(values []string) bool {
	seen := make(map[string]struct{}, len(values))
	for _, v := range values {
		if _, dup := seen[v]; dup {
			return true
		}
		seen[v] = struct{}{}
	}
	return false
}

// Usage mirrors the Python envelope. OutputTokens is structurally zero: a
// System One model generates nothing.
type Usage struct {
	InputTokens  int `json:"inputTokens"`
	OutputTokens int `json:"outputTokens"`
	TotalTokens  int `json:"totalTokens"`
}

// Response is the POST /v1/systemone reply.
type Response struct {
	Model    string         `json:"model"`
	Answers  map[string]any `json:"answers"`
	Usage    Usage          `json:"usage"`
	Warnings []string       `json:"warnings,omitempty"`
}

// BuildAnswer projects one calibrated distribution into the answer shape the
// caller asked for.
func BuildAnswer(q Normalized, probabilities []float64, confidence, score float64) map[string]any {
	const decimals = 6

	switch q.Kind {
	case KindChoice:
		dist := make(map[string]float64, len(q.OptionKeys))
		best, bestIdx := -1.0, 0
		for i, key := range q.OptionKeys {
			dist[key] = round(probabilities[i], decimals)
			if probabilities[i] > best {
				best, bestIdx = probabilities[i], i
			}
		}
		return map[string]any{
			"type":          KindChoice,
			"choice":        q.OptionKeys[bestIdx],
			"probabilities": dist,
			"confidence":    round(confidence, decimals),
		}

	case KindScore:
		dist := make(map[string]float64, len(q.OptionKeys))
		for i, key := range q.OptionKeys {
			dist[key] = round(probabilities[i], decimals)
		}
		return map[string]any{
			"type":          KindScore,
			"score":         round(score, decimals),
			"legend":        q.OptionKeys,
			"probabilities": dist,
			"confidence":    round(confidence, decimals),
		}

	default:
		pTrue := round(probabilities[1], decimals)
		if q.Dialect == "boolean" {
			return map[string]any{
				"type":        "boolean",
				"probability": pTrue,
				"confidence":  round(confidence, decimals),
			}
		}
		return map[string]any{
			"type":       KindNoul,
			"noul":       pTrue,
			"confidence": round(confidence, decimals),
		}
	}
}

// SortedKeys is used where deterministic output matters more than input order.
func SortedKeys[V any](m map[string]V) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}
