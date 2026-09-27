package contract

import (
	"bytes"
	"encoding/json"
)

// Answers preserves the order the caller asked their questions in.
//
// Go maps are unordered and encoding/json sorts their keys, so a response built
// from map[string]any comes out alphabetised while the Python server echoes the
// request's order. Object key order means nothing in JSON, but the two servers
// emitting differently shaped bodies for the same answer is a divergence, and a
// response that reads in the order you asked is easier to diff against the
// request that produced it.
type Answers struct {
	Keys   []string
	Values []any
}

// Set appends an answer, preserving insertion order.
func (a *Answers) Set(key string, value any) {
	a.Keys = append(a.Keys, key)
	a.Values = append(a.Values, value)
}

// Len reports how many answers were produced.
func (a Answers) Len() int { return len(a.Keys) }

// MarshalJSON writes the entries in insertion order.
func (a Answers) MarshalJSON() ([]byte, error) {
	var buf bytes.Buffer
	buf.WriteByte('{')
	for i, key := range a.Keys {
		if i > 0 {
			buf.WriteByte(',')
		}
		encodedKey, err := json.Marshal(key)
		if err != nil {
			return nil, err
		}
		buf.Write(encodedKey)
		buf.WriteByte(':')
		encodedValue, err := json.Marshal(a.Values[i])
		if err != nil {
			return nil, err
		}
		buf.Write(encodedValue)
	}
	buf.WriteByte('}')
	return buf.Bytes(), nil
}

// ChoiceAnswer is a struct rather than a map so its fields serialise in this
// order, matching the Python server's model field order.
type ChoiceAnswer struct {
	Type          string       `json:"type"`
	Choice        string       `json:"choice"`
	Probabilities Distribution `json:"probabilities"`
	Confidence    float64      `json:"confidence"`
}

// ScoreAnswer carries the legend so a caller can read the score against it.
type ScoreAnswer struct {
	Type          string       `json:"type"`
	Score         float64      `json:"score"`
	Legend        []string     `json:"legend"`
	Probabilities Distribution `json:"probabilities"`
	Confidence    float64      `json:"confidence"`
}

// NoulAnswer answers in whichever dialect was asked.
//
// Exactly one of Noul and Probability is set: `noul` for TypeSafe's spelling,
// `probability` for the AI SDK's. Both are pointers with omitempty so the
// unused one is absent rather than null -- a `noul` answer carrying a null
// `probability` would be a third shape neither client expects.
type NoulAnswer struct {
	Type        string   `json:"type"`
	Noul        *float64 `json:"noul,omitempty"`
	Probability *float64 `json:"probability,omitempty"`
	Confidence  float64  `json:"confidence"`
}
