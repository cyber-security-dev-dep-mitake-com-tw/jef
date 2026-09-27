package contract

import (
	"bytes"
	"encoding/json"
	"strconv"
)

// Distribution is a probability map that marshals in option order.
//
// Go's encoding/json sorts map keys, so marshalling map[string]float64 emits
// the options alphabetically while the Python server emits them in the order
// the caller supplied. Object key order carries no meaning in JSON itself, but
// two servers behind one load balancer producing differently-shaped bodies is a
// divergence, and clients do read the first key.
//
// Option order is also the index each option occupies in the distribution, so
// preserving it keeps the JSON legible against the question that produced it.
type Distribution struct {
	Keys   []string
	Values []float64
}

// MarshalJSON writes the entries in Keys order.
func (d Distribution) MarshalJSON() ([]byte, error) {
	var buf bytes.Buffer
	buf.WriteByte('{')
	for i, key := range d.Keys {
		if i > 0 {
			buf.WriteByte(',')
		}
		encoded, err := json.Marshal(key)
		if err != nil {
			return nil, err
		}
		buf.Write(encoded)
		buf.WriteByte(':')
		// 'g' with -1 precision is Go's shortest round-trip form, which is what
		// Python's repr produces too -- so 0.5 stays "0.5" rather than "0.500000".
		buf.WriteString(strconv.FormatFloat(d.Values[i], 'g', -1, 64))
	}
	buf.WriteByte('}')
	return buf.Bytes(), nil
}

// UnmarshalJSON is provided so a Response can round-trip in tests.
func (d *Distribution) UnmarshalJSON(data []byte) error {
	dec := json.NewDecoder(bytes.NewReader(data))
	tok, err := dec.Token()
	if err != nil {
		return err
	}
	if tok != json.Delim('{') {
		return &json.UnmarshalTypeError{Value: "non-object", Type: nil}
	}
	d.Keys, d.Values = nil, nil
	for dec.More() {
		keyTok, err := dec.Token()
		if err != nil {
			return err
		}
		key, _ := keyTok.(string)
		var value float64
		if err := dec.Decode(&value); err != nil {
			return err
		}
		d.Keys = append(d.Keys, key)
		d.Values = append(d.Values, value)
	}
	return nil
}
