// Package engine evaluates typed questions against one shared state encoding.
//
// The load-bearing property (D3): Prepare runs the backbone once, and every
// question attends over that same encoding. Adding a question costs a short
// query encode plus a tiny matmul, not another pass over the state. The Go
// server counts state encodes and exports the number, so the guarantee is
// observable in production rather than merely intended.
package engine

import (
	"fmt"
	"strings"
	"sync/atomic"

	"github.com/cyber-security-dev-dep-mitake-com-tw/jef/jef-go/internal/backbone"
	"github.com/cyber-security-dev-dep-mitake-com-tw/jef/jef-go/internal/calib"
	"github.com/cyber-security-dev-dep-mitake-com-tw/jef/jef-go/internal/contract"
	"github.com/cyber-security-dev-dep-mitake-com-tw/jef/jef-go/internal/head"
	"github.com/cyber-security-dev-dep-mitake-com-tw/jef/jef-go/internal/mathx"
)

// Head scores options against a pooled context.
type Head interface {
	Logits(ctx, queries [][]float64) ([]float64, error)
}

// Engine ties a frozen backbone, a head and a calibrator together.
type Engine struct {
	Backbone   backbone.Backbone
	Head       Head
	Calibrator *calib.Calibrator
	ModelName  string
	HeadName   string

	// stateEncodes is a process counter. state_encodes / requests must stay at
	// 1.0; a rising ratio means shared-state encoding has regressed into
	// per-question encoding.
	stateEncodes atomic.Int64
}

// New assembles an engine, defaulting the model name from its parts.
func New(bb backbone.Backbone, h Head, c *calib.Calibrator, modelName, headName string) *Engine {
	if c == nil {
		c = calib.New()
	}
	if headName == "" {
		headName = "zeroshot"
	}
	if modelName == "" {
		modelName = fmt.Sprintf("jef/%s+%s", bb.Name(), headName)
	}
	return &Engine{Backbone: bb, Head: h, Calibrator: c, ModelName: modelName, HeadName: headName}
}

// StateEncodes reports how many full state encodes this process has performed.
func (e *Engine) StateEncodes() int64 { return e.stateEncodes.Load() }

// SharedState is one state, encoded once, reusable by any number of questions.
type SharedState struct {
	Text     string
	Encoding backbone.Encoding
}

// Prepare encodes the state. The only expensive call in the pipeline.
func (e *Engine) Prepare(text string) (*SharedState, error) {
	enc, err := e.Backbone.EncodeState(text)
	if err != nil {
		return nil, err
	}
	e.stateEncodes.Add(1)
	return &SharedState{Text: text, Encoding: enc}, nil
}

// Result carries one answer plus the values scene gates threshold on.
type Result struct {
	Question      contract.Normalized
	Probabilities []float64
	Confidence    float64
	Score         float64
	PCorrect      float64
	HasPCorrect   bool
	PredictionSet []int
}

// Answer evaluates one question against an already-encoded state.
//
// Nothing here reads any other question's result: questions are independent by
// construction, which is what makes "no context rot" a property rather than a
// hope.
func (e *Engine) Answer(shared *SharedState, q contract.Normalized, alpha float64) (Result, error) {
	queries, err := e.Backbone.EncodeQueries(BuildQueryTexts(q))
	if err != nil {
		return Result{}, err
	}
	pooled, err := head.AttentionPool(shared.Encoding.Hidden, shared.Encoding.Mask, queries)
	if err != nil {
		return Result{}, err
	}
	logits, err := e.Head.Logits(pooled, queries)
	if err != nil {
		return Result{}, err
	}

	probs, err := e.Calibrator.Apply(logits, q.Kind, q.NOptions())
	if err != nil {
		return Result{}, err
	}
	confidence, err := mathx.Confidence(probs)
	if err != nil {
		return Result{}, err
	}

	score := 0.0
	if q.Kind == contract.KindScore {
		if score, err = mathx.ScoreExpectation(probs); err != nil {
			return Result{}, err
		}
	}

	pCorrect, hasPCorrect := e.Calibrator.PCorrect(confidence, q.Kind, q.NOptions())

	return Result{
		Question:      q,
		Probabilities: probs,
		Confidence:    confidence,
		Score:         score,
		PCorrect:      pCorrect,
		HasPCorrect:   hasPCorrect,
		PredictionSet: e.Calibrator.PredictionSet(probs, q.Kind, q.NOptions(), alpha),
	}, nil
}

// Evaluate answers every question against one encoding of the state.
func (e *Engine) Evaluate(
	stateText string,
	questions map[string]contract.Question,
	questionOrder []string,
	alpha float64,
) (contract.Response, error) {
	shared, err := e.Prepare(stateText)
	if err != nil {
		return contract.Response{}, err
	}

	var answers contract.Answers
	inputTokens := shared.Encoding.NTokens

	// Answer in the order asked. Falling back to sorted keys only matters for a
	// caller that somehow supplied no order; the Python server answers in
	// request order, and the two must not differ.
	order := questionOrder
	if len(order) != len(questions) {
		order = contract.SortedKeys(questions)
	}

	for _, id := range order {
		normalized, err := contract.Normalize(id, questions[id])
		if err != nil {
			return contract.Response{}, err
		}
		result, err := e.Answer(shared, normalized, alpha)
		if err != nil {
			return contract.Response{}, err
		}
		answers.Set(
			id,
			contract.BuildAnswer(normalized, result.Probabilities, result.Confidence, result.Score),
		)
		for _, text := range BuildQueryTexts(normalized) {
			inputTokens += e.Backbone.CountTokens(text)
		}
	}

	return contract.Response{
		Model:   e.ModelName,
		Answers: answers,
		Usage: contract.Usage{
			InputTokens: inputTokens,
			// A System One model generates nothing. This is structural.
			OutputTokens: 0,
			TotalTokens:  inputTokens,
		},
		Warnings: e.Warnings(),
	}, nil
}

// Warnings reports anything a caller should not have to discover from results.
func (e *Engine) Warnings() []string {
	var out []string
	if e.Backbone.Name() == "hashing" {
		out = append(out, "backbone='hashing' produces deterministic but semantically meaningless vectors; for testing only")
	}
	if !e.Calibrator.IsFitted() {
		out = append(out, "calibrator is unfitted: probabilities are uncalibrated and confidence reflects distribution concentration only")
	}
	return out
}

// kindHint mirrors jef_core.head._KIND_HINT. Query text must match the Python
// side exactly or a head trained there scores different vectors here.
var kindHint = map[string]string{
	contract.KindChoice: "選項",
	contract.KindScore:  "等級",
	contract.KindNoul:   "判斷",
}

// BuildQueryTexts renders one query string per option, identically to
// jef_core.head.build_query_texts.
func BuildQueryTexts(q contract.Normalized) []string {
	hint := kindHint[q.Kind]
	out := make([]string, len(q.OptionKeys))
	for i, key := range q.OptionKeys {
		parts := []string{q.Instructions, hint + ": " + key}
		if i < len(q.OptionLabels) && q.OptionLabels[i] != "" {
			parts = append(parts, q.OptionLabels[i])
		}
		out[i] = strings.Join(parts, " | ")
	}
	return out
}
