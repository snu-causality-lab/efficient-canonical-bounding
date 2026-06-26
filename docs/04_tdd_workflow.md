# TDD workflow for `canonred/`

Every module after `graph.py` is written **test-first**. Tests are derived from
`docs/01_concepts.md` (the spec), not from the implementation.

## Red → Green → Refactor rhythm

For each module `canonred/foo.py`:

1. **Red**: create `tests/test_foo.py` with tests covering the spec. Run:
   ```bash
   pytest tests/test_foo.py
   ```
   Expected: `ModuleNotFoundError: No module named 'canonred.foo'` (or similar).
   The red state is captured in a commit message `tdd(foo): red — tests scaffolded`.

2. **Green**: implement `canonred/foo.py` to the minimum that makes all tests pass.
   Run:
   ```bash
   pytest tests/test_foo.py
   ```
   Expected: all pass. Commit: `tdd(foo): green — tests pass`.

3. **Refactor**: clean up duplication, improve names, tighten type hints.
   Tests must stay green throughout. Commit: `tdd(foo): refactor — ...`.

4. **Full-suite check**: `pytest` should be green across all modules. If a
   downstream module's test fails due to an API change here, update the
   downstream test together with this refactor step.

## Rules

- **Never write implementation code without a failing test demanding it.** If you
  find yourself adding a method, stop and write the test first.
- **One concept per test function.** Prefer `test_nde_c_component_xy` over
  bundled `test_nde`. Small tests → clear failure messages → fast diagnosis.
- **Tests read top-to-bottom as spec.** Each test name + docstring should be
  understandable on its own; a reader should be able to infer the module's API
  from the test file alone.
- **Spec quotes in test docstrings.** When a number or property comes from the
  1086 paper, ZTB22, or YB25, cite the source (e.g., `"1086 §4.1 NDE: 2 * 4 * 16 = 128"`).
- **Randomised property tests use seeded RNG.** Determinism is non-negotiable;
  every flaky test is a debugging tax.
- **Parametrise over the canonical example set.** All named graphs in
  `canonred.examples` are fixtures; sweep them with `@pytest.mark.parametrize`.
- **Slow tests marked `@pytest.mark.slow`.** The default CI run is fast (<1s);
  full enumeration tests that exceed ~0.5s get marked slow.

## Running the suite

- Fast loop during development: `pytest -x -q tests/test_foo.py`
- Entire suite: `pytest`
- With coverage: `pytest --cov=canonred --cov-report=term-missing`

## Module status (TDD baseline)

162 tests passing, 1 xfailed (PNS cardinality discrepancy with paper — see `02_ambiguities.md`).

| Module | Tests | Impl | Notes |
|---|---|---|---|
| `types.py` | n/a | ✅ | type aliases; no tests needed |
| `graph.py` | ✅ 37 + 1xfail | ✅ | written concurrently pre-TDD |
| `worlds.py` | ✅ 19 | ✅ | strict TDD; one-shot Green |
| `counterfactual.py` | ✅ 27 | ✅ | strict TDD; 1 Refactor cycle (CFQuery gained `graph` field + ADMG made hashable) |
| `expansion.py` | ✅ 15 | ✅ | edge-split graph + S_V(Q); one-shot Green |
| `canonical.py` | ✅ 26 | ✅ | ResponseFn + eval; one-shot Green |
| `observation.py` | ✅ 12 | ✅ | MomentConstraint + factories; one-shot Green |
| `context_set.py` | ✅ 12 | ✅ | m_V + Prop 4.2 + PCH; one-shot Green |
| `signature.py` | ✅ 9 | ✅ | coarsest safe quotient (Prop 3.1); 2 test-number fixes (P_obs case & stale `class_members` ref) |
| `lp.py` | ✅ 5 | ✅ | CVXPY-backed; tolerance 1e-4 matches paper §5 convention |
| `posterior.py` | — | — | generalized Gibbs + Dirichlet |
| `viz.py` | — | — | matplotlib wrappers |
| `examples.py` | implicit | ✅ (graphs) | queries/constraints populated as needed |

## "Done" criteria per module

A module is done when:
1. All tests in `tests/test_foo.py` pass.
2. Every public function has at least one test.
3. Every spec item in `docs/01_concepts.md` that the module implements has a
   matching test (traceability).
4. `pytest` run time stays fast unless a test is explicitly marked slow.
5. Type hints on every public function, checked by `mypy` (optional, but nice).
