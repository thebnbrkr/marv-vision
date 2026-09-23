"""Static check on a notebook BEFORE it costs a GPU run.

Colab sessions are a limited resource. Every bug this catches on a laptop is a run
you do not spend finding it. It simulates top-to-bottom execution, tracking what
each cell defines and what it uses, and reports:

  * names used before any cell defines them          (the NameError that cost run 3)
  * names used only inside a cell that never runs first
  * cells that reference a module attribute by POSITION (`named_children()[-1]`),
    which is what made run 2 hook act_fn instead of linear_fc2

It cannot check anything that needs the real model. It can check everything that
does not.

    python scripts/check_notebook.py notebooks/*.ipynb
"""
from __future__ import annotations

import ast
import builtins
import json
import sys

BUILTINS = set(dir(builtins)) | {"__name__", "__file__", "get_ipython", "display", "In", "Out"}

# names Colab/IPython magic lines bring in, and things defined by ! or % lines
IPY_EXTRA = {"_", "__", "___"}


class Scan(ast.NodeVisitor):
    """Collect names bound and names loaded at module level of one cell."""

    def __init__(self) -> None:
        self.bound: set[str] = set()
        self.used: set[str] = set()

    # --- bindings ---
    def visit_Assign(self, n):
        for t in n.targets:
            self._bind(t)
        self.visit(n.value)

    def visit_AnnAssign(self, n):
        self._bind(n.target)
        if n.value:
            self.visit(n.value)

    def visit_AugAssign(self, n):
        self._bind(n.target)
        self.visit(n.value)

    def visit_For(self, n):
        self._bind(n.target)
        self.generic_visit(n)

    def visit_withitem(self, n):
        if n.optional_vars:
            self._bind(n.optional_vars)
        self.visit(n.context_expr)

    def visit_FunctionDef(self, n):
        self.bound.add(n.name)
        # do not descend: locals and params are that function's business, and its
        # free variables resolve at call time
        for d in n.decorator_list:
            self.visit(d)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Lambda(self, n):
        # params are bound inside; free variables resolve at call time
        for d in n.args.defaults + [d for d in n.args.kw_defaults if d]:
            self.visit(d)

    def visit_ClassDef(self, n):
        self.bound.add(n.name)
        self.generic_visit(n)

    def visit_Import(self, n):
        for a in n.names:
            self.bound.add((a.asname or a.name).split(".")[0])

    def visit_ImportFrom(self, n):
        for a in n.names:
            self.bound.add(a.asname or a.name)

    def visit_ExceptHandler(self, n):
        if n.name:
            self.bound.add(n.name)
        self.generic_visit(n)

    def visit_comprehension(self, n):
        self._bind(n.target)
        self.visit(n.iter)
        for i in n.ifs:
            self.visit(i)

    def _bind(self, t):
        if isinstance(t, ast.Name):
            self.bound.add(t.id)
        elif isinstance(t, (ast.Tuple, ast.List)):
            for e in t.elts:
                self._bind(e)
        elif isinstance(t, ast.Starred):
            self._bind(t.value)
        else:  # attribute / subscript target -- a use, not a binding
            self.visit(t)

    # --- uses ---
    def visit_Name(self, n):
        if isinstance(n.ctx, ast.Load):
            self.used.add(n.id)


def strip_magics(src: str) -> str:
    out = []
    for line in src.splitlines():
        s = line.lstrip()
        if s.startswith(("!", "%", "?")):
            out.append("pass")
        else:
            out.append(line)
    return "\n".join(out)


def check(path: str) -> list[str]:
    nb = json.load(open(path))
    problems: list[str] = []
    defined: set[str] = set(BUILTINS) | IPY_EXTRA

    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = strip_magics("".join(cell["source"]))
        try:
            tree = ast.parse(src)
        except SyntaxError as e:
            problems.append(f"cell {i}: SyntaxError: {e}")
            continue

        sc = Scan()
        sc.visit(tree)

        missing = sorted(n for n in sc.used - sc.bound - defined if not n.startswith("_"))
        if missing:
            problems.append(f"cell {i}: uses undefined name(s): {', '.join(missing)}")

        # the run-2 bug: picking a submodule by position
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Subscript)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and node.value.func.attr in ("named_children", "children", "named_modules")
            ):
                problems.append(
                    f"cell {i}: indexes {node.value.func.attr}() by position -- "
                    "name the submodule explicitly (this is how run 2 hooked act_fn)"
                )

        defined |= sc.bound

    return problems


def main() -> int:
    paths = sys.argv[1:] or ["notebooks/marv_vision_qwen3vl_colab.ipynb"]
    bad = 0
    for p in paths:
        probs = check(p)
        print(f"{'FAIL' if probs else 'ok  '}  {p}")
        for q in probs:
            print(f"        {q}")
        bad += len(probs)
    if bad:
        print(f"\n{bad} problem(s). Fix before spending a GPU run.")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
