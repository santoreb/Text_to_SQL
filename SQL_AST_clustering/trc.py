# convert SQL -> sqlglot AST -> a normalized Tuple Relational Calculus (TRC)-style token string

from typing import List, Dict, Tuple, Optional
import re
from sqlglot import parse_one, exp

_LIT_PLACE = "LIT"
_IDENT_PLACE = "IDENT"

_lit_re = re.compile(r"('([^']|'')*'|\"([^\"]|\"\")*\"|\b\d+(\.\d+)?\b)", flags=re.IGNORECASE)


def _placeholderize_literals(s: str) -> str:
    return _lit_re.sub(f" {_LIT_PLACE} ", s)


def _collect_idents(tree: exp.Expression) -> List[str]:
    seen = []
    def visit(node: exp.Expression):
        if isinstance(node, exp.Table):
            name = getattr(node, "this", None) or getattr(node, "name", None)
            if isinstance(name, str) and name.lower() not in seen:
                seen.append(name.lower())
        if isinstance(node, (exp.Column, exp.Identifier)):
            name = getattr(node, "this", None) or getattr(node, "name", None)
            if isinstance(name, str) and name.lower() not in seen:
                seen.append(name.lower())
        for v in node.args.values():
            if isinstance(v, list):
                for c in v:
                    if isinstance(c, exp.Expression):
                        visit(c)
            elif isinstance(v, exp.Expression):
                visit(v)
    visit(tree)
    return seen


def _build_ident_map(idents: List[str]) -> Dict[str, str]:
    return {name: f"IDENT_{i+1}" for i, name in enumerate(idents)}


def _replace_idents_in_str(s: str, idmap: Dict[str, str]) -> str:
    # replace longest-first to avoid partial matches
    for raw in sorted(idmap.keys(), key=lambda x: -len(x)):
        s = re.sub(rf"\b{re.escape(raw)}\b", f" {idmap[raw]} ", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _col_to_token(col: exp.Column, tmap: Dict[str, str], tcounter: List[int], outer_maps: Optional[List[Dict[str,str]]] = None) -> str:
    col_name = getattr(col, "this", None) or getattr(col, "name", None)
    tbl = None
    try:
        # sqlglot Column may have .table property or .this might be a dotted expression
        tbl = col.table
    except Exception:
        tbl = None
    if tbl:
        # tbl can be an Identifier node or a string; extract raw name
        tbl_name = getattr(tbl, "this", None) or getattr(tbl, "name", None) or str(tbl)
        try:
            key = tbl_name.lower()
        except Exception:
            key = str(tbl_name).lower()
        # Prefer local mapping, then check outer maps (nearest outer first), else assign locally
        tv = None
        if key in tmap:
            tv = tmap[key]
        else:
            if outer_maps:
                for om in outer_maps:
                    if key in om:
                        tv = om[key]
                        break
        if tv is None:
            # assign a new tuple var in local map
            tmap[key] = f"t{tcounter[0]}"
            tv = tmap[key]
            tcounter[0] += 1
        return f"{tv}.{str(col_name).lower()}"
    
    # print(str(col_name))
    return f"{_IDENT_PLACE}.{str(col_name).lower() if col_name else 'col'}"


def _get_local_tables(node: exp.Expression) -> List[str]:
    """
    Collect tables declared in the FROM and JOIN clauses of this specific node.
    Does not recurse into nested subqueries.
    """
    tables: List[str] = []

    def add_table(n: exp.Expression):
        if isinstance(n, exp.Table):
            name = getattr(n, "this", None) or getattr(n, "name", None)
            if isinstance(name, str) and name.lower() not in tables:
                tables.append(name.lower())
            # capture alias if present for local table
            alias_node = n.args.get("alias") or getattr(n, "alias", None)
            if alias_node:
                alias_name = None
                if isinstance(alias_node, exp.Alias):
                    alias_name = getattr(alias_node, "this", None) or getattr(alias_node, "name", None)
                else:
                    alias_name = getattr(alias_node, "this", None) or getattr(alias_node, "name", None) or str(alias_node)
                if isinstance(alias_name, str) and alias_name.lower() not in tables:
                    tables.append(alias_name.lower())

    frm = node.args.get("from")
    if frm and hasattr(frm, "expressions"):
        for expr in frm.expressions:
            add_table(expr)

    joins = node.args.get("joins") or []
    for j in joins:
        # sqlglot Join may have .this pointing to the table expression
        jt = getattr(j, "this", None)
        if isinstance(jt, exp.Table):
            add_table(jt)

    return tables


def _query_to_tokens(node: exp.Expression, idmap: Dict[str, str], tmap: Dict[str, str], tcounter: List[int], outer_maps: Optional[List[Dict[str,str]]] = None) -> List[str]:
    """Generate TRC tokens for a specific SELECT/Query node (recursive)."""
    tokens: List[str] = []

    # Create a local tuple-var map and counter so this query's tables are t1,t2,...
    # This keeps tuple-var numbering local to each SELECT (resets in subqueries).
    local_tmap: Dict[str, str] = {}
    local_counter: List[int] = [1]

    # Local existential declarations (for this query node)
    local_tables = _get_local_tables(node)
    for tbl in local_tables:
        # if this table/alias exists in an outer map, reuse that tuple-var (correlated ref)
        found = None
        if outer_maps:
            for om in outer_maps:
                if tbl in om:
                    found = om[tbl]
                    break
        if found:
            tv = found
        else:
            if tbl not in local_tmap:
                local_tmap[tbl] = f"t{local_counter[0]}"
                local_counter[0] += 1
            tv = local_tmap.get(tbl, f"t?")
        tokens.extend(["EXISTS", tv, "IN", f"TABLE_{tbl.upper()}"])

    # WHERE
    where = node.args.get("where")
    if where:
        pred = where.this if hasattr(where, "this") else where
        # pass outer_maps with local_tmap at front so nested lookups see this scope first
        new_outer = [local_tmap] + (outer_maps or [])
        pred_tok = _expr_to_tokens(pred, idmap, local_tmap, local_counter, new_outer)
        if pred_tok:
            tokens.extend(["AND", pred_tok])

    # JOIN ON (local only)
    joins = node.args.get("joins") or []
    for j in joins:
        on = j.args.get("on")
        if isinstance(on, exp.Expression):
            new_outer = [local_tmap] + (outer_maps or [])
            jt = _expr_to_tokens(on, idmap, local_tmap, local_counter, new_outer)
            if jt:
                tokens.extend(["AND", jt])

    # GROUP BY / HAVING
    group = node.args.get("group")
    if group:
        tokens.append("GROUPBY")
        for g in (getattr(group, "expressions", []) or []):
            tokens.append(_expr_to_tokens(g, idmap, local_tmap, local_counter, [local_tmap] + (outer_maps or [])))
    having = node.args.get("having")
    if having:
        hv = having.this if hasattr(having, "this") else having
        hv_tok = _expr_to_tokens(hv, idmap, local_tmap, local_counter, [local_tmap] + (outer_maps or []))
        if hv_tok:
            tokens.extend(["HAVING", hv_tok])

    # PROJ
    select_expressions = node.args.get("expressions") or []
    proj_items: List[str] = []
    for e in select_expressions:
        n = e.this if isinstance(e, exp.Alias) else e
        if isinstance(n, exp.Star):
            proj_items.append("PROJ_ALL")
        elif isinstance(n, exp.Func):
            fname = getattr(n, "name", None) or n.this
            fname_tok = f"AGG_{str(fname).upper()}" if fname else "AGG_FUNC"
            proj_items.append(fname_tok)
        else:
            proj_items.append("PROJ_COL")
    if proj_items:
        tokens.append("PROJ")
        tokens.extend(proj_items)

    # ORDER BY
    order = node.args.get("order")
    if order:
        tokens.append("ORDERBY")
        for o in (getattr(order, "expressions", []) or []):
            tokens.append(_expr_to_tokens(o, idmap, local_tmap, local_counter, [local_tmap] + (outer_maps or [])))

    return tokens


def _expr_to_tokens(node: exp.Expression, idmap: Dict[str,str], tmap: Dict[str,str], tcounter: List[int], outer_maps: Optional[List[Dict[str,str]]] = None) -> str:
    # handle many common node types; fallback to sql() -> placeholderization
    if node is None:
        return ""
    if isinstance(node, exp.Column):
        return _col_to_token(node, tmap, tcounter, outer_maps)
    if isinstance(node, exp.Literal):
        return _LIT_PLACE
    # Subquery / SELECT inline handling
    if isinstance(node, (exp.Subquery, exp.Select)):
        target = node.this if isinstance(node, exp.Subquery) else node
        # when inlining a subquery, pass current tmap as the first outer map
        new_outer = [tmap] + (outer_maps or [])
        sub = _query_to_tokens(target, idmap, tmap, tcounter, new_outer)
        return f"({' '.join(sub)})"
    if isinstance(node, exp.Exists):
        targ = node.this
        if isinstance(targ, exp.Subquery) and hasattr(targ, 'this'):
            targ = targ.this
        if isinstance(targ, (exp.Select, exp.Expression)):
            new_outer = [tmap] + (outer_maps or [])
            sub = _query_to_tokens(targ, idmap, tmap, tcounter, new_outer)
            return f"(EXISTS {' '.join(sub)})"
    if isinstance(node, exp.EQ):
        L = _expr_to_tokens(node.args.get("this"), idmap, tmap, tcounter, outer_maps)
        R = _expr_to_tokens(node.args.get("expression"), idmap, tmap, tcounter, outer_maps)
        return f"({L} EQ {R})"
    if isinstance(node, exp.NEQ):
        L = _expr_to_tokens(node.args.get("this"), idmap, tmap, tcounter, outer_maps)
        R = _expr_to_tokens(node.args.get("expression"), idmap, tmap, tcounter, outer_maps)
        return f"({L} NEQ {R})"
    if isinstance(node, (exp.GT, exp.GTE, exp.LT, exp.LTE)):
        op = node.__class__.__name__.upper()
        L = _expr_to_tokens(node.args.get("this"), idmap, tmap, tcounter, outer_maps)
        R = _expr_to_tokens(node.args.get("expression"), idmap, tmap, tcounter, outer_maps)
        return f"({L} {op} {R})"
    if isinstance(node, exp.And):
        parts = []
        for k,v in node.args.items():
            if isinstance(v, exp.Expression):
                parts.append(_expr_to_tokens(v, idmap, tmap, tcounter, outer_maps))
            elif isinstance(v, list):
                for c in v:
                    if isinstance(c, exp.Expression):
                        parts.append(_expr_to_tokens(c, idmap, tmap, tcounter, outer_maps))
        return " AND ".join([p for p in parts if p])
    if isinstance(node, exp.Or):
        parts = []
        for k,v in node.args.items():
            if isinstance(v, exp.Expression):
                parts.append(_expr_to_tokens(v, idmap, tmap, tcounter, outer_maps))
            elif isinstance(v, list):
                for c in v:
                    if isinstance(c, exp.Expression):
                        parts.append(_expr_to_tokens(c, idmap, tmap, tcounter, outer_maps))
        return " OR ".join([p for p in parts if p])
    if isinstance(node, exp.Func):
        fname = getattr(node, "name", None) or node.this
        fname_tok = f"FUNC_{str(fname).upper()}" if fname else "FUNC"
        args = []
        for a in node.expressions:
            if isinstance(a, exp.Expression):
                args.append(_expr_to_tokens(a, idmap, tmap, tcounter, outer_maps))
            else:
                args.append(str(a))
        return f"{fname_tok}({', '.join(args)})"
    if isinstance(node, exp.In):
        left = _expr_to_tokens(node.args.get("this"), idmap, tmap, tcounter)
        # list-style IN (val1, val2, ...)
        right = node.args.get("expressions") or []
        if isinstance(right, list) and right:
            rights = []
            for r in right:
                rights.append(_expr_to_tokens(r, idmap, tmap, tcounter, outer_maps) if isinstance(r, exp.Expression) else _LIT_PLACE)
            return f"({left} IN ({', '.join(rights)}))"

        # IN (subquery) - try to find a subquery in different arg slots
        subq = None
        for v in node.args.values():
            if isinstance(v, (exp.Subquery, exp.Select)):
                subq = v
                break
            if isinstance(v, list):
                for c in v:
                    if isinstance(c, (exp.Subquery, exp.Select)):
                        subq = c
                        break
                if subq:
                    break

        if subq is not None:
            target = subq.this if isinstance(subq, exp.Subquery) else subq
            new_outer = [tmap] + (outer_maps or [])
            sub = _query_to_tokens(target, idmap, tmap, tcounter, new_outer)
            return f"({left} IN ({' '.join(sub)}))"
    # fallback: try node.sql() and placeholderize
    try:
        s = node.sql()
        s = _placeholderize_literals(s)
        s = _replace_idents_in_str(s, idmap)
        return s
    except Exception:
        return "EXPR"


def sql_to_trc(sql: str) -> str:
    """
    Convert SQL string to a normalized TRC-style token string.
    Not exhaustive — handles SELECT/FROM/JOIN/WHERE/AGG/GROUP BY/HAVING/CTE basics.
    """
    try:
        tree = parse_one(sql)
    except Exception:
        return "PARSE_ERROR"

    idents = _collect_idents(tree)
    idmap = _build_ident_map(idents)

    # Gather FROM / tables (including JOIN's tables)
    tables: List[str] = []
    def gather_tables(node: exp.Expression):
        if isinstance(node, exp.Table):
            name = getattr(node, "this", None) or getattr(node, "name", None)
            if isinstance(name, str) and name.lower() not in tables:
                tables.append(name.lower())
            # capture alias if present so alias-based column refs map to tuple vars
            alias_node = node.args.get("alias") or getattr(node, "alias", None)
            if alias_node:
                alias_name = None
                if isinstance(alias_node, exp.Alias):
                    alias_name = getattr(alias_node, "this", None) or getattr(alias_node, "name", None)
                else:
                    alias_name = getattr(alias_node, "this", None) or getattr(alias_node, "name", None) or str(alias_node)
                if isinstance(alias_name, str) and alias_name.lower() not in tables:
                    tables.append(alias_name.lower())
        for v in node.args.values():
            if isinstance(v, list):
                for c in v:
                    if isinstance(c, exp.Expression):
                        gather_tables(c)
            elif isinstance(v, exp.Expression):
                gather_tables(v)
    gather_tables(tree)
    # tuple var map: table -> t1, t2...  (will be populated on-demand per-query)
    tmap: Dict[str, str] = {}
    # counter for assigning tuple vars; use list to allow mutation inside helpers
    tcounter: List[int] = [1]

    tokens: List[str] = []

    # Handle CTEs (WITH)
    if tree.args.get("with"):
        with_expr = tree.args["with"]
        # list of CTE aliases
        ctes = []
        for e in with_expr.expressions:
            alias = getattr(e, "alias", None) or getattr(e, "this", None)
            if alias:
                ctes.append(str(alias))
        if ctes:
            tokens.append("WITH_CTES")
            tokens.extend([ct.upper() for ct in ctes])

    # Generate tokens for the main query body (recursively handles subqueries)
    tokens.extend(_query_to_tokens(tree, idmap, tmap, tcounter))

    # Final serialization: replace literals & identifiers in token text where useful
    token_str = " ".join(tokens).strip()
    token_str = _placeholderize_literals(token_str)
    token_str = _replace_idents_in_str(token_str, idmap)
    # normalize whitespace
    token_str = re.sub(r"\s+", " ", token_str).strip()
    if not token_str:
        return "UNHANDLED_SQL"
    return token_str


# convenience wrapper: parse -> ast (returning sqlglot AST) and trc
def sql_to_trc_with_ast(sql: str) -> Tuple[Optional[exp.Expression], str]:
    try:
        tree = parse_one(sql)
    except Exception:
        return None, "PARSE_ERROR"
    return tree, sql_to_trc(sql)


# Basic demo/main when run directly
if __name__ == "__main__":
    examples = [
        "SELECT name, age FROM person WHERE age > 30",
        "SELECT p.name FROM person p JOIN orders o ON p.id = o.person_id WHERE o.price > 100",
        "SELECT COUNT(*) FROM orders WHERE created_at >= '2020-01-01'",
        "WITH recent AS (SELECT * FROM orders WHERE created_at > '2021-01-01') SELECT r.id FROM recent r WHERE r.amount > 50",
        "SELECT name FROM users WHERE age > (SELECT AVG(age) FROM users)",
        "SELECT name FROM users WHERE id IN (SELECT user_id FROM orders WHERE amount > 100)",
        "SELECT name FROM users u WHERE EXISTS (SELECT 1 FROM orders o WHERE o.user_id = u.id)",
        "SELECT u.name FROM users u WHERE EXISTS (SELECT 1 FROM orders o WHERE o.user_id = u.id AND o.amount > 100)",
        "SELECT e.name FROM employees e WHERE e.salary > (SELECT AVG(salary) FROM salaries s WHERE s.dept_id = e.dept_id)",
        "SELECT name FROM customers WHERE id IN (SELECT customer_id FROM orders WHERE created_at > '2023-01-01')",
        "SELECT a.col FROM a WHERE a.x > (SELECT MAX(b.x) FROM b WHERE b.y IN (SELECT c.y FROM c WHERE c.flag = 1))",
        "WITH recent_orders AS (SELECT * FROM orders WHERE created_at > '2024-01-01'), big_customers AS (SELECT customer_id, SUM(amount) total FROM recent_orders GROUP BY customer_id HAVING SUM(amount) > 1000) SELECT c.name FROM customers c JOIN big_customers b ON c.id = b.customer_id",
        "SELECT u.name, (SELECT COUNT() FROM orders o WHERE o.user_id = u.id) AS order_count FROM users u",
        "SELECT p.category, COUNT() cnt FROM products p GROUP BY p.category HAVING COUNT() > (SELECT AVG(cnt2) FROM (SELECT COUNT() cnt2 FROM products WHERE status='active' GROUP BY category) t)",
        "SELECT id FROM a UNION SELECT id FROM b",
        "SELECT id, CASE WHEN score >= 90 THEN 'A' WHEN score >= 80 THEN 'B' ELSE 'C' END AS grade, COALESCE(phone,'N/A') FROM students",
        "SELECT id, SUM(amount) OVER (PARTITION BY user_id ORDER BY ts) AS running_total FROM transactions",
        "SELECT u.id, x.* FROM users u CROSS JOIN LATERAL (SELECT o.* FROM orders o WHERE o.user_id = u.id ORDER BY o.created_at DESC LIMIT 1) x",
        "SELECT p.*, c.name FROM purchases p JOIN customers c ON p.customer_id = c.id AND c.active = 1 JOIN products pr ON pr.id = p.product_id WHERE p.created_at >= '2024-01-01' AND (pr.price > 100 OR pr.on_sale = 1)"
    ]
    for q in examples:
        ast, trc = sql_to_trc_with_ast(q)
        print("SQL:", q)
        print("AST:", ast)
        print("TRC tokens:", trc)
        print("-" * 60)