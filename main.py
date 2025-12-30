import json
from pathlib import Path
from copy import deepcopy
from datetime import datetime

import dash
from dash import html, dcc, Input, Output, State, callback_context
import dash_cytoscape as cyto
from dash import dash_table

cyto.load_extra_layouts()

BASE_DIR = Path("/content/drive/MyDrive/ColabNotebooks/my_vocab1")
MODEL_PATH = BASE_DIR / "model.json"
GRAPHML_OUT = BASE_DIR / "model.graphml"
GRAPHML_FOCUS_OUT = BASE_DIR / "model_focus.graphml"
BACKUP_DIR = BASE_DIR / "model_backups"

#MODEL_PATH = Path("/content/drive/MyDrive/ColabNotebooks/why_vocab1/model.json")
#GRAPHML_OUT = Path("/content/drive/MyDrive/ColabNotebooks/why_vocab1/model.graphml")
#GRAPHML_FOCUS_OUT = Path("/content/drive/MyDrive/ColabNotebooks/why_vocab1/model_focus.graphml")
#BACKUP_DIR = Path("/content/drive/MyDrive/ColabNotebooks/why_vocab1/model_backups")


DEFAULT_RELATIONS = {
    "PROVIDES": {"sub_types": ["Architecture", "Standard", "DesignPattern"], "obj_types": ["Design", "DesignElement", "Solution"], "kind": "binary"},
    "DESIGNED_WITH": {"sub_types": ["Design", "RefactoredDesign"], "obj_types": ["DesignElement"], "kind": "binary"},
    "FACES": {"sub_types": ["Design", "RefactoredDesign"], "obj_types": ["MaintainabilityProblem"], "kind": "faces"},
    "SOLVES": {"sub_types": ["DesignPattern"], "obj_types": ["MaintainabilityProblem"], "kind": "solves"},
    "REFACTORS_TO": {"sub_types": ["DesignPattern"], "obj_types": ["RefactoredDesign"], "kind": "refactors"},
    "EXISTS_IN_TO_FULFILL": {"sub_types": ["VariabilityCase"], "obj_types": ["Requirement"], "kind": "exists"},
    "OFFERS": {"sub_types": ["RefactoredDesign"], "obj_types": ["Benefit"], "kind": "binary"},
    # thesis-friendly defaults
    "CONTAINS": {"sub_types": ["Chapter", "Section", "Topic"], "obj_types": ["Section", "Subsection", "Topic", "Concept"], "kind": "binary"},
    "DISCUSSES": {"sub_types": ["Paper"], "obj_types": ["Topic", "Concept", "Problem", "Gap"], "kind": "binary"},
    "HAS_GAP": {"sub_types": ["Topic", "Problem"], "obj_types": ["Gap"], "kind": "binary"},
    "MOTIVATES": {"sub_types": ["Gap", "Problem"], "obj_types": ["ResearchQuestion"], "kind": "binary"},
    "ADDRESSED_BY": {"sub_types": ["ResearchQuestion", "Problem"], "obj_types": ["Method"], "kind": "binary"},
    "ILLUSTRATED_BY": {"sub_types": ["Chapter", "Section", "Topic", "Method", "Problem"], "obj_types": ["Figure"], "kind": "binary"},
}

MOD_TYPES = {
    "WITH": ["DesignElement"],
    "FOR": ["VariabilityCase"],
    "DUE_TO": ["Cause"],
    "OF": ["Design", "RefactoredDesign", "Topic", "Problem", "Method"],
    "FROM": ["Design"],
    "IN": ["Design", "RefactoredDesign"],

}

# -----------------------------
# Fact templates (thesis-friendly)
# -----------------------------
TEMPLATES = [
    {
        "name": "Thesis structure: Chapter CONTAINS Section",
        "pred": "CONTAINS",
        "sub_types": ["Chapter"],
        "obj_types": ["Section"],
        "note": "Use this to build your chapter outline.",
    },
    {
        "name": "Thesis structure: Section CONTAINS Topic",
        "pred": "CONTAINS",
        "sub_types": ["Section"],
        "obj_types": ["Topic"],
        "note": "Use this to connect sections to the topics they cover.",
    },
    {
        "name": "Thesis structure: Topic CONTAINS Concept",
        "pred": "CONTAINS",
        "sub_types": ["Topic"],
        "obj_types": ["Concept"],
        "note": "Use this to decompose a topic into concepts/terms.",
    },
    {
        "name": "Related work: Paper DISCUSSES Topic/Concept",
        "pred": "DISCUSSES",
        "sub_types": ["Paper"],
        "obj_types": ["Topic", "Concept", "Problem", "Gap"],
        "note": "Link a paper to what it discusses.",
    },
    {
        "name": "Motivation: Topic/Problem HAS_GAP Gap",
        "pred": "HAS_GAP",
        "sub_types": ["Topic", "Problem"],
        "obj_types": ["Gap"],
        "note": "Represent a research gap.",
    },
    {
        "name": "Motivation: Gap MOTIVATES ResearchQuestion",
        "pred": "MOTIVATES",
        "sub_types": ["Gap", "Problem"],
        "obj_types": ["ResearchQuestion"],
        "note": "Explain why an RQ exists.",
    },
    {
        "name": "Method: RQ ADDRESSED_BY Method",
        "pred": "ADDRESSED_BY",
        "sub_types": ["ResearchQuestion", "Problem"],
        "obj_types": ["Method"],
        "note": "Connect your RQ/problem to your approach.",
    },
    {
        "name": "Figure: Topic/Method ILLUSTRATED_BY Figure",
        "pred": "ILLUSTRATED_BY",
        "sub_types": ["Chapter", "Section", "Topic", "Method", "Problem"],
        "obj_types": ["Figure"],
        "note": "Attach a figure to something (overview diagram, etc.).",
    },
]

# -----------------------------
# Model IO
# -----------------------------

def load_model():
    if not MODEL_PATH.exists():
        return {"entities": [], "facts": [], "relations": deepcopy(DEFAULT_RELATIONS)}
    m = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    m.setdefault("entities", [])
    m.setdefault("facts", [])
    m.setdefault("relations", deepcopy(DEFAULT_RELATIONS))
    for k, v in DEFAULT_RELATIONS.items():
        m["relations"].setdefault(k, deepcopy(v))
    return m

def save_model_with_backup(m):
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = BACKUP_DIR / f"model_{ts}.json"
    backup_path.write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
    MODEL_PATH.write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(backup_path)

# -----------------------------
# Helpers
# -----------------------------

def ent_map(m):
    return {e["id"]: e for e in m.get("entities", [])}

def find_entity(m, eid):
    for e in m.get("entities", []):
        if e.get("id") == eid:
            return e
    return None

def ensure_entity(m, eid, etype="Unknown", label=""):
    if find_entity(m, eid) is None:
        m["entities"].append({"id": eid, "type": etype, "label": label})

def rename_entity_id(m, old_id: str, new_id: str):
    if old_id == new_id:
        return
    for e in m.get("entities") or []:
        if e.get("id") == old_id:
            e["id"] = new_id
            break
    for f in m.get("facts") or []:
        if f.get("sub") == old_id:
            f["sub"] = new_id
        if f.get("obj") == old_id:
            f["obj"] = new_id
        mods = f.get("mods") or {}
        for k, vals in mods.items():
            mods[k] = [new_id if v == old_id else v for v in (vals or [])]
        f["mods"] = mods

def lanes(m):
    return sorted({(e.get("lane") or "").strip() for e in m.get("entities", []) if (e.get("lane") or "").strip()})

def lane_node_id(lane_name: str) -> str:
    return f"LANE::{lane_name}"

def unique_types(m):
    return sorted({e.get("type","") for e in m.get("entities", []) if e.get("type","")})

def unique_preds(m):
    return sorted({f.get("pred","") for f in m.get("facts", []) if f.get("pred","")}, key=lambda x: (-len(x), x))

def candidates_by_type(m, allowed_types):
    return sorted([e["id"] for e in m.get("entities", []) if e.get("type") in allowed_types])

def relation_kind(m, pred: str) -> str:
    return (m.get("relations") or {}).get(pred, {}).get("kind", "binary")

def allowed_predicates_for_pair(m, sub_id, obj_id):
    em = ent_map(m)
    if sub_id not in em or obj_id not in em:
        return []
    st, ot = em[sub_id].get("type"), em[obj_id].get("type")
    out = []
    for pred, spec in (m.get("relations") or {}).items():
        if st in spec.get("sub_types", []) and ot in spec.get("obj_types", []):
            out.append(pred)
    return sorted(out, key=lambda x: (-len(x), x))

def summarize_mods(mods: dict) -> str:
    if not mods:
        return ""
    parts = []
    for k in ["WITH", "FOR", "IN", "OF", "FROM", "DUE_TO"]:
        vals = mods.get(k) or []
        if vals:
            parts.append(f"{k}={','.join(vals)}")
    return " ".join(parts)

def normalize_fact(f: dict) -> tuple:
    pred = f.get("pred")
    sub = f.get("sub")
    obj = f.get("obj")
    mods = f.get("mods") or {}
    mods_norm = tuple(sorted((k, tuple(sorted(v or []))) for k, v in mods.items()))
    return (pred, sub, obj, mods_norm)

def is_duplicate_fact(m, fact_dict: dict, ignore_index=None) -> bool:
    target = normalize_fact(fact_dict)
    for i, f in enumerate(m.get("facts") or []):
        if ignore_index is not None and i == ignore_index:
            continue
        if normalize_fact(f) == target:
            return True
    return False

def fact_label(m, f, show_mods=True):
    pred = f["pred"]
    mods = f.get("mods") or {}
    if not show_mods or not mods:
        return pred
    parts = []
    for k in ["WITH", "FOR", "IN", "OF", "FROM", "DUE_TO"]:
        if k in mods and mods[k]:
            parts.append(f"{k}={','.join(mods[k])}")
    return pred + "\\n" + " ".join(parts) if parts else pred

def compute_focus_ids(m, focus_id: str, hops: int = 1, direction: str = "both", include_mods: bool = True):
    if not focus_id:
        return set()
    ids = {focus_id}
    frontier = {focus_id}
    hops = max(1, int(hops or 1))
    direction = (direction or "both").lower()

    for _ in range(hops):
        new = set()
        for f in (m.get("facts") or []):
            sub = f.get("sub")
            obj = f.get("obj")
            mods = f.get("mods") or {}
            mod_targets = []
            if include_mods:
                for vals in mods.values():
                    mod_targets.extend(vals or [])

            if direction == "out":
                if sub in frontier:
                    if obj: new.add(obj)
                    if include_mods: new.update(mod_targets)
                if include_mods and any(t in frontier for t in mod_targets):
                    if sub: new.add(sub)
                    if obj: new.add(obj)

            elif direction == "in":
                if obj in frontier:
                    if sub: new.add(sub)
                    if include_mods: new.update(mod_targets)
                if include_mods and any(t in frontier for t in mod_targets):
                    if sub: new.add(sub)
                    if obj: new.add(obj)

            else:  # both
                if sub in frontier and obj: new.add(obj)
                if obj in frontier and sub: new.add(sub)
                if include_mods and (sub in frontier or obj in frontier):
                    new.update(mod_targets)
                if include_mods and any(t in frontier for t in mod_targets):
                    if sub: new.add(sub)
                    if obj: new.add(obj)

        new -= ids
        ids |= new
        frontier = new
        if not frontier:
            break

    return ids

def compute_kept_ids_and_facts(m, filters: dict):
    """
    Returns:
      kept_ids: set of entity ids to render
      kept_fact_items: list of dicts: {"gidx": <index in m["facts"]>, "fact": <fact dict>}
    This is important so graph edges can carry the GLOBAL fact index for reliable delete/update.
    """
    type_filter = set(filters.get("type_filter") or [])
    pred_filter = set(filters.get("pred_filter") or [])
    search = (filters.get("search") or "").strip().lower()

    focus_node = filters.get("focus_node")
    focus_hops = int(filters.get("focus_hops") or 1)
    focus_dir = filters.get("focus_dir") or "both"
    focus_include_mods = bool(filters.get("focus_include_mods", True))

    focus_ids = set()
    if focus_node:
        focus_ids = compute_focus_ids(m, focus_node, hops=focus_hops, direction=focus_dir, include_mods=focus_include_mods)

    def match_entity(e):
        if type_filter and e.get("type") not in type_filter:
            return False
        if search:
            blob = f"{e.get('id','')} {e.get('type','')} {e.get('label','')}".lower()
            if search not in blob:
                return False
        return True

    base_entities = [e for e in m.get("entities", []) if match_entity(e)]
    base_ids = {e["id"] for e in base_entities}

    if focus_node:
        kept_ids = (base_ids & focus_ids) | {focus_node}
    else:
        kept_ids = base_ids

    def match_fact(f):
        if f.get("sub") not in kept_ids or f.get("obj") not in kept_ids:
            return False
        if pred_filter and f.get("pred") not in pred_filter:
            return False
        return True

    kept_fact_items = []
    for gidx, f in enumerate(m.get("facts", [])):
        if match_fact(f):
            kept_fact_items.append({"gidx": gidx, "fact": f})

    return kept_ids, kept_fact_items

def to_cytoscape_elements(m, filters):
    kept_ids, kept_fact_items = compute_kept_ids_and_facts(m, filters)

    use_lanes = bool(filters.get("use_lanes"))
    show_mod_edges = bool(filters.get("show_mod_edges", False))

    elements = []
    if use_lanes:
        for ln in lanes(m):
            elements.append({"data": {"id": lane_node_id(ln), "label": ln, "is_lane": "1"}})

    for e in m.get("entities", []):
        if e["id"] not in kept_ids:
            continue
        nid = e["id"]
        label_mode = filters.get("node_label_mode", "id")
        if label_mode == "label":
            nlab = e.get("label") or nid
        elif label_mode == "id_label":
            nlab = f"{nid} — {e.get('label') or ''}".strip(" —")
        else:
            nlab = nid

        node = {"data": {"id": nid, "label": nlab, "type": e.get("type", "")}}
        pos = e.get("pos")
        if isinstance(pos, dict) and "x" in pos and "y" in pos:
            node["position"] = {"x": pos["x"], "y": pos["y"]}

        if use_lanes:
            ln = (e.get("lane") or "").strip()
            if ln:
                node["data"]["parent"] = lane_node_id(ln)

        elements.append(node)

    # Fact edges (IMPORTANT: use GLOBAL index so "delete selected edge" always targets the correct fact)
    for item in kept_fact_items:
        gidx = item["gidx"]
        f = item["fact"]
        elements.append({
            "data": {
                "id": f"factg_{gidx}",
                "source": f["sub"],
                "target": f["obj"],
                "label": fact_label(m, f, show_mods=bool(filters.get("show_edge_mods"))),
                "pred": f["pred"],
                "gidx": gidx,
            }
        })

    # Optional: modifier edges
    if show_mod_edges:
        mod_edge_counter = 0
        for item in kept_fact_items:
            f = item["fact"]
            sub = f.get("sub")
            obj = f.get("obj")
            mods = f.get("mods") or {}
            for mod, targets in mods.items():
                for t in (targets or []):
                    if t not in kept_ids:
                        continue
                    mod_edge_counter += 1
                    if mod == "DUE_TO":
                        src, tgt = obj, t
                    elif mod in ("OF", "IN"):
                        src, tgt = obj, t
                    else:
                        src, tgt = sub, t
                    if not src or not tgt:
                        continue
                    elements.append({
                        "data": {
                            "id": f"mod_{mod_edge_counter}",
                            "source": src,
                            "target": tgt,
                            "label": mod,
                            "pred": f"MOD_{mod}",
                        }
                    })

    return elements

def build_stylesheet(show_node_labels=True, show_edge_labels=True):
    node_label = "data(label)" if show_node_labels else ""
    edge_label = "data(label)" if show_edge_labels else ""
    return [
        {"selector": 'node[is_lane = "1"]',
         "style": {"label": "data(label)" if show_node_labels else "",
                   "text-valign": "top", "text-halign": "center",
                   "background-opacity": 0.06,
                   "border-width": 2, "border-style": "dashed",
                   "padding": "18px", "shape": "roundrectangle"}},
        {"selector": "node",
         "style": {"label": node_label, "text-wrap": "wrap", "text-max-width": 180,
                   "font-size": "10px", "width": "40px", "height": "40px",
                   "shape": "roundrectangle"}},
        {"selector": "edge",
         "style": {"label": edge_label, "font-size": "9px",
                   "curve-style": "taxi", "taxi-direction": "rightward",
                   "target-arrow-shape": "triangle", "arrow-scale": 0.9}},
        {"selector": 'edge[pred ^= "MOD_"]',
         "style": {"line-style": "dashed", "target-arrow-shape": "triangle"}}
    ]

# -----------------------------
# yEd GraphML
# -----------------------------
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom

def pretty_xml(elem):
    return minidom.parseString(tostring(elem, encoding="utf-8")).toprettyxml(indent="  ")

def export_graphml_yed_simple(m, out_path: Path):
    gml = Element("graphml", {
        "xmlns": "http://graphml.graphdrawing.org/xmlns",
        "xmlns:y": "http://www.yworks.com/xml/graphml",
        "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
        "xsi:schemaLocation": "http://graphml.graphdrawing.org/xmlns "
                              "http://www.yworks.com/xml/schema/graphml/1.1/ygraphml.xsd",
    })
    SubElement(gml, "key", id="d0", **{"for": "node", "yfiles.type": "nodegraphics"})
    SubElement(gml, "key", id="d1", **{"for": "edge", "yfiles.type": "edgegraphics"})
    graph = SubElement(gml, "graph", id="G", edgedefault="directed")

    def add_node(node_id, label):
        n = SubElement(graph, "node", id=node_id)
        d = SubElement(n, "data", key="d0")
        sn = SubElement(d, "{http://www.yworks.com/xml/graphml}ShapeNode")
        SubElement(sn, "{http://www.yworks.com/xml/graphml}Geometry", height="32.0", width="250.0", x="0.0", y="0.0")
        SubElement(sn, "{http://www.yworks.com/xml/graphml}Fill", color="#FFCC00", transparent="false")
        SubElement(sn, "{http://www.yworks.com/xml/graphml}BorderStyle", color="#000000", type="line", width="1.0")
        nl = SubElement(sn, "{http://www.yworks.com/xml/graphml}NodeLabel")
        nl.text = label
        SubElement(sn, "{http://www.yworks.com/xml/graphml}Shape", type="roundrectangle")

    def add_edge(eid, src, tgt, label):
        e = SubElement(graph, "edge", id=eid, source=src, target=tgt)
        d = SubElement(e, "data", key="d1")
        pe = SubElement(d, "{http://www.yworks.com/xml/graphml}PolyLineEdge")
        SubElement(pe, "{http://www.yworks.com/xml/graphml}LineStyle", color="#000000", type="line", width="1.0")
        SubElement(pe, "{http://www.yworks.com/xml/graphml}Arrows", source="none", target="standard")
        el = SubElement(pe, "{http://www.yworks.com/xml/graphml}EdgeLabel")
        el.text = label
        SubElement(pe, "{http://www.yworks.com/xml/graphml}BendStyle", smoothed="false")

    for e in m.get("entities", []):
        add_node(e["id"], f"{e['id']} : {e.get('label','')}\\n[{e.get('type','')}]")

    for i, f in enumerate(m.get("facts", []), start=1):
        add_edge(f"e{i}", f["sub"], f["obj"], fact_label(m, f, show_mods=True))

    out_path.write_text(pretty_xml(gml), encoding="utf-8")

def extract_submodel(m, filters):
    kept_ids, kept_fact_items = compute_kept_ids_and_facts(m, filters)
    sub_entities = [e for e in m.get("entities", []) if e["id"] in kept_ids]
    sub_facts = [item["fact"] for item in kept_fact_items]
    return {"entities": sub_entities, "facts": sub_facts, "relations": deepcopy(m.get("relations") or {})}


# -----------------------------
# Validation / Model health
# -----------------------------
def find_orphans(m):
    used = set()
    for f in (m.get("facts") or []):
        used.add(f.get("sub"))
        used.add(f.get("obj"))
        mods = f.get("mods") or {}
        for vals in mods.values():
            used.update(vals or [])
    return sorted([e["id"] for e in (m.get("entities") or []) if e.get("id") and e["id"] not in used])

def find_duplicate_fact_indices(m):
    seen = {}
    dups = []
    for i, f in enumerate(m.get("facts") or []):
        key = normalize_fact(f)
        if key in seen:
            dups.append((seen[key], i))
        else:
            seen[key] = i
    return dups

def find_broken_facts(m):
    em = ent_map(m)
    broken = []
    rels = m.get("relations") or {}
    for i, f in enumerate(m.get("facts") or []):
        pred = f.get("pred")
        sub = f.get("sub")
        obj = f.get("obj")
        if pred not in rels:
            broken.append((i, "Unknown predicate", f))
            continue
        if sub not in em:
            broken.append((i, "Missing subject entity", f))
            continue
        if obj not in em:
            broken.append((i, "Missing object entity", f))
            continue
        st = em[sub].get("type")
        ot = em[obj].get("type")
        spec = rels[pred]
        if st not in (spec.get("sub_types") or []) or ot not in (spec.get("obj_types") or []):
            broken.append((i, "Type mismatch vs relation definition", f))
            continue
        mods = f.get("mods") or {}
        for mk, vals in mods.items():
            for vid in (vals or []):
                if vid not in em:
                    broken.append((i, f"Missing modifier target for {mk}", f))
                    break
    return broken

def render_health(m):
    orphans = find_orphans(m)
    dups = find_duplicate_fact_indices(m)
    broken = find_broken_facts(m)

    lines = []
    lines.append(f"Entities: {len(m.get('entities') or [])}   Facts: {len(m.get('facts') or [])}   Relations: {len(m.get('relations') or {})}")
    lines.append("")

    lines.append(f"Orphan entities (not referenced): {len(orphans)}")
    for eid in orphans[:20]:
        lines.append(f"  - {eid}")
    if len(orphans) > 20:
        lines.append(f"  ... (+{len(orphans)-20} more)")
    lines.append("")

    lines.append(f"Duplicate facts: {len(dups)}")
    for a, b in dups[:20]:
        lines.append(f"  - facts #{a} and #{b} are identical")
    if len(dups) > 20:
        lines.append(f"  ... (+{len(dups)-20} more)")
    lines.append("")

    lines.append(f"Broken facts: {len(broken)}")
    for i, reason, f in broken[:12]:
        lines.append(f"  - #{i}: {reason}: {f.get('sub')} {f.get('pred')} {f.get('obj')}")
    if len(broken) > 12:
        lines.append(f"  ... (+{len(broken)-12} more)")
    return "\n".join(lines)

# -----------------------------
# App
# -----------------------------
m0 = load_model()
app = dash.Dash(__name__)
app.title = "Why-Vocab Graph Editor (v4)"

def opts(values):
    return [{"label": v, "value": v} for v in values]

def entity_options(m):
    out = []
    for e in m.get("entities", []):
        eid = e.get("id","")
        typ = e.get("type","") or "?"
        lab = (e.get("label") or "").strip()
        if len(lab) > 42:
            lab = lab[:39] + "..."
        label = f"{eid} [{typ}] — {lab}" if lab else f"{eid} [{typ}]"
        out.append({"label": label, "value": eid})
    return sorted(out, key=lambda x: x["value"])

def facts_table_rows(m):
    rows = []
    for i, f in enumerate(m.get("facts", [])):
        rows.append({"idx": i, "sub": f.get("sub",""), "pred": f.get("pred",""), "obj": f.get("obj",""), "mods": summarize_mods(f.get("mods") or {})})
    return rows

# ---- Left panels ----
view_panel = html.Div(
    id="panel_view",
    children=[
        html.H4("Filters"),
        dcc.Input(id="flt_search", placeholder="Search (id/type/label)", style={"width": "100%"}),
        dcc.Dropdown(id="flt_type", multi=True, placeholder="Filter by entity type", style={"marginTop": "6px"}),
        dcc.Dropdown(id="flt_pred", multi=True, placeholder="Filter by predicate", style={"marginTop": "6px"}),
        dcc.Checklist(
            id="flt_checks",
            options=[
                {"label": "Swimlanes (compound nodes)", "value": "use_lanes"},
                {"label": "Show edge modifiers (WITH/FOR/..)", "value": "show_edge_mods"},
                {"label": "Show node labels", "value": "show_node_labels"},
                {"label": "Show edge labels", "value": "show_edge_labels"},
            ],
            value=["show_node_labels", "show_edge_labels"],
            style={"fontSize": "13px", "marginTop":"8px"},
        ),
        dcc.Dropdown(
            id="flt_node_label_mode",
            options=[
                {"label": "Node label = ID", "value": "id"},
                {"label": "Node label = Label", "value": "label"},
                {"label": "Node label = ID — Label", "value": "id_label"},
            ],
            value="id",
            style={"marginTop": "6px"},
        ),
        html.Hr(),
        html.H4("Focus filter (show related nodes)"),
        dcc.Dropdown(id="flt_focus_node", placeholder="Focus node (e.g., a Topic, Chapter, Paper)", style={"marginTop": "6px"}),
        html.Div(style={"marginTop": "6px"}, children=[
            html.Div("Hops (distance):", style={"fontSize":"12px", "marginBottom":"4px"}),
            dcc.Slider(id="flt_focus_hops", min=1, max=3, step=1, value=1, marks={1:"1", 2:"2", 3:"3"}),
        ]),
        dcc.RadioItems(
            id="flt_focus_dir",
            options=[{"label":"Both directions", "value":"both"}, {"label":"Outgoing only", "value":"out"}, {"label":"Incoming only", "value":"in"}],
            value="both",
            style={"fontSize":"13px", "marginTop":"6px"},
        ),
        dcc.Checklist(
            id="flt_focus_opts",
            options=[
                {"label":"Include modifier nodes (WITH/FOR/OF/IN/FROM/DUE_TO)", "value":"include_mods"},
                {"label":"Draw modifier edges", "value":"show_mod_edges"},
            ],
            value=["include_mods"],
            style={"fontSize":"13px", "marginTop":"6px"},
        ),
        html.Div(style={"fontSize":"12px","opacity":0.85,"marginTop":"6px"}, children="Tip: hops=2 is excellent for thesis section diagrams."),

        html.Hr(),
        html.H4("Model health"),
        html.Div("Warnings and quick stats to keep your thesis model consistent.", style={"fontSize":"12px","opacity":0.85}),
        html.Pre(id="health", style={"whiteSpace":"pre-wrap","fontSize":"12px","maxHeight":"240px","overflowY":"auto","border":"1px solid #f0f0f0","borderRadius":"10px","padding":"8px","background":"#fafafa","marginTop":"6px"}),

    ],
)

entities_panel = html.Div(
    id="panel_entities",
    style={"display": "none"},
    children=[
        html.H4("Entities"),
        dcc.Dropdown(id="ent_select", placeholder="Select entity"),
        dcc.Input(id="ent_id", placeholder="ID (rename supported)", style={"width": "100%", "marginTop": "6px"}),
        dcc.Input(id="ent_type", placeholder="Type", style={"width": "100%", "marginTop": "6px"}),
        dcc.Input(id="ent_label", placeholder="Label", style={"width": "100%", "marginTop": "6px"}),
        html.Div("Tip: to SEE label changes in the graph, set View → Node label mode to ‘Label’ or ‘ID — Label’.", style={"fontSize":"12px","opacity":0.8,"marginTop":"6px"}),
        dcc.Input(id="ent_lane", placeholder="Lane (optional, e.g., 'RelatedWork')", style={"width": "100%", "marginTop": "6px"}),
        html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr 1fr","gap":"6px","marginTop":"8px"}, children=[
            html.Button("Add", id="btn_ent_add"),
            html.Button("Update", id="btn_ent_upd"),
            html.Button("Delete", id="btn_ent_del"),
        ]),
    ],
)

relations_panel = html.Div(
    id="panel_relations",
    style={"display":"none"},
    children=[
        html.H4("Relations"),
        dcc.Dropdown(id="rel_select", placeholder="Select relation"),
        dcc.Input(id="rel_name", placeholder="Name (e.g., OFFERS2)", style={"width":"100%","marginTop":"6px"}),
        dcc.Input(id="rel_sub_types", placeholder="Subject types (comma-separated)", style={"width":"100%","marginTop":"6px"}),
        dcc.Input(id="rel_obj_types", placeholder="Object types (comma-separated)", style={"width":"100%","marginTop":"6px"}),
        dcc.Dropdown(id="rel_kind", options=opts(["binary","faces","solves","refactors","exists"]), value="binary", style={"marginTop":"6px"}),
        html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr 1fr","gap":"6px","marginTop":"8px"}, children=[
            html.Button("Add", id="btn_rel_add"),
            html.Button("Update", id="btn_rel_upd"),
            html.Button("Delete", id="btn_rel_del"),
        ]),
    ],
)

facts_panel = html.Div(
    id="panel_facts",
    style={"display":"none"},
    children=[
        html.H4("Facts (list)"),
        html.Div("Tip: quick editing is on the right sidebar.", style={"fontSize":"12px","opacity":0.85}),
        html.Div(id="facts_list", style={"fontSize":"12px","maxHeight":"65vh","overflowY":"auto","marginTop":"8px"}),
    ],
)

# -----------------------------
# Layout
# -----------------------------
app.layout = html.Div(
    style={"display":"grid","gridTemplateColumns":"420px 1fr","gap":"12px","padding":"12px"},
    children=[
        html.Div(
            style={"border":"1px solid #ddd","borderRadius":"12px","padding":"12px"},
            children=[
                html.H3("Editor"),
                html.Div(id="status", style={"whiteSpace":"pre-wrap","fontSize":"13px"}),
                dcc.Tabs(
                    id="tabs", value="tab_view",
                    children=[
                        dcc.Tab(label="View", value="tab_view"),
                        dcc.Tab(label="Entities", value="tab_entities"),
                        dcc.Tab(label="Relations", value="tab_relations"),
                        dcc.Tab(label="Facts", value="tab_facts"),
                    ],
                ),
                html.Div(style={"marginTop":"10px"}, children=[view_panel, entities_panel, relations_panel, facts_panel]),
                html.Hr(),
                html.H4("Save / Export"),
                html.Button("Save model.json (auto-backup)", id="btn_save", style={"width":"100%"}),
                html.Button("Export yEd GraphML (full)", id="btn_export", style={"marginTop":"8px","width":"100%"}),
                html.Button("Export yEd GraphML (focused)", id="btn_export_focus", style={"marginTop":"8px","width":"100%"}),
                html.Div("Focused export uses current filters + focus settings.", style={"fontSize":"12px","opacity":0.85,"marginTop":"6px"}),
            ],
        ),
        html.Div(
            style={"border":"1px solid #ddd","borderRadius":"12px","padding":"12px"},
            children=[
                html.Div(
                    style={"display":"grid","gridTemplateColumns":"1fr 420px","gap":"10px"},
                    children=[
                        cyto.Cytoscape(
                            id="cy",
                            elements=to_cytoscape_elements(m0, filters={
                                "use_lanes": False, "node_label_mode":"id", "show_edge_mods": False,
                                "type_filter": [], "pred_filter": [], "search": "",
                                "show_node_labels": True, "show_edge_labels": True,
                                "focus_node": None, "focus_hops": 1, "focus_dir": "both",
                                "focus_include_mods": True, "show_mod_edges": False,
                            }),
                            style={"width":"100%","height":"84vh"},
                            layout={"name":"dagre","rankDir":"LR","nodeSep":50,"edgeSep":10,"rankSep":70,"padding":10},
                            stylesheet=build_stylesheet(True, True),
                        ),
                        html.Div(
                            style={"border":"1px solid #eee","borderRadius":"12px","padding":"10px","height":"84vh","overflowY":"auto"},
                            children=[
                                html.H4("Details"),
                                html.Div(id="details", style={"whiteSpace":"pre-wrap","fontSize":"13px"}),
                                html.Button("Focus here + Export diagram", id="btn_focus_export", style={"marginTop":"8px","width":"100%"}),
                                html.Div("Tip: click a node, then use this to create a thesis-ready focused GraphML quickly.", style={"fontSize":"12px","opacity":0.85}),

                                html.Hr(),
                                html.H4("Add / Edit Fact"),

                                html.Hr(),
                                html.H4("Templates (fill-in-the-blanks)"),
                                dcc.Dropdown(
                                    id="tmpl_select",
                                    options=[{"label": t["name"], "value": i} for i, t in enumerate(TEMPLATES)],
                                    placeholder="Choose a template (optional)",
                                    style={"fontSize":"12px"},
                                ),
                                html.Div(id="tmpl_note", style={"fontSize":"12px","opacity":0.85,"marginTop":"6px","whiteSpace":"pre-wrap"}),
                                html.Button("Apply template", id="btn_tmpl_apply", style={"marginTop":"6px","width":"100%"}),

                                html.Div(style={"fontSize":"12px","opacity":0.85}, children="Workflow: pick Subject/Object → choose predicate → optional modifiers → Add or Update."),

                                html.Div(style={"display":"grid","gridTemplateColumns":"1fr 130px","gap":"6px","marginTop":"6px"}, children=[
                                    dcc.Dropdown(id="fact_sub", placeholder="Subject", clearable=True, searchable=True, maxHeight=320, optionHeight=38, style={"fontSize":"12px"}),
                                    html.Button("Pick Subject", id="btn_pick_sub"),
                                ]),
                                html.Div(style={"display":"grid","gridTemplateColumns":"1fr 130px","gap":"6px","marginTop":"6px"}, children=[
                                    dcc.Dropdown(id="fact_obj", placeholder="Object", clearable=True, searchable=True, maxHeight=320, optionHeight=38, style={"fontSize":"12px"}),
                                    html.Button("Pick Object", id="btn_pick_obj"),
                                ]),
                                html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr","gap":"6px","marginTop":"6px"}, children=[
                                    html.Button("Swap", id="btn_swap"),
                                    html.Button("Clear", id="btn_clear_fact"),
                                ]),
                                dcc.Dropdown(id="fact_pred", placeholder="Predicate (auto-filtered)", style={"marginTop":"6px","fontSize":"12px"}, clearable=True, searchable=True, maxHeight=240, optionHeight=32),
                                html.Div(id="pred_hint", style={"fontSize":"12px","marginTop":"6px","opacity":0.85}),
                                html.Div(id="fact_preview", style={"whiteSpace":"pre-wrap","fontSize":"12px","marginTop":"8px","padding":"8px","border":"1px solid #f0f0f0","borderRadius":"10px","background":"#fafafa"}),

                                html.Hr(),
                                html.H4("Modifiers"),
                                html.Div(id="mods_hint", style={"fontSize":"12px","opacity":0.85}),
                                html.Div(id="wrap_with", children=[dcc.Dropdown(id="mod_with", multi=True, placeholder="WITH (DesignElement)")], style={"marginTop":"6px"}),
                                html.Div(id="wrap_for", children=[dcc.Dropdown(id="mod_for", multi=True, placeholder="FOR (VariabilityCase)")], style={"marginTop":"6px"}),
                                html.Div(id="wrap_in", children=[dcc.Dropdown(id="mod_in", multi=False, placeholder="IN (Design/RefactoredDesign)")], style={"marginTop":"6px"}),
                                html.Div(id="wrap_of", children=[dcc.Dropdown(id="mod_of", multi=False, placeholder="OF (Design/RefactoredDesign/Topic/Problem/Method)")], style={"marginTop":"6px"}),
                                html.Div(id="wrap_from", children=[dcc.Dropdown(id="mod_from", multi=False, placeholder="FROM (Design)")], style={"marginTop":"6px"}),
                                html.Div(id="wrap_due_existing", children=[dcc.Dropdown(id="mod_due_existing", multi=True, placeholder="DUE_TO (existing Cause IDs)")], style={"marginTop":"6px"}),
                                html.Div(id="wrap_due_new", children=[dcc.Input(id="mod_due_new", placeholder="Add NEW causes as text (separate by ';')", style={"width":"100%"})], style={"marginTop":"6px"}),

                                html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr 1fr","gap":"6px","marginTop":"10px"}, children=[
                                    html.Button("Add fact", id="btn_fact_add"),
                                    html.Button("Update fact", id="btn_fact_update"),
                                    html.Button("Clear edit", id="btn_fact_clear_edit"),
                                ]),
                                html.Button("Delete selected edge", id="btn_fact_delete", style={"marginTop":"6px","width":"100%"}),

                                html.Hr(),
                                html.H4("Facts (quick edit)"),
                                html.Div(style={"fontSize":"12px","opacity":0.85}, children="Click a row to load it into the editor above, then Update."),
                                dash_table.DataTable(
                                    id="facts_table",
                                    columns=[{"name":"#","id":"idx"},{"name":"Subject","id":"sub"},{"name":"Predicate","id":"pred"},{"name":"Object","id":"obj"},{"name":"Mods","id":"mods"}],
                                    data=facts_table_rows(m0),
                                    page_size=8,
                                    sort_action="native",
                                    filter_action="native",
                                    row_selectable="single",
                                    selected_rows=[],
                                    style_table={"overflowX":"auto"},
                                    style_cell={"fontSize":"11px","padding":"6px","whiteSpace":"normal","height":"auto"},
                                    style_header={"fontSize":"11px","fontWeight":"bold"},
                                ),

                                html.Hr(),
                                html.H4("Layout"),
                                dcc.RadioItems(
                                    id="rank_dir",
                                    options=[{"label":"Left → Right","value":"LR"},{"label":"Top → Bottom","value":"TB"}],
                                    value="LR",
                                    style={"fontSize":"13px"},
                                ),
                                html.Button("Re-run layout", id="btn_layout", style={"marginTop":"6px","width":"100%"}),

                                html.Hr(),
                                html.H4("Positions"),
                                html.Div("Stores position of the last tapped node into model.json as pos={x,y}.", style={"fontSize":"12px"}),
                                html.Button("Save tapped node position", id="btn_save_pos", style={"marginTop":"6px","width":"100%"}),
                            ],
                        ),
                    ],
                ),

                dcc.Store(id="store_model", data=m0),
                dcc.Store(id="store_filters", data={
                    "use_lanes": False, "node_label_mode":"id", "show_edge_mods": False,
                    "type_filter": [], "pred_filter": [], "search": "",
                    "show_node_labels": True, "show_edge_labels": True,
                    "focus_node": None, "focus_hops": 1, "focus_dir": "both",
                    "focus_include_mods": True, "show_mod_edges": False,
                }),
                dcc.Store(id="store_sel", data={"edge": None, "tapped_node": None, "tapped_pos": None}),
                dcc.Store(id="store_pick", data={"mode": None}),
                dcc.Store(id="store_fact_edit", data={"index": None}),
                dcc.Store(id="store_template", data={"active": False, "sub_types": [], "obj_types": [], "pred": None}),
                dcc.Store(id="store_status", data="Ready."),
                dcc.Store(id="store_details", data="Click a node or edge."),
            ],
        ),
    ],
)

# -----------------------------
# Tab switching
# -----------------------------
@app.callback(
    Output("panel_view", "style"),
    Output("panel_entities", "style"),
    Output("panel_relations", "style"),
    Output("panel_facts", "style"),
    Input("tabs", "value"),
)
def switch_tabs(tab):
    def show(is_show):
        return {"display":"block"} if is_show else {"display":"none"}
    return show(tab=="tab_view"), show(tab=="tab_entities"), show(tab=="tab_relations"), show(tab=="tab_facts")

@app.callback(Output("status","children"), Input("store_status","data"))
def render_status(msg): return msg or ""

@app.callback(Output("details","children"), Input("store_details","data"))
def render_details(msg): return msg or ""


@app.callback(
    Output("tmpl_note","children"),
    Input("tmpl_select","value"),
)
def show_template_note(idx):
    if idx is None:
        return "Templates are optional. They restrict Subject/Object pickers to the right types and try to keep your chosen predicate."
    try:
        t = TEMPLATES[int(idx)]
        return f"Template: {t.get('pred')}  |  Subject types: {', '.join(t.get('sub_types') or [])}  |  Object types: {', '.join(t.get('obj_types') or [])}\n{t.get('note','')}"
    except Exception:
        return "Invalid template selection."

@app.callback(
    Output("store_template","data"),
    Output("fact_pred","value", allow_duplicate=True),
    Output("fact_sub","value", allow_duplicate=True),
    Output("fact_obj","value", allow_duplicate=True),
    Output("store_status","data", allow_duplicate=True),
    Input("btn_tmpl_apply","n_clicks"),
    State("tmpl_select","value"),
    prevent_initial_call=True,
)
def apply_template(_n, idx):
    if idx is None:
        return {"active": False, "sub_types": [], "obj_types": [], "pred": None}, None, None, None, "Template cleared."
    t = TEMPLATES[int(idx)]
    data = {"active": True, "sub_types": t.get("sub_types") or [], "obj_types": t.get("obj_types") or [], "pred": t.get("pred")}
    return data, t.get("pred"), None, None, f"Applied template: {t.get('name')}"
# -----------------------------
# Filter options
# -----------------------------
@app.callback(
    Output("flt_type","options"),
    Output("flt_pred","options"),
    Input("store_model","data"),
)
def refresh_filter_options(m):
    return opts(unique_types(m)), opts(unique_preds(m))


@app.callback(
    Output("health","children"),
    Input("store_model","data"),
)
def update_health(m):
    return render_health(m)
@app.callback(
    Output("store_filters","data"),
    Output("store_status","data", allow_duplicate=True),
    Input("flt_search","value"),
    Input("flt_type","value"),
    Input("flt_pred","value"),
    Input("flt_checks","value"),
    Input("flt_node_label_mode","value"),
    Input("flt_focus_node","value"),
    Input("flt_focus_hops","value"),
    Input("flt_focus_dir","value"),
    Input("flt_focus_opts","value"),
    State("store_filters","data"),
    prevent_initial_call=True,
)
def update_filters(search, type_filter, pred_filter, checks, label_mode, focus_node, focus_hops, focus_dir, focus_opts, cur):
    cur = deepcopy(cur)
    cur["search"] = search or ""
    cur["type_filter"] = type_filter or []
    cur["pred_filter"] = pred_filter or []
    checks = set(checks or [])
    cur["use_lanes"] = "use_lanes" in checks
    cur["show_edge_mods"] = "show_edge_mods" in checks
    cur["show_node_labels"] = "show_node_labels" in checks
    cur["show_edge_labels"] = "show_edge_labels" in checks
    cur["node_label_mode"] = label_mode or "id"
    cur["focus_node"] = focus_node or None
    cur["focus_hops"] = int(focus_hops or 1)
    cur["focus_dir"] = focus_dir or "both"
    focus_opts = set(focus_opts or [])
    cur["focus_include_mods"] = "include_mods" in focus_opts
    cur["show_mod_edges"] = "show_mod_edges" in focus_opts
    return cur, "Updated filters."

@app.callback(
    Output("cy","elements"),
    Output("cy","stylesheet"),
    Input("store_model","data"),
    Input("store_filters","data"),
)
def refresh_graph(m, filters):
    return (
        to_cytoscape_elements(m, filters),
        build_stylesheet(show_node_labels=bool(filters.get("show_node_labels", True)),
                         show_edge_labels=bool(filters.get("show_edge_labels", True))),
    )

# -----------------------------
# Dropdown refresh
# -----------------------------
@app.callback(
    Output("ent_select","options"),
    Output("rel_select","options"),
    Output("flt_focus_node","options"),
    Output("fact_sub","options"),
    Output("fact_obj","options"),
    Input("store_model","data"),
    Input("store_template","data"),
)
def refresh_dropdowns(m, tmpl):
    ent_ids = sorted([e["id"] for e in m.get("entities", [])])
    rels = sorted(list((m.get("relations") or {}).keys()), key=lambda x: (-len(x), x))
    ent_opts = entity_options(m)

    tmpl = tmpl or {"active": False}
    if tmpl.get("active"):
        em = ent_map(m)
        sub_types = set(tmpl.get("sub_types") or [])
        obj_types = set(tmpl.get("obj_types") or [])
        sub_opts = [o for o in ent_opts if (em.get(o["value"], {}).get("type") in sub_types)] if sub_types else ent_opts
        obj_opts = [o for o in ent_opts if (em.get(o["value"], {}).get("type") in obj_types)] if obj_types else ent_opts
    else:
        sub_opts, obj_opts = ent_opts, ent_opts

    return opts(ent_ids), opts(rels), ent_opts, sub_opts, obj_opts

@app.callback(
    Output("ent_id","value"),
    Output("ent_type","value"),
    Output("ent_label","value"),
    Output("ent_lane","value"),
    Input("ent_select","value"),
    State("store_model","data"),
    prevent_initial_call=True,
)
def load_entity_to_form(eid, m):
    if not eid:
        return "", "", "", ""
    e = find_entity(m, eid) or {}
    return e.get("id",""), e.get("type",""), e.get("label",""), e.get("lane","")

@app.callback(
    Output("rel_name","value"),
    Output("rel_sub_types","value"),
    Output("rel_obj_types","value"),
    Output("rel_kind","value"),
    Input("rel_select","value"),
    State("store_model","data"),
    prevent_initial_call=True,
)
def load_relation_to_form(rname, m):
    if not rname:
        return "", "", "", "binary"
    spec = (m.get("relations") or {}).get(rname, {}) or {}
    return rname, ",".join(spec.get("sub_types", [])), ",".join(spec.get("obj_types", [])), spec.get("kind", "binary")

# Facts list tab
@app.callback(Output("facts_list","children"), Input("store_model","data"))
def render_facts_list(m):
    if not m.get("facts"):
        return "No facts yet."
    items = [html.Li(f"{i}: {f['sub']} {f['pred']} {f['obj']}  {summarize_mods(f.get('mods') or {})}") for i, f in enumerate(m["facts"])]
    return html.Ul(items)

# Right table refresh
@app.callback(Output("facts_table","data"), Input("store_model","data"))
def update_facts_table(m): return facts_table_rows(m)

# -----------------------------
# UI: pick, swap, clear, clicks
# -----------------------------
@app.callback(
    Output("store_pick","data"),
    Output("fact_sub","value"),
    Output("fact_obj","value"),
    Output("store_sel","data"),
    Output("store_details","data", allow_duplicate=True),
    Output("store_status","data", allow_duplicate=True),
    Input("btn_pick_sub","n_clicks"),
    Input("btn_pick_obj","n_clicks"),
    Input("btn_swap","n_clicks"),
    Input("btn_clear_fact","n_clicks"),
    Input("cy","tapNode"),
    Input("cy","tapEdge"),
    State("store_pick","data"),
    State("fact_sub","value"),
    State("fact_obj","value"),
    State("store_sel","data"),
    State("store_model","data"),
    prevent_initial_call=True,
)
def ui_actions(_ps, _po, _swap, _clear, tap_node, tap_edge, pick, sub_val, obj_val, sel, m):
    pick = deepcopy(pick or {"mode": None})
    sel = deepcopy(sel or {"edge": None, "tapped_node": None, "tapped_pos": None})
    triggered = callback_context.triggered[0]["prop_id"].split(".")[0]

    if triggered == "btn_pick_sub":
        pick["mode"] = "sub"
        return pick, sub_val, obj_val, sel, "Pick SUBJECT: click a node in the graph (or use dropdown).", "Pick mode: SUBJECT"
    if triggered == "btn_pick_obj":
        pick["mode"] = "obj"
        return pick, sub_val, obj_val, sel, "Pick OBJECT: click a node in the graph (or use dropdown).", "Pick mode: OBJECT"
    if triggered == "btn_swap":
        return pick, obj_val, sub_val, sel, "Swapped Subject/Object.", "Swapped."
    if triggered == "btn_clear_fact":
        pick["mode"] = None
        return pick, None, None, sel, "Cleared Subject/Object.", "Cleared fact fields."

    if triggered == "cy" and tap_edge:
        data = tap_edge.get("data", {})
        sel["edge"] = data.get("id")
        gidx = data.get("gidx")
        extra = f"\nGLOBAL FACT INDEX: {gidx}" if gidx is not None else ""
        return pick, sub_val, obj_val, sel, f"EDGE\nid: {data.get('id')}\npred: {data.get('pred')}{extra}\nlabel:\n{data.get('label')}", f"Selected edge: {sel['edge']}"

    if triggered == "cy" and tap_node:
        data = tap_node.get("data", {})
        pos = tap_node.get("position")
        nid = data.get("id")
        sel["tapped_node"] = nid
        sel["tapped_pos"] = pos
        e = find_entity(m, nid) or {}
        details = f"NODE\nid: {nid}\ntype: {e.get('type','')}\nlabel: {e.get('label','')}\nlane: {e.get('lane','')}\nposition: {pos}"

        if pick.get("mode") == "sub":
            pick["mode"] = None
            return pick, nid, obj_val, sel, details, f"Picked SUBJECT: {nid}"
        if pick.get("mode") == "obj":
            pick["mode"] = None
            return pick, sub_val, nid, sel, details, f"Picked OBJECT: {nid}"

        return pick, sub_val, obj_val, sel, details, f"Tapped node: {nid}"

    return pick, sub_val, obj_val, sel, "Click a node or edge.", "Ready."

# Predicate options
@app.callback(
    Output("fact_pred","options"),
    Output("fact_pred","value"),
    Output("pred_hint","children"),
    Input("fact_sub","value"),
    Input("fact_obj","value"),
    State("fact_pred","value"),
    State("store_model","data"),
)
def update_predicates(sub_id, obj_id, current_pred, m):
    if not sub_id or not obj_id:
        return [], None, "Select Subject + Object (or use Pick)."
    preds = allowed_predicates_for_pair(m, sub_id, obj_id)
    if not preds:
        em = ent_map(m)
        st = em.get(sub_id, {}).get("type")
        ot = em.get(obj_id, {}).get("type")
        return [], None, f"No relation matches: {sub_id} ({st}) → {obj_id} ({ot}). Fix types or add a relation."
    chosen = current_pred if current_pred in preds else preds[0]
    hint = "Predicate kept from your selection/template." if current_pred in preds else "Predicate auto-selected (you can change it)."
    return [{"label": p, "value": p} for p in preds], chosen, hint

# Preview
@app.callback(
    Output("fact_preview","children"),
    Input("fact_sub","value"),
    Input("fact_obj","value"),
    Input("fact_pred","value"),
    State("store_model","data"),
)
def show_fact_preview(sub_id, obj_id, pred, m):
    em = ent_map(m)
    def one(eid):
        if not eid or eid not in em:
            return "(none)"
        e = em[eid]
        lab = (e.get("label") or "").strip()
        typ = e.get("type") or "?"
        return f"{eid}  [{typ}]\n  {lab}" if lab else f"{eid}  [{typ}]"
    lines = ["Preview", f"SUBJECT:\n{one(sub_id)}", f"OBJECT:\n{one(obj_id)}", f"PREDICATE: {pred or '(none)'}"]
    return "\n\n".join(lines)

# Mod candidates
@app.callback(
    Output("mod_with","options"),
    Output("mod_for","options"),
    Output("mod_in","options"),
    Output("mod_of","options"),
    Output("mod_from","options"),
    Output("mod_due_existing","options"),
    Input("store_model","data"),
)
def refresh_mod_candidates(m):
    return (
        [{"label": x, "value": x} for x in candidates_by_type(m, MOD_TYPES["WITH"])],
        [{"label": x, "value": x} for x in candidates_by_type(m, MOD_TYPES["FOR"])],
        [{"label": x, "value": x} for x in candidates_by_type(m, MOD_TYPES["IN"])],
        [{"label": x, "value": x} for x in candidates_by_type(m, MOD_TYPES["OF"])],
        [{"label": x, "value": x} for x in candidates_by_type(m, MOD_TYPES["FROM"])],
        [{"label": x, "value": x} for x in candidates_by_type(m, MOD_TYPES["DUE_TO"])],
    )

# Mod visibility
@app.callback(
    Output("wrap_with","style"),
    Output("wrap_for","style"),
    Output("wrap_in","style"),
    Output("wrap_of","style"),
    Output("wrap_from","style"),
    Output("wrap_due_existing","style"),
    Output("wrap_due_new","style"),
    Output("mods_hint","children"),
    Input("fact_pred","value"),
    State("store_model","data"),
)
def mod_visibility(pred, m):
    def show(b): return {"display":"block"} if b else {"display":"none"}
    if not pred:
        return show(False), show(False), show(False), show(False), show(False), show(False), show(False), "Pick a predicate to see relevant modifiers."
    k = relation_kind(m, pred)
    if k == "faces":
        return show(True), show(True), show(False), show(False), show(False), show(True), show(True), "FACES: optional WITH/FOR/DUE_TO."
    if k == "exists":
        return show(False), show(False), show(True), show(False), show(False), show(False), show(False), "EXISTS: requires IN."
    if k == "solves":
        return show(False), show(False), show(False), show(True), show(False), show(False), show(False), "SOLVES: requires OF."
    if k == "refactors":
        return show(True), show(False), show(False), show(False), show(True), show(False), show(False), "REFACTORS: requires FROM, optional WITH."
    return show(False), show(False), show(False), show(False), show(False), show(False), show(False), "No modifiers needed for this predicate."

# Layout rerun
@app.callback(
    Output("cy","layout"),
    Output("store_status","data", allow_duplicate=True),
    Input("btn_layout","n_clicks"),
    Input("rank_dir","value"),
    prevent_initial_call=True,
)
def rerun_layout(_n, rank_dir):
    rank_dir = rank_dir or "LR"
    layout = {"name":"dagre","rankDir":rank_dir,"nodeSep":50,"edgeSep":10,"rankSep":70,"padding":10}
    return layout, f"Re-ran layout ({rank_dir})."

# Fact selection -> load
@app.callback(
    Output("store_fact_edit","data"),
    Output("fact_sub","value", allow_duplicate=True),
    Output("fact_obj","value", allow_duplicate=True),
    Output("fact_pred","value", allow_duplicate=True),
    Output("mod_with","value", allow_duplicate=True),
    Output("mod_for","value", allow_duplicate=True),
    Output("mod_in","value", allow_duplicate=True),
    Output("mod_of","value", allow_duplicate=True),
    Output("mod_from","value", allow_duplicate=True),
    Output("mod_due_existing","value", allow_duplicate=True),
    Output("mod_due_new","value", allow_duplicate=True),
    Output("store_status","data", allow_duplicate=True),
    Input("facts_table","selected_rows"),
    State("facts_table","data"),
    State("store_model","data"),
    prevent_initial_call=True,
)
def load_fact_from_table(selected_rows, table_data, m):
    if not selected_rows:
        return {"index": None}, None, None, None, None, None, None, None, None, None, "", "Edit cleared."
    row = table_data[selected_rows[0]]
    idx = int(row["idx"])
    facts = m.get("facts") or []
    if idx < 0 or idx >= len(facts):
        return {"index": None}, None, None, None, None, None, None, None, None, None, "", "Selected row out of range."
    f = facts[idx]
    mods = f.get("mods") or {}
    return (
        {"index": idx},
        f.get("sub"),
        f.get("obj"),
        f.get("pred"),
        mods.get("WITH"),
        mods.get("FOR"),
        (mods.get("IN") or [None])[0] if mods.get("IN") else None,
        (mods.get("OF") or [None])[0] if mods.get("OF") else None,
        (mods.get("FROM") or [None])[0] if mods.get("FROM") else None,
        mods.get("DUE_TO"),
        "",
        f"Loaded fact #{idx}. Modify fields and click Update fact."
    )

@app.callback(
    Output("store_fact_edit","data", allow_duplicate=True),
    Output("facts_table","selected_rows", allow_duplicate=True),
    Output("store_status","data", allow_duplicate=True),
    Input("btn_fact_clear_edit","n_clicks"),
    prevent_initial_call=True,
)
def clear_fact_edit(_n):
    return {"index": None}, [], "Cleared edit mode."

# -----------------------------
# Mutations
# -----------------------------
@app.callback(
    Output("store_model","data"),
    Output("store_status","data", allow_duplicate=True),
    Output("store_fact_edit","data", allow_duplicate=True),
    Output("facts_table","selected_rows", allow_duplicate=True),
    Input("btn_ent_add","n_clicks"),
    Input("btn_ent_upd","n_clicks"),
    Input("btn_ent_del","n_clicks"),
    State("ent_select","value"),
    State("ent_id","value"),
    State("ent_type","value"),
    State("ent_label","value"),
    State("ent_lane","value"),
    Input("btn_rel_add","n_clicks"),
    Input("btn_rel_upd","n_clicks"),
    Input("btn_rel_del","n_clicks"),
    State("rel_select","value"),
    State("rel_name","value"),
    State("rel_sub_types","value"),
    State("rel_obj_types","value"),
    State("rel_kind","value"),
    Input("btn_fact_add","n_clicks"),
    Input("btn_fact_update","n_clicks"),
    Input("btn_fact_delete","n_clicks"),
    State("fact_sub","value"),
    State("fact_obj","value"),
    State("fact_pred","value"),
    State("mod_with","value"),
    State("mod_for","value"),
    State("mod_in","value"),
    State("mod_of","value"),
    State("mod_from","value"),
    State("mod_due_existing","value"),
    State("mod_due_new","value"),
    State("store_sel","data"),
    State("store_fact_edit","data"),
    Input("btn_save_pos","n_clicks"),
    Input("btn_save","n_clicks"),
    Input("btn_export","n_clicks"),
    Input("btn_focus_export","n_clicks"),
    Input("btn_export_focus","n_clicks"),
    State("store_filters","data"),
    State("store_model","data"),
    prevent_initial_call=True,
)
def mutate_model(
    _ent_add, _ent_upd, _ent_del,
    ent_select, ent_id, ent_type, ent_label, ent_lane,
    _rel_add, _rel_upd, _rel_del,
    rel_select, rel_name, rel_sub_types, rel_obj_types, rel_kind,
    _fact_add, _fact_update, _fact_del_edge,
    fact_sub, fact_obj, fact_pred,
    mod_with, mod_for, mod_in, mod_of, mod_from, mod_due_existing, mod_due_new,
    sel, fact_edit,
    _save_pos, _save, _export, _focus_export, _export_focus,
    filters,
    m
):
    m = deepcopy(m)
    sel = sel or {"edge": None, "tapped_node": None, "tapped_pos": None}
    fact_edit = fact_edit or {"index": None}
    edit_idx = fact_edit.get("index")
    triggered = callback_context.triggered[0]["prop_id"].split(".")[0]

    def parse_csv(s):
        return [x.strip() for x in (s or "").split(",") if x.strip()]

    def build_fact(pred, sub, obj):
        if not sub or not obj:
            return None, "Pick Subject and Object."
        if not pred:
            return None, "Pick a predicate."
        if pred not in allowed_predicates_for_pair(m, sub, obj):
            return None, "Type mismatch: predicate not valid for Subject/Object types."
        kind = relation_kind(m, pred)
        mods = {}
        if kind == "faces":
            if mod_with: mods["WITH"] = mod_with
            if mod_for: mods["FOR"] = mod_for
            due_ids = list(mod_due_existing or [])
            txt = (mod_due_new or "").strip()
            if txt:
                for part in [p.strip() for p in txt.split(";") if p.strip()]:
                    cid = f"C_{abs(hash(part)) % 10_000_000}"
                    ensure_entity(m, cid, "Cause", part)
                    due_ids.append(cid)
            if due_ids:
                mods["DUE_TO"] = due_ids
        elif kind == "exists":
            if not mod_in:
                return None, "This predicate requires IN."
            mods["IN"] = [mod_in]
        elif kind == "solves":
            if not mod_of:
                return None, "This predicate requires OF."
            mods["OF"] = [mod_of]
        elif kind == "refactors":
            if not mod_from:
                return None, "This predicate requires FROM."
            mods["FROM"] = [mod_from]
            if mod_with:
                mods["WITH"] = mod_with
        return {"pred": pred, "sub": sub, "obj": obj, "mods": mods}, None

    # Entities
    if triggered == "btn_ent_add":
        eid = (ent_id or "").strip()
        et = (ent_type or "").strip()
        if not eid or not et:
            return m, "Entity needs ID and Type.", {"index": None}, []
        if find_entity(m, eid):
            return m, f"Entity '{eid}' already exists.", {"index": None}, []
        m["entities"].append({"id": eid, "type": et, "label": (ent_label or "").strip(), "lane": (ent_lane or "").strip()})
        return m, f"Added entity: {eid}", {"index": None}, []

    if triggered == "btn_ent_upd":
        if not ent_select:
            return m, "Select an entity first.", {"index": None}, []
        e = find_entity(m, ent_select)
        if not e:
            return m, "Entity not found.", {"index": None}, []
        new_id = (ent_id or "").strip() or ent_select
        if new_id != ent_select:
            if find_entity(m, new_id):
                return m, f"Cannot rename: '{new_id}' already exists.", {"index": None}, []
            rename_entity_id(m, ent_select, new_id)
            ent_select = new_id
            e = find_entity(m, ent_select)
        e["type"] = (ent_type or e.get("type","")).strip()
        e["label"] = (ent_label or "").strip()
        e["lane"] = (ent_lane or "").strip()
        return m, f"Updated entity: {ent_select}", {"index": None}, []

    if triggered == "btn_ent_del":
        if not ent_select:
            return m, "Select an entity first.", {"index": None}, []
        eid = ent_select
        m["entities"] = [e for e in m["entities"] if e["id"] != eid]
        new_facts = []
        for f in m["facts"]:
            mods = f.get("mods") or {}
            mod_vals = []
            for k in mods:
                mod_vals.extend(mods.get(k) or [])
            if f["sub"] == eid or f["obj"] == eid or eid in mod_vals:
                continue
            new_facts.append(f)
        m["facts"] = new_facts
        return m, f"Deleted entity: {eid} (and related facts)", {"index": None}, []

    # Relations
    if triggered == "btn_rel_add":
        name = (rel_name or "").strip()
        if not name:
            return m, "Relation needs a name.", {"index": None}, []
        if name in (m.get("relations") or {}):
            return m, f"Relation '{name}' already exists.", {"index": None}, []
        m["relations"][name] = {"sub_types": parse_csv(rel_sub_types), "obj_types": parse_csv(rel_obj_types), "kind": rel_kind or "binary"}
        return m, f"Added relation: {name}", {"index": None}, []

    if triggered == "btn_rel_upd":
        if not rel_select:
            return m, "Select a relation first.", {"index": None}, []
        m["relations"][rel_select] = {"sub_types": parse_csv(rel_sub_types), "obj_types": parse_csv(rel_obj_types), "kind": rel_kind or "binary"}
        return m, f"Updated relation: {rel_select}", {"index": None}, []

    if triggered == "btn_rel_del":
        if not rel_select:
            return m, "Select a relation first.", {"index": None}, []
        m["relations"].pop(rel_select, None)
        m["facts"] = [f for f in m["facts"] if f["pred"] != rel_select]
        return m, f"Deleted relation: {rel_select} (and related facts)", {"index": None}, []

    # Facts add
    if triggered == "btn_fact_add":
        fact, err = build_fact(fact_pred, fact_sub, fact_obj)
        if err:
            return m, err, {"index": None}, []
        if is_duplicate_fact(m, fact):
            return m, "Duplicate fact detected (not added).", {"index": None}, []
        m["facts"].append(fact)
        return m, f"Added fact: {fact_sub} {fact_pred} {fact_obj}", {"index": None}, []

    # Facts update
    if triggered == "btn_fact_update":
        if edit_idx is None:
            return m, "Select a fact row in the table to edit.", {"index": None}, []
        if not (0 <= int(edit_idx) < len(m.get("facts") or [])):
            return m, "Edit index out of range.", {"index": None}, []
        fact, err = build_fact(fact_pred, fact_sub, fact_obj)
        if err:
            return m, err, {"index": None}, []
        if is_duplicate_fact(m, fact, ignore_index=int(edit_idx)):
            return m, "Duplicate fact detected (update cancelled).", {"index": None}, []
        m["facts"][int(edit_idx)] = fact
        return m, f"Updated fact #{edit_idx}.", {"index": None}, []

    # Facts delete selected edge (reliable: deletes by GLOBAL fact index factg_<gidx>)
    # ALSO supports deleting the fact currently loaded from the Facts table (edit_idx).
    if triggered == "btn_fact_delete":
        edge_id = (sel or {}).get("edge")

        # 1) Prefer deleting the selected FACT edge in the graph
        if edge_id and edge_id.startswith("factg_"):
            try:
                gidx = int(edge_id.split("_", 1)[1])
            except Exception:
                return m, "Could not parse selected fact id.", {"index": None}, []
            if 0 <= gidx < len(m["facts"]):
                f = m["facts"].pop(gidx)
                return m, f"Deleted fact #{gidx}: {f['sub']} {f['pred']} {f['obj']}", {"index": None}, []
            return m, "Selected fact index out of range.", {"index": None}, []

        # 2) Fallback: if a row was selected in the Facts table (loaded into editor), delete that
        if edit_idx is not None and 0 <= int(edit_idx) < len(m["facts"]):
            gidx = int(edit_idx)
            f = m["facts"].pop(gidx)
            return m, f"Deleted fact (from table) #{gidx}: {f['sub']} {f['pred']} {f['obj']}", {"index": None}, []

        return m, "Select a FACT edge in the graph OR select a row in the Facts table first.", {"index": None}, []

    # Save position
    if triggered == "btn_save_pos":
        nid = (sel or {}).get("tapped_node")
        pos = (sel or {}).get("tapped_pos")
        if not nid or not isinstance(pos, dict):
            return m, "Tap a node first.", {"index": None}, []
        e = find_entity(m, nid)
        if not e:
            return m, "Tapped node not found.", {"index": None}, []
        e["pos"] = {"x": pos.get("x", 0), "y": pos.get("y", 0)}
        return m, f"Stored position for {nid}.", {"index": None}, []

    # Save (backup)
    if triggered == "btn_save":
        backup_path = save_model_with_backup(m)
        return m, f"Saved model.json (backup: {backup_path})", {"index": None}, []

    # Export full
    if triggered == "btn_export":
        export_graphml_yed_simple(m, GRAPHML_OUT)
        return m, f"Exported GraphML (full) to {GRAPHML_OUT}", {"index": None}, []

    # Focus here + export (uses tapped node as focus, then exports focused GraphML)
    if triggered == "btn_focus_export":
        nid = (sel or {}).get("tapped_node")
        if not nid:
            return m, "Click a node first, then Focus+Export.", {"index": None}, []
        filters2 = deepcopy(filters or {})
        filters2["focus_node"] = nid
        sub = extract_submodel(m, filters2)
        export_graphml_yed_simple(sub, GRAPHML_FOCUS_OUT)
        return m, f"Focused export created: {GRAPHML_FOCUS_OUT} (focus={nid})", {"index": None}, []

    # Export focused
    if triggered == "btn_export_focus":
        sub = extract_submodel(m, filters or {})
        export_graphml_yed_simple(sub, GRAPHML_FOCUS_OUT)
        focus_node = (filters or {}).get("focus_node")
        return m, f"Exported GraphML (focused) to {GRAPHML_FOCUS_OUT} (focus={focus_node})", {"index": None}, []

    return m, "Ready.", {"index": None}, []

