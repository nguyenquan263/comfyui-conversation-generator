#!/home/qnguyen3/.venvs/comfyui/bin/python
"""
Write one loadable ComfyUI workflow per pipeline into workflows/ and into ComfyUI's own
workflow folder, so they show up in the ComfyUI sidebar (Workflows > Open). They are the exact
graphs api.py submits, converted to the editor's graph format with the node definitions of the
running ComfyUI (so it must be up: ./api.py starts it).

    ./export_workflows.py

Each pipeline also gets a <name>.api.json next to it in workflows/ (the raw API form).
"""
from __future__ import annotations

import json
import shutil
import sys
import urllib.request
from pathlib import Path

import pipelines

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "workflows"
COMFY_USER_WORKFLOWS = pipelines.COMFY_DIR / "user" / "default" / "workflows"
COMFY_URL = "http://127.0.0.1:8188"

PROMPT = "A red fox trots through a snowy forest at dawn, soft golden light, snow falling gently, cinematic"
WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}


def flux2_then_ltx25(o: pipelines.Options) -> dict:
    """Both stages in one graph: the Flux keyframe feeds LTX directly instead of going through a file."""
    flux = pipelines.build_flux2_image(o, 768, 512)["graph"]
    ltx = pipelines.build_ltx25(o, image_file="placeholder.png")["graph"]
    g = dict(flux)
    for nid, node in ltx.items():
        node = json.loads(json.dumps(node))
        for v in node["inputs"].values():
            if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str):
                v[0] = "L" + v[0]
        g["L" + nid] = node
    g["L21"]["inputs"]["image"] = ["12", 0]        # LTXVPreprocess <- Flux VAEDecode
    del g["L20"]                                   # the LoadImage placeholder
    return g


# ----------------------------------------------------------------------------- API -> editor format

def is_link(v) -> bool:
    return isinstance(v, list) and len(v) == 2 and isinstance(v[0], str)


def input_specs(defn: dict) -> list[tuple[str, str, dict, bool]]:
    """(name, type, options, is_widget) for every declared input, in the editor's order."""
    out = []
    order = defn.get("input_order", {})
    for section in ("required", "optional"):
        inputs = defn["input"].get(section, {})
        for name in order.get(section, list(inputs)):
            spec = inputs[name]
            typ = spec[0] if isinstance(spec, list) else spec
            opts = spec[1] if isinstance(spec, list) and len(spec) > 1 and isinstance(spec[1], dict) else {}
            if isinstance(typ, list):
                typ, opts = "COMBO", {**opts, "options": typ}
            out.append((name, typ, opts, typ in WIDGET_TYPES))
    return out


def widget_default(typ: str, opts: dict):
    if "default" in opts:
        return opts["default"]
    if typ == "COMBO":
        return (opts.get("options") or [""])[0]
    return {"INT": 0, "FLOAT": 0.0, "STRING": "", "BOOLEAN": False}[typ]


def to_editor(graph: dict, oi: dict) -> dict:
    ids = {nid: i + 1 for i, nid in enumerate(graph)}
    # depth = longest chain of links feeding a node, for a left-to-right layout
    depth: dict[str, int] = {}
    def d(nid: str) -> int:
        if nid not in depth:
            depth[nid] = 1 + max([d(v[0]) for v in graph[nid]["inputs"].values() if is_link(v)] or [-1])
        return depth[nid]
    for nid in graph:
        d(nid)

    nodes, links = [], []
    link_id = 0
    out_links: dict[tuple[str, int], list[int]] = {}
    column_y: dict[int, int] = {}
    for order, (nid, node) in enumerate(graph.items()):
        ctype = node["class_type"]
        defn = oi[ctype]
        specs = input_specs(defn)
        inputs, widgets = [], []
        for name, typ, opts, is_widget in specs:
            val = node["inputs"].get(name)
            if ctype == "SaveVideo" and name in ("format", "codec"):
                continue
            if is_widget and not is_link(val):
                widgets.append(val if val is not None else widget_default(typ, opts))
                if opts.get("control_after_generate"):
                    widgets.append("fixed")
                if opts.get("image_upload"):
                    widgets.append("image")
                continue
            link = None
            if is_link(val):
                link_id += 1
                link = link_id
                src, slot = val
                links.append([link_id, ids[src], slot, ids[nid], len(inputs), oi[graph[src]["class_type"]]["output"][slot]])
                out_links.setdefault((src, slot), []).append(link_id)
            entry = {"name": name, "type": typ, "link": link}
            if is_widget:                     # a widget driven by a link
                entry["widget"] = {"name": name}
            inputs.append(entry)
        if ctype == "SaveVideo":
            widgets += [node["inputs"].get("format", "mp4"), "auto"]
        outputs = [{"name": n, "type": t, "links": None, "slot_index": i}
                   for i, (t, n) in enumerate(zip(defn["output"], defn.get("output_name") or defn["output"]))]
        col = depth[nid]
        y = column_y.get(col, 0)
        multiline = any(t == "STRING" and o.get("multiline") for _, t, o, w in specs if w)
        height = 60 + 24 * (len(inputs) + len(outputs) + len(widgets)) + (90 if multiline else 0)
        nodes.append({
            "id": ids[nid], "type": ctype, "pos": [60 + col * 420, 60 + y], "size": [360, height],
            "flags": {}, "order": order, "mode": 0, "inputs": inputs, "outputs": outputs,
            "properties": {"Node name for S&R": ctype}, "widgets_values": widgets,
        })
        column_y[col] = y + height + 40
    for n, nid in zip(nodes, graph):
        for o in n["outputs"]:
            o["links"] = out_links.get((nid, o["slot_index"]), [])
    return {"last_node_id": len(nodes), "last_link_id": link_id, "nodes": nodes, "links": links,
            "groups": [], "config": {}, "extra": {}, "version": 0.4}


def main() -> None:
    try:
        oi = json.load(urllib.request.urlopen(COMFY_URL + "/object_info", timeout=10))
    except Exception as e:
        sys.exit(f"ComfyUI must be running at {COMFY_URL} to read node definitions ({e}); start ./api.py first")
    o = pipelines.Options(prompt=PROMPT, seed=42, seconds=3)
    graphs = {
        "wan_cartoon_t2v": pipelines.build_wan(pipelines.Options(prompt="A cheerful orange cat in a blue scarf walks through a sunny meadow", seed=42))["graph"],
        "flux2_klein_image": pipelines.build_flux2_image(o)["graph"],
        "ltx25_t2v": pipelines.build_ltx25(o)["graph"],
        "ltx25_i2v": pipelines.build_ltx25(o, image_file="example.png")["graph"],
        "flux2_to_ltx25": flux2_then_ltx25(o),
    }
    OUT.mkdir(exist_ok=True)
    COMFY_USER_WORKFLOWS.mkdir(parents=True, exist_ok=True)
    for name, graph in graphs.items():
        (OUT / f"{name}.api.json").write_text(json.dumps(graph, indent=1, ensure_ascii=False))
        editor = to_editor(graph, oi)
        path = OUT / f"{name}.json"
        path.write_text(json.dumps(editor, indent=1, ensure_ascii=False))
        shutil.copyfile(path, COMFY_USER_WORKFLOWS / f"{name}.json")
        print(f"  {name:22} {len(editor['nodes']):3} nodes {len(editor['links']):3} links")
    print(f"written to {OUT} and {COMFY_USER_WORKFLOWS}")


if __name__ == "__main__":
    main()
