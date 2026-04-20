import json, os, glob
from datetime import datetime

HNSW_CUTOFF = 1776177447691

session_files = sorted(glob.glob("/app/data/conversation_history/**/*.json", recursive=True))

rows = []
for f in session_files:
    try:
        with open(f) as fh:
            d = json.load(fh)
        sid = int(os.path.basename(f).replace("session_","").replace(".json",""))
        meta = d.get("graph", {})
        nodes = d.get("nodes", [])
        created_at = meta.get("created_at","")
        status = meta.get("status","?")
        query = meta.get("original_query","")
        mode = "full" if "full" in query.lower() else "fast"
        total_tokens = 0
        all_starts = []
        all_ends = []
        for n in nodes:
            if not isinstance(n, dict): continue
            total_tokens += (n.get("total_tokens") or 0)
            if n.get("start_time"):
                try: all_starts.append(datetime.fromisoformat(n["start_time"]))
                except: pass
            if n.get("end_time"):
                try: all_ends.append(datetime.fromisoformat(n["end_time"]))
                except: pass
        wall_s = None
        if all_starts and all_ends:
            wall_s = round((max(all_ends) - min(all_starts)).total_seconds(), 1)
        is_12b = "12b" in str(meta.get("globals_schema",{}))
        era = "BEFORE_FlatL2" if sid <= HNSW_CUTOFF else "AFTER_HNSW"
        rows.append({"sid":sid,"date":created_at[:10] if created_at else "?","era":era,"mode":mode,"status":status,"wall_s":wall_s,"nodes":len(nodes),"tokens":total_tokens,"is_12b":is_12b})
    except:
        pass

print("SID                DATE         ERA             MODE   STATUS       WALL(s)    NODES  TOKENS  12B")
print("-"*100)
for r in sorted(rows, key=lambda x: x["sid"]):
    w = str(r["wall_s"])+"s" if r["wall_s"] else "?"
    flag = "YES" if r["is_12b"] else ""
    print("{:<18} {:<12} {:<16} {:<7} {:<12} {:<10} {:<7} {:<8} {}".format(
        r["sid"],r["date"],r["era"],r["mode"],r["status"],w,r["nodes"],r["tokens"],flag))

before = [r for r in rows if r["era"]=="BEFORE_FlatL2" and r["wall_s"] and not r["is_12b"] and r["status"]=="completed"]
after  = [r for r in rows if r["era"]=="AFTER_HNSW" and r["wall_s"] and not r["is_12b"] and r["status"]=="completed"]

print()
print("=== SUMMARY (12b excluded, completed only) ===")
if before:
    avg_b = sum(r["wall_s"] for r in before)/len(before)
    print("BEFORE HNSWFlat  ({} sessions): avg={:.1f}s  min={:.1f}s  max={:.1f}s".format(
        len(before), avg_b, min(r["wall_s"] for r in before), max(r["wall_s"] for r in before)))
if after:
    avg_a = sum(r["wall_s"] for r in after)/len(after)
    print("AFTER  HNSWFlat  ({} sessions): avg={:.1f}s  min={:.1f}s  max={:.1f}s".format(
        len(after), avg_a, min(r["wall_s"] for r in after), max(r["wall_s"] for r in after)))
