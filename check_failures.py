import json, glob

for sid in ["1776179048937","1776179255813","1776180981877","1776182009483","1776182140352","1776182399303"]:
    f = "/app/data/conversation_history/2026/04/14/session_{}.json".format(sid)
    with open(f) as fh:
        d = json.load(fh)
    nodes = d.get("nodes",[])
    meta = d.get("graph",{})
    errors = [(n.get("id"),n.get("error")) for n in nodes if isinstance(n,dict) and n.get("error")]
    print("Session {}: status={} errors={}".format(sid, meta.get("status"), str(errors)[:120]))
