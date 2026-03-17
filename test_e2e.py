import urllib.request, json, time

req = urllib.request.Request(
    'http://localhost:8000/runs',
    data=json.dumps({
        'query': '[Patient ID: test-e2e] [Execution Mode: fast] Request: {"hemoglobin": 7.7, "wbc": 7000, "rbc": 4.5, "platelets": 250000}'
    }).encode(),
    headers={'Content-Type': 'application/json'},
    method='POST'
)
resp = urllib.request.urlopen(req)
run_data = json.loads(resp.read().decode())
run_id = run_data['id']
print(f'Run started: {run_id}')

for i in range(60):
    time.sleep(3)
    r = urllib.request.urlopen(f'http://localhost:8000/runs/{run_id}')
    data = json.loads(r.read().decode())
    status = data.get('status')
    if status in ('completed', 'failed'):
        print(f'Status: {status}')
        for node in data.get('graph', {}).get('nodes', []):
            nid = node.get('id')
            out_str = node.get('data', {}).get('output', '')
            if out_str and out_str != '""':
                try:
                    out = json.loads(out_str) if isinstance(out_str, str) else out_str
                except:
                    out = None
                if isinstance(out, dict):
                    c = out.get('confidence')
                    r_ = out.get('risk_level')
                    f = out.get('flags')
                    if c is not None or r_ or f:
                        print(f'  {nid}: confidence={c}, risk_level={r_}, flags={f}')
        break
    if i % 5 == 0:
        print(f'  [{i*3}s] status={status}')
else:
    print('Timed out waiting for run')
