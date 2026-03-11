import json, sys

with open(r'storage\chats\32d300fe548e.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

msgs = data.get('messages', [])
print(f'Totale messaggi: {len(msgs)}')

for i, m in enumerate(msgs):
    role = m.get('role', '?')
    text = m.get('content', '')[:300]
    meta = m.get('metadata', {})
    backend = meta.get('backend', '')
    cit_count = len(meta.get('citations', []))
    print(f'\n--- MSG {i} [{role}] ---')
    print(text)
    if meta:
        print(f'  [backend={backend} citations={cit_count}]')
