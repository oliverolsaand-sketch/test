#!/usr/bin/env python3
"""
fetch_skin.py - download a Java-edition skin by username (or UUID) from Mojang.

  python3 fetch_skin.py <name|uuid> [-o out.png]

Prints JSON: {"name","uuid","slim","path"}.  `slim` comes from Mojang's
"model": "slim" metadata, which is more reliable than guessing from pixels.
Fallback host: mc-heads.net (if the Mojang API is rate-limited).
Bedrock players without a Java account: ask the user for the skin PNG.
"""
import base64, json, sys, argparse, urllib.request, os

UA = {'User-Agent': 'mc-thumbnail-skill/1.0'}

def get(url, timeout=20):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout).read()

def fetch(who: str, out: str):
    uuid, name = None, who
    if len(who.replace('-', '')) == 32:
        uuid = who.replace('-', '')
    else:
        try:
            j = json.loads(get(f'https://api.mojang.com/users/profiles/minecraft/{who}'))
            uuid, name = j['id'], j['name']
        except Exception as e:
            print(f'mojang lookup failed ({e}); trying mc-heads', file=sys.stderr)
    slim = None
    if uuid:
        prof = json.loads(get(f'https://sessionserver.mojang.com/session/minecraft/profile/{uuid}'))
        name = prof.get('name', name)
        tex = json.loads(base64.b64decode(prof['properties'][0]['value']))
        skin = tex['textures'].get('SKIN')
        if skin:
            url = skin['url'].replace('http://', 'https://')
            slim = skin.get('metadata', {}).get('model') == 'slim'
            data = get(url)
        else:
            data = None
    else:
        data = None
    if data is None:
        data = get(f'https://mc-heads.net/skin/{who}')
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    open(out, 'wb').write(data)
    return dict(name=name, uuid=uuid, slim=slim, path=os.path.abspath(out))

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('who'); ap.add_argument('-o', '--out', default=None)
    a = ap.parse_args()
    print(json.dumps(fetch(a.who, a.out or f'{a.who}_skin.png')))
