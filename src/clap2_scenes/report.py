"""Export a self-contained, playable HTML report with no external services."""

import base64, json, subprocess, tempfile
from pathlib import Path


def render_report(result, audio, output):
    """Embed mono 128 kbit/s MP3 and the result in an offline HTML player."""
    with tempfile.TemporaryDirectory() as folder:
        mp3 = Path(folder) / "audio.mp3"
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-nostdin",
                "-y",
                "-protocol_whitelist",
                "file,pipe",
                "-i",
                str(Path(audio).resolve()),
                "-map",
                "0:a:0",
                "-vn",
                "-ac",
                "1",
                "-c:a",
                "libmp3lame",
                "-b:a",
                "128k",
                str(mp3),
            ],
            check=True,
        )
        encoded = base64.b64encode(mp3.read_bytes()).decode("ascii")
    data = json.dumps(result, ensure_ascii=False, allow_nan=False).replace(
        "<", "\\u003c"
    )
    html = TEMPLATE.replace("__RESULT_JSON__", data).replace(
        "__AUDIO_BASE64__", encoded
    )
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(html, encoding="utf8")
    return target


TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CLAP2 · Audio scene explorer</title>
<style>
:root{color-scheme:light;--ink:#192b35;--muted:#5d707b;--line:#dae4e8;--paper:#fff;--teal:#067b77}*{box-sizing:border-box}body{margin:0;background:#f2f6f7;color:var(--ink);font:16px/1.55 system-ui,sans-serif}main{max-width:1180px;margin:0 auto;padding:48px 28px}header{margin-bottom:28px}.eyebrow{text-transform:uppercase;letter-spacing:.17em;font-size:12px;font-weight:750;color:var(--teal)}h1{font-size:clamp(30px,5vw,48px);letter-spacing:-.04em;line-height:1.1;margin:14px 0}h2{font-size:23px;letter-spacing:-.02em;margin:8px 0 20px}p{color:var(--muted);max-width:850px}.pill{display:inline-block;border:1px solid var(--line);background:white;padding:5px 11px;border-radius:99px;font-size:12px;margin:0 5px 5px 0}.player{padding:24px;background:white;border:1px solid var(--line);border-radius:20px;box-shadow:0 12px 40px #102f3508}audio{width:100%;display:block;margin:12px 0 22px}#strip{display:flex;gap:4px;height:48px;position:relative}.segment{border:0;border-radius:8px;min-width:0;cursor:pointer;font-weight:750;color:#143039}.segment:focus-visible,button:focus-visible{outline:3px solid #202d3d;outline-offset:3px}#cursor{position:absolute;width:2px;height:58px;top:-5px;background:#192b35;pointer-events:none}#clock{font-variant-numeric:tabular-nums;font-weight:650}.between{display:flex;justify-content:space-between;align-items:center;gap:10px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:22px;margin-top:24px}.panel{background:white;border:1px solid var(--line);border-radius:20px;padding:25px;min-width:0}.matches{list-style:none;padding:0;margin:0}.match{padding:15px 0;border-top:1px solid var(--line);display:grid;grid-template-columns:28px 1fr 60px;gap:10px;align-items:start}.rank{font-size:12px;font-weight:800;color:var(--teal);padding-top:3px}.caption{font-size:17px;font-weight:600;line-height:1.45;overflow-wrap:anywhere}.score{text-align:right;font-variant-numeric:tabular-nums;font-weight:700;font-size:14px}.origin{font-size:11px;color:var(--muted);font-weight:500;margin-top:5px}.bar{height:3px;border-radius:4px;margin-top:8px;background:var(--teal)}small{color:var(--muted)}details{background:#e8eff1;border-radius:14px;margin-top:24px;padding:18px 22px}summary{font-weight:650;cursor:pointer}#download{border:1px solid var(--line);background:white;padding:8px 13px;border-radius:9px;color:var(--ink);cursor:pointer}footer{margin:20px 0;font-size:12px;color:var(--muted)}@media(max-width:760px){main{padding:28px 16px}.grid{grid-template-columns:1fr}.panel{padding:20px}.caption{font-size:16px}.between{flex-wrap:wrap}}@media(prefers-reduced-motion:no-preference){.match{animation:appear .15s ease}@keyframes appear{from{opacity:.5}to{opacity:1}}}
</style></head><body><main><header><div class="eyebrow">CLAP2 / Audio scene explorer</div><h1>Hear the scene. Follow the captions.</h1><p>Play or seek through the recording. The left panel follows each time step; the right panel summarizes the current segment using its nearest caption matches.</p><div id="metadata"></div></header>
<section class="player" aria-label="Audio and segments"><div class="between"><strong id="source"></strong><span id="clock">0.00 s</span></div><audio controls preload="none" src="data:audio/mpeg;base64,__AUDIO_BASE64__"></audio><div id="strip" aria-label="Click a segment to seek"></div><div class="between"><small>Click a colored segment to seek.</small><small id="timing"></small></div></section>
<div class="grid"><section class="panel"><div class="eyebrow">01 / Timeline matches</div><h2 id="bin-title"></h2><ol class="matches" id="timeline"></ol></section><section class="panel"><div class="eyebrow">02 / Segment matches</div><h2 id="segment-title"></h2><ol class="matches" id="segment"></ol></section></div>
<details><summary>How to read this report</summary><p>Each timeline query is retrieved against the full selected candidate index. Sliding-window features average the embeddings of overlapping windows; optional smoothing averages nearby queries. Segment captions use a normalized duration-weighted mean of the cached features, unless fresh isolated-cut encoding was requested.</p><p>Numbers are cosine similarities, or temporal means of cosine similarities. They are not probabilities. Boundaries are heuristic acoustic changes; they do not separate simultaneous sound sources. Adjacent hierarchical clustering merges neighboring spans and may return fewer than the requested number to respect the minimum duration.</p><p id="configuration"></p><button id="download">Download predictions · JSON</button></details><footer>Self-contained report · mono MP3, 128 kbit/s · audio loads on demand · no network services required</footer></main>
<script type="application/json" id="result">__RESULT_JSON__</script><script>
const result=JSON.parse(document.getElementById('result').textContent),audio=document.querySelector('audio'),$=id=>document.getElementById(id),colors=['#a5dbd0','#bbd4f1','#ddcaef','#f1d1a4','#eebfc8','#cedcaa'];let lastBin=-1,lastSegment=-1;
$('source').textContent=result.source||'Audio recording';$('timing').textContent=result.duration_seconds.toFixed(2)+' seconds · '+result.segments.length+' segments';
for(const text of ['CLAP2 '+result.model.size.toUpperCase(),result.index.index_size+' + Luna',result.index.rows.toLocaleString()+' candidates',result.config.window_seconds+' s windows / '+result.config.stride_seconds+' s steps']){const e=document.createElement('span');e.className='pill';e.textContent=text;$('metadata').append(e)}
$('configuration').textContent='Method: '+result.config.segmentation.method+'. Minimum segment: '+result.config.segmentation.min_segment_seconds+' s. Smoothing: '+result.config.smoothing+' ('+result.config.smoothing_seconds+' s). Feature mode: '+result.config.feature_method+'.';
result.segments.forEach((s,i)=>{const b=document.createElement('button');b.className='segment';b.style.flex=String(s.end-s.start);b.style.background=colors[i%colors.length];b.textContent=String(i+1);b.title='Segment '+(i+1)+' · '+s.start.toFixed(2)+'–'+s.end.toFixed(2)+' s';b.setAttribute('aria-label',b.title);b.onclick=()=>{audio.currentTime=s.start;update()};$('strip').append(b)});const cursor=document.createElement('i');cursor.id='cursor';$('strip').append(cursor);
function matches(id,hits){$(id).replaceChildren();hits.forEach((h,i)=>{const li=document.createElement('li');li.className='match';const rank=document.createElement('span');rank.className='rank';rank.textContent=String(i+1).padStart(2,'0');const body=document.createElement('div'),caption=document.createElement('div'),origin=document.createElement('div'),bar=document.createElement('div'),score=document.createElement('span');caption.className='caption';caption.textContent=h.text;origin.className='origin';origin.textContent=h.origin==='luna_taxonomy'?'Luna taxonomy sentence':'Retrieved cluster caption';bar.className='bar';bar.style.width=Math.max(0,Math.min(100,h.score*100))+'%';bar.style.background=colors[i%colors.length];score.className='score';score.textContent=h.score.toFixed(3);body.append(caption,origin,bar);li.append(rank,body,score);$(id).append(li)})}
function at(rows,t){let lo=0,hi=rows.length;while(lo<hi){const m=(lo+hi)>>1;if(rows[m].start<=t+1e-7)lo=m+1;else hi=m}return Math.max(0,lo-1)}
function update(){const t=Math.min(audio.currentTime||0,result.duration_seconds),i=at(result.timeline,t),j=at(result.segments,t);$('clock').textContent=t.toFixed(2)+' / '+result.duration_seconds.toFixed(2)+' s';cursor.style.left=(t/result.duration_seconds*100)+'%';if(i!==lastBin){const b=result.timeline[i];$('bin-title').textContent=b.start.toFixed(2)+'–'+b.end.toFixed(2)+' s';matches('timeline',b.predictions);lastBin=i}if(j!==lastSegment){const s=result.segments[j];$('segment-title').textContent='Segment '+(j+1)+' · '+s.start.toFixed(2)+'–'+s.end.toFixed(2)+' s';matches('segment',s.predictions);lastSegment=j}}
audio.addEventListener('timeupdate',update);audio.addEventListener('seeked',update);$('download').onclick=()=>{const url=URL.createObjectURL(new Blob([JSON.stringify(result,null,2)],{type:'application/json'})),a=document.createElement('a');a.href=url;a.download='clap2-scene-predictions.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};update();
</script></body></html>"""
