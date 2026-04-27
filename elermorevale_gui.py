#!/usr/bin/env python3
"""
elermorevale_gui.py — Dual-View Power Flow Dashboard

Two views switchable via tabs:
  STATIC TOPOLOGY — original node colours by voltage level, no animation
  LIVE FLOW       — animated particles, voltage-coloured nodes, violation glow

Features:
  * Canvas-rendered network with spring layout
  * Animated yellow-gold particles flowing along edges (power flow direction)
  * Voltage-coloured nodes with pulsing glow on violations
  * Play / pause / speed / time scrubber
  * Baseline <-> QP toggle
  * Synced Plotly charts (substation power + voltage histogram)
  * Full sidebar with live stats, legend, network info, AS 60038 limits

Usage:
    python elermorevale_gui.py                            # topology only
    python elermorevale_gui.py --simulate --day 190       # with simulation
    python elermorevale_gui.py --simulate --open          # simulate + open

Prerequisites:
    pip install networkx numpy
    pip install dss-python pandas   (only for --simulate)
"""
import argparse, glob, json, logging, os, re, webbrowser
from collections import defaultdict, deque
import networkx as nx, numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def parse_glm(fp):
    with open(fp,"r",encoding="utf-8",errors="replace") as f: text=f.read()
    text=re.sub(r"//[^\n]*","",text); objs=[]
    for m in re.finditer(r"object\s+(\w+)\s*\{([^}]*)\}",text,re.DOTALL):
        props={}
        for pm in re.finditer(r"([\w.]+)\s+([^;]+);",m.group(2)):
            props[pm.group(1).strip()]=pm.group(2).strip()
        objs.append((m.group(1),props))
    return objs

def gfloat(v,d=0.0):
    if v is None: return float(d)
    s=str(v).strip(); m=re.match(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?",s)
    return float(m.group(0)) if m else float(d)

def build_topology(glm_dir):
    all_objs=[]
    for fp in sorted(glob.glob(os.path.join(glm_dir,"**","*.glm"),recursive=True)):
        if "__MACOSX" in fp: continue
        for ot,p in parse_glm(fp): all_objs.append((ot,p))
    by_type=defaultdict(list)
    for ot,p in all_objs: by_type[ot].append(p)
    G=nx.Graph(); bus_vl={}
    for p in by_type.get("transformer",[]):
        f,t=p.get("from",""),p.get("to","")
        if f and t:
            G.add_edge(f,t,element="transformer",name=p.get("name",""))
            if "132" in f or "Jesmond" in f: bus_vl[f]="HV";bus_vl[t]="MV"
            elif "TXZoneSub" not in p.get("name",""): bus_vl.setdefault(f,"MV");bus_vl[t]="LV"
    for p in by_type.get("regulator",[]):
        f,t=p.get("from",""),p.get("to","")
        if f and t: G.add_edge(f,t,element="regulator",name=p.get("name",""))
    for ot in ["overhead_line","underground_line","triplex_line"]:
        for p in by_type.get(ot,[]):
            f,t=p.get("from",""),p.get("to","")
            if f and t: G.add_edge(f,t,element=ot)
    for ot in ["switch","fuse"]:
        for p in by_type.get(ot,[]):
            f,t=p.get("from",""),p.get("to","")
            if f and t: G.add_edge(f,t,element=ot)
    for n in G.nodes():
        if n not in bus_vl:
            if "132" in n or "Jesmond" in n: bus_vl[n]="HV"
            elif n.startswith("_100") or "BusZone" in n: bus_vl[n]="MV"
            else: bus_vl[n]="LV"
    nx.set_node_attributes(G,bus_vl,"vl")
    parent_of={}
    for pl in by_type.values():
        for p in pl:
            n,par=p.get("name",""),p.get("parent","")
            if n and par: parent_of[n]=par
    def resolve(nm):
        c=nm
        for _ in range(10):
            if c not in parent_of: return c
            c=parent_of[c]
        return c
    lc=defaultdict(int)
    for p in by_type.get("load",[]): lc[resolve(p.get("parent",""))]+=1
    nx.set_node_attributes(G,{n:n in lc for n in G.nodes()},"hl")
    nx.set_node_attributes(G,dict(lc),"nl")
    # BFS depth for edge directionality
    source=None
    for n in G.nodes():
        if "Jesmond" in n or "132" in n: source=n; break
    depth={}
    if source:
        depth[source]=0; q=deque([source])
        while q:
            c=q.popleft()
            for nb in G.neighbors(c):
                if nb not in depth: depth[nb]=depth[c]+1; q.append(nb)
    nx.set_node_attributes(G,depth,"depth")
    logger.info("Topology: %d nodes, %d edges, %d components",
                G.number_of_nodes(),G.number_of_edges(),nx.number_connected_components(G))
    return G

def run_simulation(glm_dir,common_dir,profiles_csv,day_idx):
    from elermorevale_openDSS import (build_elermorevale,get_network_load_names,
        map_customers_to_network_loads,select_monitored_loads,add_monitors,
        attach_baseline_shapes,attach_loadshapes,run_daily,collect_voltages,
        collect_tx_power,collect_losses,load_profiles_from_csv)
    from dss import DSS as dss
    profiles=load_profiles_from_csv(profiles_csv)
    cids=sorted(profiles.keys())
    build_elermorevale(glm_dir,common_dir,skip_generators=True)
    ln=get_network_load_names()
    lcm=map_customers_to_network_loads(cids,ln)
    mon=select_monitored_loads(lcm,n_monitors=150)
    res={}
    for lbl,ub in [("baseline",True),("qp",False)]:
        logger.info("Running %s for day %d ...",lbl,day_idx)
        build_elermorevale(glm_dir,common_dir,skip_generators=True)
        add_monitors(mon)
        if ub: ds=attach_baseline_shapes(lcm,profiles,day_idx)
        else: ds=attach_loadshapes(lcm,profiles,day_idx)
        run_daily(); v=collect_voltages(mon); tp,tq=collect_tx_power(); lk,_=collect_losses()
        res[lbl]={"voltages":{k:val.tolist() for k,val in v.items()},
                  "tx_p":tp.tolist(),"loss_kw":float(lk),"date":ds}
    lbm={}
    for l in mon:
        try:
            dss.ActiveCircuit.SetActiveElement(f"Load.{l}")
            bn=dss.ActiveCircuit.ActiveElement.BusNames
            if bn: lbm[l]=bn[0].split(".")[0]
        except: pass
    res["load_bus_map"]=lbm; return res

def generate_dashboard(G,sim_data=None,output_path="elermorevale_dashboard.html"):
    logger.info("Computing spring layout (may take ~30s for %d nodes) ...", G.number_of_nodes())
    pos=nx.spring_layout(G, k=2.0, iterations=80, seed=42)
    nl=list(G.nodes()); ni={n:i for i,n in enumerate(nl)}
    nx_a=[round(float(pos[n][0]),5) for n in nl]
    ny_a=[round(float(pos[n][1]),5) for n in nl]
    vl_a=[{"HV":0,"MV":1,"LV":2}.get(G.nodes[n].get("vl","LV"),2) for n in nl]
    hl_a=[1 if G.nodes[n].get("hl",False) else 0 for n in nl]
    nloads=[G.nodes[n].get("nl",0) for n in nl]
    dp_a=[G.nodes[n].get("depth",999) for n in nl]
    ea,eb,et=[],[],[]
    for u,v,d in G.edges(data=True):
        if u in ni and v in ni:
            du=G.nodes[u].get("depth",999); dv=G.nodes[v].get("depth",999)
            if du<=dv: ea.append(ni[u]);eb.append(ni[v])
            else: ea.append(ni[v]);eb.append(ni[u])
            e=d.get("element","line")
            et.append(0 if e in ("transformer","regulator") else 1)
    has_sim=sim_data is not None; sim_js="{}"
    if has_sim:
        lbm = sim_data.get("load_bus_map", {})
        ni_lower = {n.lower(): i for i, n in enumerate(nl)}   
        for sc in ["baseline", "qp"]:
            sd = sim_data[sc]
            nv = {}
            unmatched = 0
            for lname, vs in sd["voltages"].items():
                bus = lbm.get(lname, "").lower()
                idx = ni_lower.get(bus)                       
                if idx is not None:
                    nv[str(idx)] = vs
                else:
                    unmatched += 1
            sd["nv"] = nv
            logger.info("Mapped %d voltages to nodes for %s (%d unmatched)", len(nv), sc, unmatched)
        sim_js=json.dumps({"baseline":{"nv":sim_data["baseline"]["nv"],"tp":sim_data["baseline"]["tx_p"],
            "dt":sim_data["baseline"].get("date",""),"lk":sim_data["baseline"]["loss_kw"]},
            "qp":{"nv":sim_data["qp"]["nv"],"tp":sim_data["qp"]["tx_p"],
            "dt":sim_data["qp"].get("date",""),"lk":sim_data["qp"]["loss_kw"]}})
    n_mv=sum(1 for v in vl_a if v==1); n_lv=sum(1 for v in vl_a if v==2)
    n_loads=sum(hl_a); n_tx=sum(1 for t in et if t==0); total_loads=sum(nloads)
    hrs=[f"{h//2}:{(h%2)*30:02d}" for h in range(48)]

    html=f'''<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Elermore Vale — Power Flow Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&family=Outfit:wght@300;400;600;700&display=swap');
:root{{--bg:#04070d;--pn:#0a1018;--cd:#101a28;--bd:#162240;--tx:#b8c8da;--dm:#4a5e78;--ac:#00d4ff;--aw:#ffd600;--ag:#00e676;--ar:#ff1744;--ao:#ff6b35}}
*{{margin:0;padding:0;box-sizing:border-box}}body{{font-family:'Outfit',sans-serif;background:var(--bg);color:var(--tx);overflow:hidden;height:100vh}}
.D{{display:grid;grid-template-rows:48px 1fr 180px;grid-template-columns:1fr 300px;height:100vh;gap:1px;background:var(--bd)}}
.H{{grid-column:1/-1;background:var(--pn);display:flex;align-items:center;justify-content:space-between;padding:0 16px}}
.HL{{display:flex;align-items:center;gap:12px}}.H h1{{font-family:'IBM Plex Mono',monospace;font-size:.9rem;font-weight:700;color:var(--ac)}}
.VT{{display:flex;background:var(--bg);border-radius:4px;overflow:hidden;border:1px solid var(--bd)}}
.VB{{padding:4px 12px;font-family:'IBM Plex Mono',monospace;font-size:.6rem;font-weight:600;border:none;cursor:pointer;background:0;color:var(--dm);transition:.12s}}.VB.on{{background:#1e3a5f;color:var(--ac)}}.VB:hover:not(.on){{color:var(--tx)}}
.MT{{display:flex;background:var(--bg);border-radius:4px;overflow:hidden;border:1px solid var(--bd)}}
.MB{{padding:4px 12px;font-family:'IBM Plex Mono',monospace;font-size:.6rem;font-weight:600;border:none;cursor:pointer;background:0;color:var(--dm);transition:.12s}}.MB.on{{background:var(--ac);color:var(--bg)}}.MB:hover:not(.on){{color:var(--tx)}}
.PB{{display:flex;align-items:center;gap:8px}}.PBT{{width:28px;height:28px;border-radius:50%;border:2px solid var(--ac);background:0;color:var(--ac);cursor:pointer;display:flex;align-items:center;justify-content:center;font-size:.8rem;transition:.1s}}.PBT:hover{{background:var(--ac);color:var(--bg)}}
.TS{{width:140px;accent-color:var(--ac);cursor:pointer}}.TD{{font-family:'IBM Plex Mono',monospace;font-size:.75rem;color:var(--ac);min-width:40px}}
.SBn{{padding:3px 6px;font-family:'IBM Plex Mono',monospace;font-size:.55rem;border:1px solid var(--bd);background:0;color:var(--dm);border-radius:3px;cursor:pointer}}.SBn.on{{border-color:var(--ac);color:var(--ac)}}
.NP{{background:var(--bg);position:relative}}canvas{{width:100%;height:100%;display:block}}
.SI{{background:var(--pn);padding:12px;overflow-y:auto;display:flex;flex-direction:column;gap:10px;font-size:.8rem}}
.SI h3{{font-family:'IBM Plex Mono',monospace;font-size:.6rem;color:var(--dm);text-transform:uppercase;letter-spacing:1.2px;margin-bottom:2px}}
.SG{{display:grid;grid-template-columns:1fr 1fr;gap:6px}}.SC{{background:var(--cd);border:1px solid var(--bd);border-radius:4px;padding:7px 9px}}
.SC .L{{font-size:.55rem;color:var(--dm);text-transform:uppercase;letter-spacing:.4px}}.SC .V{{font-family:'IBM Plex Mono',monospace;font-size:.95rem;font-weight:700;margin-top:1px}}
.V.w{{color:var(--aw)}}.V.d{{color:var(--ar)}}.V.g{{color:var(--ag)}}
.CB{{height:10px;border-radius:2px;background:linear-gradient(90deg,#1565c0,#42a5f5,#cfd8dc,#ef5350,#b71c1c);margin:3px 0 1px}}
.CBL{{display:flex;justify-content:space-between;font-family:'IBM Plex Mono',monospace;font-size:.5rem;color:var(--dm)}}
.LR{{display:flex;align-items:center;gap:6px;font-size:.72rem;margin:1px 0}}.LD{{width:8px;height:8px;border-radius:50%;flex-shrink:0}}
.BP{{grid-column:1/-1;background:var(--pn);display:grid;grid-template-columns:1fr 1fr;gap:1px}}.CP{{background:var(--bg);padding:2px}}
#pc,#hc{{width:100%;height:100%}}.NS{{display:flex;align-items:center;justify-content:center;height:100%;font-size:.75rem;color:var(--dm);text-align:center;padding:12px}}
.TT{{position:absolute;background:var(--cd);border:1px solid var(--ac);border-radius:4px;padding:6px 10px;font-size:.72rem;pointer-events:none;display:none;z-index:10;font-family:'IBM Plex Mono',monospace;max-width:280px;line-height:1.4}}
.note{{font-size:.65rem;color:#3a4a5e;line-height:1.4;padding:8px;background:var(--bg);border-radius:4px;border:1px solid var(--bd)}}.note b{{color:var(--dm)}}
</style></head><body>
<div class="D">
<div class="H"><div class="HL"><h1>ELERMORE VALE</h1>
<div class="VT"><button class="VB on" data-v="static" onclick="setView('static')">STATIC TOPOLOGY</button>
<button class="VB" data-v="dynamic" onclick="setView('dynamic')">LIVE FLOW</button></div>
<div class="MT" id="mt" style="display:{'flex' if has_sim else 'none'}">
<button class="MB on" data-m="baseline" onclick="setMode('baseline')">BASELINE</button>
<button class="MB" data-m="qp" onclick="setMode('qp')">QP DISPATCH</button></div></div>
<div class="PB" id="pbx" style="display:{'flex' if has_sim else 'none'}">
<button class="PBT" id="pb" onclick="togglePlay()">&#9654;</button>
<input type="range" class="TS" id="ts" min="0" max="47" value="0" oninput="setTime(+this.value)">
<span class="TD" id="td">0:00</span>
<button class="SBn on" data-s="1" onclick="setSpd(1)">1x</button>
<button class="SBn" data-s="2" onclick="setSpd(2)">2x</button>
<button class="SBn" data-s="4" onclick="setSpd(4)">4x</button></div></div>
<div class="NP"><canvas id="cv"></canvas><div class="TT" id="tt"></div></div>
<div class="SI">
<h3>Live Statistics</h3>
<div class="SG">
<div class="SC"><div class="L">V Min</div><div class="V" id="sn">&mdash;</div></div>
<div class="SC"><div class="L">V Max</div><div class="V" id="sx">&mdash;</div></div>
<div class="SC"><div class="L">Violations</div><div class="V" id="sv">&mdash;</div></div>
<div class="SC"><div class="L">TX Power</div><div class="V" id="sp">&mdash;</div></div>
</div>
<h3>Voltage Scale</h3><div class="CB"></div>
<div class="CBL"><span>0.90</span><span>0.94</span><span>1.00</span><span>1.06</span><span>1.10</span></div>
<h3>Topology Legend</h3>
<div class="LR"><div class="LD" style="background:#e74c3c"></div>132 kV Source Bus</div>
<div class="LR"><div class="LD" style="background:#f39c12"></div>11 kV MV Backbone ({n_mv} buses)</div>
<div class="LR"><div class="LD" style="background:#2ecc71"></div>LV Load Bus ({n_loads} buses)</div>
<div class="LR"><div class="LD" style="background:#7f8fa6"></div>LV Junction ({n_lv - n_loads})</div>
<div class="LR" style="margin-top:4px"><div class="LD" style="background:var(--aw);box-shadow:0 0 6px var(--aw)"></div>Particle = power flow direction</div>
<h3>Network</h3>
<div class="SG">
<div class="SC"><div class="L">132/11 kV Sub</div><div class="V">50 MVA</div></div>
<div class="SC"><div class="L">MV Feeder</div><div class="V">{n_mv} buses</div></div>
<div class="SC"><div class="L">Dist. TXs</div><div class="V">{n_tx}</div></div>
<div class="SC"><div class="L">Total Buses</div><div class="V">{len(nl)}</div></div>
<div class="SC"><div class="L">Branches</div><div class="V">{len(ea)}</div></div>
<div class="SC"><div class="L">Total Loads</div><div class="V">{total_loads}</div></div>
</div>
<h3>Voltage Limits (AS 60038)</h3>
<div class="SC"><div class="L">Nominal</div><div class="V">230 V (1.00 p.u.)</div></div>
<div class="SG" style="margin-top:4px">
<div class="SC"><div class="L">Upper</div><div class="V">253 V (+10%)</div></div>
<div class="SC"><div class="L">Lower</div><div class="V">216 V (&#8722;6%)</div></div>
</div>
<div class="note"><b>About:</b> Ported from the Ausgrid Smart Grid Smart City GridLAB-D model.
Elermore Vale, Newcastle NSW &mdash; ~1,785 residential customers across 23 distribution transformers
on a 31.68 km radial 11 kV feeder. Scroll to zoom, drag to pan, hover for bus details.
<br><br><b>Static view:</b> topology coloured by voltage level. <b>Live view:</b> animated power flow with voltage heatmap.</div>
</div>
<div class="BP">
<div class="CP">{"<div id='pc'></div>" if has_sim else "<div class='NS'>Run with --simulate to see substation power flow</div>"}</div>
<div class="CP">{"<div id='hc'></div>" if has_sim else "<div class='NS'>Run with --simulate to see voltage distribution</div>"}</div>
</div></div>
<script>
const NX={json.dumps(nx_a)},NY={json.dumps(ny_a)},VL={json.dumps(vl_a)},
HL={json.dumps(hl_a)},NLd={json.dumps(nloads)},DP={json.dumps(dp_a)},
EA={json.dumps(ea)},EB={json.dumps(eb)},ET={json.dumps(et)},
NM={json.dumps(nl)},hrs={json.dumps(hrs)},
hasSim={'true' if has_sim else 'false'},S={sim_js};
const cv=document.getElementById('cv'),ctx=cv.getContext('2d'),tt=document.getElementById('tt');
let W,H,camX=0,camY=0,zoom=1,drag=false,dx0,dy0,hoverNode=-1;
let mode='baseline',ct=0,playing=false,spd=1,lt=0,gt=0,view='static';
function resize(){{W=cv.parentElement.clientWidth;H=cv.parentElement.clientHeight;cv.width=W*devicePixelRatio;cv.height=H*devicePixelRatio;ctx.setTransform(devicePixelRatio,0,0,devicePixelRatio,0,0)}}
window.addEventListener('resize',resize);resize();
function w2s(wx,wy){{return[(wx-camX)*zoom+W/2,(wy-camY)*zoom+H/2]}}
function s2w(sx,sy){{return[(sx-W/2)/zoom+camX,(sy-H/2)/zoom+camY]}}
cv.addEventListener('wheel',e=>{{e.preventDefault();const[wx,wy]=s2w(e.offsetX,e.offsetY);const f=e.deltaY<0?1.15:1/1.15;zoom*=f;camX=wx-(e.offsetX-W/2)/zoom;camY=wy-(e.offsetY-H/2)/zoom}});
cv.addEventListener('mousedown',e=>{{drag=true;dx0=e.offsetX;dy0=e.offsetY;cv.style.cursor='grabbing'}});
cv.addEventListener('mousemove',e=>{{if(drag){{camX-=(e.offsetX-dx0)/zoom;camY-=(e.offsetY-dy0)/zoom;dx0=e.offsetX;dy0=e.offsetY}}else{{const[wx,wy]=s2w(e.offsetX,e.offsetY);let best=-1,bd=1e9;for(let i=0;i<NX.length;i++){{const d=(NX[i]-wx)**2+(NY[i]-wy)**2;if(d<bd){{bd=d;best=i}}}}const thr=(15/zoom)**2;hoverNode=bd<thr?best:-1;if(hoverNode>=0){{tt.style.display='block';tt.style.left=(e.offsetX+14)+'px';tt.style.top=(e.offsetY-10)+'px';let h='<b>'+NM[hoverNode]+'</b><br>'+['132 kV','11 kV','LV'][VL[hoverNode]];if(NLd[hoverNode]>0)h+='<br>Loads: '+NLd[hoverNode];h+='<br>Depth: '+DP[hoverNode];if(hasSim&&view==='dynamic'){{const nv=S[mode].nv[hoverNode];if(nv)h+='<br>V: <b>'+nv[ct].toFixed(3)+'</b> p.u.'}}tt.innerHTML=h}}else tt.style.display='none'}}}});
cv.addEventListener('mouseup',()=>{{drag=false;cv.style.cursor='default'}});
cv.addEventListener('mouseleave',()=>{{drag=false;tt.style.display='none';hoverNode=-1}});
const DC=VL.map((v,i)=>v===0?'#e74c3c':v===1?'#f39c12':HL[i]?'#2ecc71':'#7f8fa6');
const DS=VL.map((v,i)=>v===0?8:v===1?5:HL[i]?2.8:1.8);
function v2c(v){{if(v<=0)return'#222';const t=Math.max(0,Math.min(1,(v-.90)/.20));if(t<.5){{const u=t/.5;return`rgb(${{21+203*u|0}},${{101+123*u|0}},${{192+32*u|0}})`}}else{{const u=(t-.5)/.5;return`rgb(${{224-41*u|0}},${{224-196*u|0}},${{224-196*u|0}})`}}}}
const NPART=500;let parts=[];for(let i=0;i<NPART;i++)parts.push({{ei:Math.random()*EA.length|0,t:Math.random(),sp:.002+Math.random()*.008}});
function render(){{ctx.clearRect(0,0,W,H);
ctx.lineWidth=Math.max(0.8,zoom/300);ctx.strokeStyle=view==='static'?'#2a4570':'#1e3858';ctx.beginPath();
for(let i=0;i<EA.length;i++)if(ET[i]===1){{const[x1,y1]=w2s(NX[EA[i]],NY[EA[i]]),[x2,y2]=w2s(NX[EB[i]],NY[EB[i]]);ctx.moveTo(x1,y1);ctx.lineTo(x2,y2)}}ctx.stroke();
ctx.lineWidth=Math.max(2,zoom/120);ctx.strokeStyle=view==='static'?'#4080c0':'#2a5a90';ctx.beginPath();
for(let i=0;i<EA.length;i++)if(ET[i]===0){{const[x1,y1]=w2s(NX[EA[i]],NY[EA[i]]),[x2,y2]=w2s(NX[EB[i]],NY[EB[i]]);ctx.moveTo(x1,y1);ctx.lineTo(x2,y2)}}ctx.stroke();
if(view==='dynamic'){{const pr=Math.max(1.5,2.2*zoom/400);ctx.shadowColor='#ffd600';ctx.shadowBlur=3;ctx.fillStyle='rgba(255,214,0,0.9)';for(let p of parts){{const a=EA[p.ei],b=EB[p.ei],px=NX[a]+(NX[b]-NX[a])*p.t,py=NY[a]+(NY[b]-NY[a])*p.t;const[sx,sy]=w2s(px,py);ctx.beginPath();ctx.arc(sx,sy,pr,0,6.28);ctx.fill()}}ctx.shadowBlur=0}}
const isDyn=view==='dynamic'&&hasSim;const nc=isDyn?NX.map((_,i)=>{{const nv=S[mode].nv[i];return nv?v2c(nv[ct]):DC[i]}}):DC;
for(let i=0;i<NX.length;i++){{const[sx,sy]=w2s(NX[i],NY[i]);const r=DS[i]*Math.max(.5,zoom/400);
if(isDyn){{const nv=S[mode].nv[i];if(nv){{const v=nv[ct];if(v>0&&(v<.94||v>1.10)){{const pulse=.3+.25*Math.sin(gt*5);ctx.fillStyle=v<.94?`rgba(21,101,192,${{pulse}})`:`rgba(183,28,28,${{pulse}})`;ctx.beginPath();ctx.arc(sx,sy,r*5,0,6.28);ctx.fill()}}}}}}
ctx.fillStyle=nc[i];ctx.beginPath();ctx.arc(sx,sy,r,0,6.28);ctx.fill()}}
if(hoverNode>=0){{const[sx,sy]=w2s(NX[hoverNode],NY[hoverNode]);ctx.strokeStyle='#00d4ff';ctx.lineWidth=2;ctx.beginPath();ctx.arc(sx,sy,10,0,6.28);ctx.stroke()}}}}
let lf=performance.now();function loop(now){{const dt=(now-lf)/1000;lf=now;gt+=dt;if(view==='dynamic')for(let p of parts){{p.t+=p.sp*dt*60;if(p.t>=1){{p.t=0;p.ei=Math.random()*EA.length|0;p.sp=.002+Math.random()*.008}}}};render();if(playing&&view==='dynamic'&&now-lt>=500/spd){{lt=now;ct=(ct+1)%48;setTime(ct)}};requestAnimationFrame(loop)}}
requestAnimationFrame(loop);
function setView(v){{view=v;document.querySelectorAll('.VB').forEach(b=>b.classList.toggle('on',b.dataset.v===v));const pbx=document.getElementById('pbx');if(pbx)pbx.style.display=(v==='dynamic'&&hasSim)?'flex':'none';const mt=document.getElementById('mt');if(mt)mt.style.display=(v==='dynamic'&&hasSim)?'flex':'none';if(v==='static'){{playing=false;document.getElementById('pb').innerHTML='&#9654;'}}}}
function setTime(t){{ct=t;document.getElementById('ts').value=t;document.getElementById('td').textContent=hrs[t];if(!hasSim)return;const sc=S[mode],nv=sc.nv;const aV=Object.values(nv).map(a=>a[t]).filter(v=>v>.01);if(typeof Plotly!=='undefined'&&document.getElementById('hc')){{try{{Plotly.restyle('hc',{{x:[aV]}},[0])}}catch(e){{}}}}if(typeof Plotly!=='undefined'&&document.getElementById('pc')){{try{{Plotly.restyle('pc',{{x:[[t*.5,t*.5]]}},[2])}}catch(e){{}}}}if(view!=='dynamic')return;if(aV.length){{const mn=Math.min(...aV),mx=Math.max(...aV),vi=aV.filter(v=>v<.94||v>1.10).length;document.getElementById('sn').textContent=mn.toFixed(3);document.getElementById('sx').textContent=mx.toFixed(3);document.getElementById('sv').textContent=vi;document.getElementById('sn').className='V'+(mn<.94?' d':mn<.97?' w':' g');document.getElementById('sx').className='V'+(mx>1.10?' d':mx>1.07?' w':' g');document.getElementById('sv').className='V'+(vi>0?' d':' g')}}const p=sc.tp[t];if(p!==undefined)document.getElementById('sp').textContent=Math.round(p)+' kW'}}
function setMode(m){{mode=m;document.querySelectorAll('.MB').forEach(b=>b.classList.toggle('on',b.dataset.m===m));if(hasSim)setTime(ct)}}
function togglePlay(){{playing=!playing;document.getElementById('pb').innerHTML=playing?'&#9646;&#9646;':'&#9654;';if(playing)lt=performance.now()}}
function setSpd(s){{spd=s;document.querySelectorAll('.SBn').forEach(b=>b.classList.toggle('on',+b.dataset.s===s))}}
if(hasSim){{const bP=S.baseline.tp,qP=S.qp.tp,xH=hrs.map((_,i)=>i*.5);
Plotly.newPlot('pc',[{{x:xH,y:bP,name:'Baseline',line:{{color:'#ff6b35',width:2}},type:'scatter'}},{{x:xH,y:qP,name:'QP',line:{{color:'#00d4ff',width:2}},type:'scatter'}},{{x:[0,0],y:[Math.min(...bP,...qP)*1.1,Math.max(...bP,...qP)*1.1],mode:'lines',line:{{color:'#ffd600',width:1.5,dash:'dot'}},showlegend:false,name:'cur'}}],{{paper_bgcolor:'#04070d',plot_bgcolor:'#0a1018',font:{{family:'IBM Plex Mono',color:'#4a5e78',size:9}},xaxis:{{title:'Hour',gridcolor:'#162240',range:[0,24]}},yaxis:{{title:'kW',gridcolor:'#162240'}},legend:{{x:.02,y:.98,bgcolor:'rgba(0,0,0,0)'}},margin:{{t:6,b:32,l:48,r:6}}}},{{displayModeBar:false,responsive:true}});
Plotly.newPlot('hc',[{{x:[],type:'histogram',marker:{{color:'#00d4ff'}},nbinsx:30}}],{{paper_bgcolor:'#04070d',plot_bgcolor:'#0a1018',font:{{family:'IBM Plex Mono',color:'#4a5e78',size:9}},xaxis:{{title:'V (p.u.)',gridcolor:'#162240',range:[.85,1.15]}},yaxis:{{title:'Count',gridcolor:'#162240'}},margin:{{t:6,b:32,l:36,r:6}},shapes:[{{type:'line',x0:.94,x1:.94,y0:0,y1:1,yref:'paper',line:{{color:'#ff1744',dash:'dash',width:1}}}},{{type:'line',x0:1.10,x1:1.10,y0:0,y1:1,yref:'paper',line:{{color:'#ff1744',dash:'dash',width:1}}}},{{type:'line',x0:1,x1:1,y0:0,y1:1,yref:'paper',line:{{color:'#ffffff22',dash:'dot',width:1}}}}]}},{{displayModeBar:false,responsive:true}});setTime(0)}}
const mnx=Math.min(...NX),mxx=Math.max(...NX),mny=Math.min(...NY),mxy=Math.max(...NY);camX=(mnx+mxx)/2;camY=(mny+mxy)/2;zoom=Math.min(W/(mxx-mnx+.01),H/(mxy-mny+.01))*.88;
</script></body></html>'''

    with open(output_path,"w",encoding="utf-8") as f: f.write(html)
    logger.info("Dashboard: %s (%d KB)",output_path,os.path.getsize(output_path)//1024)
    return output_path

def main():
    p=argparse.ArgumentParser(description="Elermore Vale dual-view power flow dashboard")
    p.add_argument("--glm-dir",default="Elermorevale")
    p.add_argument("--common-dir",default="common")
    p.add_argument("--output",default="elermorevale_dashboard.html")
    p.add_argument("--simulate",action="store_true",help="Run baseline+QP simulations")
    p.add_argument("--profiles",default="profiles/fit_profiles.csv")
    p.add_argument("--day",type=int,default=190,help="Day index (default 190=summer)")
    p.add_argument("--open",action="store_true",help="Auto-open in browser")
    a=p.parse_args()
    G=build_topology(a.glm_dir)
    sd=run_simulation(a.glm_dir,a.common_dir,a.profiles,a.day) if a.simulate else None
    o=generate_dashboard(G,sd,a.output)
    if a.open: webbrowser.open(f"file://{os.path.abspath(o)}")

if __name__=="__main__": main()