# /// script
# dependencies = ["datasets>=3.0", "transformers>=4.57.1", "torch>=2.4"]
# ///
"""R6 CPU validation only. NO training, NO paid GPU."""
import ast, hashlib, json, urllib.request
from datasets import load_dataset
from transformers import AutoTokenizer
ROOT="https://raw.githubusercontent.com/steveonw/hackathon/0fdcc52e2238d366c1c1038e26b3071d82602a3e/ternary_pet/"
MODELS=[("granite","r6_granite_seed170141_imdb_hybrid.py",170141,"ibm-granite/granite-4.0-350m","bd8a1497065c0d6ba1ef19af6b0d2b14bacf71c2"),
("smol","r6_smol_seed190027_imdb_hybrid.py",190027,"HuggingFaceTB/SmolLM2-360M-Instruct","a10cc1512eabd3dde888204e902eca88bddb4951")]
for kind,path,seed,model,revision in MODELS:
    src=urllib.request.urlopen(ROOT+path,timeout=60).read().decode()
    ast.parse(src)
    assert f"SEED={seed}" in src and f'SEED=={seed}' in src
    assert 'stanfordnlp/imdb' in src and '"e628166"' in src
    assert 'FINAL_JSON_BEGIN' in src and 'FINAL_JSON_END' in src
    assert 'HfApi(token=token)' in src
    assert 'HF_TOKEN' in src
    assert 'len(docs)==32' in src
    assert 'x>=1223' not in src
    assert 'R5_INVALID' not in src
    print("R6_AST_AND_SCHEMA_OK",kind,seed,flush=True)
    tok=AutoTokenizer.from_pretrained(model,revision=revision)
    docs=[];seen=set();scanned=0
    source=load_dataset("stanfordnlp/imdb",split="test",streaming=True,revision="e628166")
    for rownum,item in enumerate(source,1):
        scanned=rownum
        if rownum>25000:break
        text=item.get("text","")
        docid=str(rownum)+":"+hashlib.sha256(str(text).encode("utf-8")).hexdigest()
        digest=hashlib.sha256(docid.encode("utf-8")).hexdigest()
        if digest in seen:continue
        seen.add(digest)
        if int.from_bytes(bytes.fromhex(digest)[:8],"big")%20!=7:continue
        if not text or not text.strip():continue
        tokens=tok.encode(text,add_special_tokens=False)
        if len(tokens)<516:continue
        docs.append({"row":rownum,"sha":digest})
        if len(docs)==32:break
    assert len(docs)==32,(kind,scanned,len(docs))
    print("R6_IMDB_SELECTOR_OK",kind,json.dumps({"scanned":scanned,"selected":len(docs),"row_first":docs[0]["row"],"row_last":docs[-1]["row"],"hash_first":docs[0]["sha"]}),flush=True)
print("R6_ALL_CPU_PREFLIGHT_OK",flush=True)
