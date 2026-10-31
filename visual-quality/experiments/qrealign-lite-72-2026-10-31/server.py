"""Authenticated Q-ReAlign Lite image scorer for an isolated Runpod pod."""
import base64, hmac, io, json, os, time
from http.server import BaseHTTPRequestHandler, HTTPServer
from PIL import Image
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
CKPT='q-future/Q-ReAlign-Lite-4B'
REV='06add46a6bc4a7dfcf0a59a2fb1ed15a6e3d30b7'
LEVELS=['excellent','good','fair','poor','bad']
processor=AutoProcessor.from_pretrained(CKPT,revision=REV)
model=AutoModelForImageTextToText.from_pretrained(CKPT,revision=REV,dtype='auto').to('cuda').eval()
tokens=[processor.tokenizer(' '+x,add_special_tokens=False).input_ids for x in LEVELS]
if len(set(x[0] for x in tokens))!=5 or any(len(x)!=1 for x in tokens):raise RuntimeError('Level tokens not distinct single tokens')
weights=torch.tensor([1.,.75,.5,.25,0.],device='cuda')
class Handler(BaseHTTPRequestHandler):
 def log_message(self,*args): pass
 def reply(self,code,data):
  b=json.dumps(data).encode();self.send_response(code);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(b)));self.end_headers();self.wfile.write(b)
 def allowed(self):return hmac.compare_digest(self.headers.get('Authorization',''),'Bearer '+os.environ['SCORER_TOKEN'])
 def do_GET(self):
  if not self.allowed():return self.reply(401,{'error':'unauthorized'})
  self.reply(200,{'model':CKPT,'revision':REV,'device':torch.cuda.get_device_name(0),'level_tokens':tokens,'torch':torch.__version__})
 def do_POST(self):
  if not self.allowed():return self.reply(401,{'error':'unauthorized'})
  try:
   n=int(self.headers.get('Content-Length','0'))
   if n<=0 or n>10_000_000:raise ValueError('invalid request size')
   request=json.loads(self.rfile.read(n));image=Image.open(io.BytesIO(base64.b64decode(request['image'],validate=True))).convert('RGB')
   started=time.monotonic();result={}
   for task in ('quality','aesthetics'):
    prompt='How would you rate the '+task+' of this image?';stem='The '+task+' of the image is'
    messages=[{'role':'user','content':[{'type':'image'},{'type':'text','text':prompt}]}]
    text=processor.apply_chat_template(messages,add_generation_prompt=True)+stem
    inputs=processor(text=[text],images=[image],return_tensors='pt').to('cuda')
    with torch.inference_mode(): logits=model(**inputs).logits[0,-1,[x[0] for x in tokens]].float();probs=logits.softmax(-1);score=(probs*weights).sum().item()
    result[task]={'native_0_1':score,'score_1_5':1+4*score,'probabilities':probs.tolist(),'input_tokens':int(inputs['input_ids'].shape[-1])}
    del inputs,logits,probs
   result.update({'seconds':round(time.monotonic()-started,3),'peak_vram':torch.cuda.max_memory_allocated()})
   self.reply(200,result)
  except Exception as e:self.reply(500,{'error':type(e).__name__+': '+str(e)[:400]})
print('MODEL_READY',flush=True)
HTTPServer(('0.0.0.0',8888),Handler).serve_forever()
