const $ = id => document.getElementById(id);
const chat = $('chat');
let messages = JSON.parse(localStorage.getItem('ds4.messages') || '[]');
let controller = null;

function showPerformance(bubble, message) {
  if (!message.performance) return;
  let stats = bubble.querySelector('.performance');
  if (!stats) { stats = document.createElement('div'); stats.className='performance'; bubble.appendChild(stats); }
  const p=message.performance, parts=[];
  if (p.promptTokens !== undefined) parts.push(`Input: ${p.promptTokens} token`);
  if (p.completionTokens !== undefined) parts.push(`Output: ${p.completionTokens} token`);
  if (p.firstTokenSeconds !== undefined) parts.push(`Primo token: ${p.firstTokenSeconds.toFixed(2)} s`);
  if (p.totalSeconds !== undefined) parts.push(`Totale: ${p.totalSeconds.toFixed(2)} s`);
  if (p.generationTps !== undefined) parts.push(`Generazione stimata: ${p.generationTps.toFixed(2)} t/s`);
  stats.textContent=parts.join(' · ');
}

function render() {
  chat.innerHTML = '';
  if (!messages.length) chat.innerHTML = '<div class="empty">Come posso aiutarti?</div>';
  for (const message of messages) addBubble(message);
  scrollTo(0, document.body.scrollHeight);
}
function addBubble(message) {
  const div = document.createElement('div');
  div.className = `message ${message.role}`;
  if (message.reasoning) {
    const reasoning = document.createElement('div');
    reasoning.className = 'reasoning'; reasoning.textContent = message.reasoning;
    div.appendChild(reasoning);
  }
  const content = document.createElement('div'); content.textContent = message.content || '';
  div.appendChild(content); showPerformance(div,message); chat.appendChild(div); return {div, content};
}
function save() { localStorage.setItem('ds4.messages', JSON.stringify(messages)); }
function settings() { return JSON.parse(localStorage.getItem('ds4.settings') || '{}'); }

async function send() {
  const prompt = $('prompt').value.trim(); if (!prompt || controller) return;
  const cfg = settings();
  messages.push({role:'user', content:prompt}); $('prompt').value='';
  const assistant = {role:'assistant', content:'', reasoning:''}; messages.push(assistant);
  render(); const bubble = chat.lastElementChild; const content = bubble.lastElementChild;
  controller = new AbortController(); $('sendButton').disabled=true; $('stopButton').hidden=false;
  const started=performance.now(); let firstToken=null;
  try {
    const response = await fetch('/v1/chat/completions', {
      method:'POST', signal:controller.signal,
      headers:{'Content-Type':'application/json', ...(cfg.apiKey ? {'Authorization':`Bearer ${cfg.apiKey}`} : {})},
      body:JSON.stringify({model:cfg.model || 'deepseek-v4-flash', messages:messages.slice(0,-1).map(({role,content})=>({role,content})), stream:true, stream_options:{include_usage:true}, temperature:Number(cfg.temperature ?? 1), max_tokens:Number(cfg.maxTokens ?? 4096), think:cfg.thinking ?? true})
    });
    if (!response.ok) throw new Error((await response.text()) || `HTTP ${response.status}`);
    const reader=response.body.getReader(), decoder=new TextDecoder(); let buffer='';
    while (true) {
      const {value,done}=await reader.read(); if(done) break; buffer+=decoder.decode(value,{stream:true});
      const lines=buffer.split('\n'); buffer=lines.pop();
      for (const line of lines) {
        if (!line.startsWith('data:')) continue;
        const data=line.slice(5).trim(); if (!data || data==='[DONE]') continue;
        const event=JSON.parse(data), delta=event.choices?.[0]?.delta || {};
        if (event.usage) assistant.performance={promptTokens:event.usage.prompt_tokens, completionTokens:event.usage.completion_tokens};
        if (firstToken===null && (delta.content || delta.reasoning_content || delta.reasoning)) firstToken=performance.now();
        assistant.content += delta.content || ''; assistant.reasoning += delta.reasoning_content || delta.reasoning || '';
        content.textContent=assistant.content;
        let reasoning=bubble.querySelector('.reasoning');
        if (assistant.reasoning && !reasoning) { reasoning=document.createElement('div'); reasoning.className='reasoning'; bubble.insertBefore(reasoning,content); }
        if (reasoning) reasoning.textContent=assistant.reasoning;
        scrollTo(0,document.body.scrollHeight);
      }
    }
    const ended=performance.now();
    assistant.performance={...assistant.performance,totalSeconds:(ended-started)/1000};
    if (firstToken!==null) {
      assistant.performance.firstTokenSeconds=(firstToken-started)/1000;
      const tokens=assistant.performance.completionTokens, seconds=(ended-firstToken)/1000;
      if (tokens>1 && seconds>0) assistant.performance.generationTps=(tokens-1)/seconds;
    }
    showPerformance(bubble,assistant);
    save(); $('status').textContent='online';
  } catch (error) {
    if (error.name !== 'AbortError') { assistant.content=`Errore: ${error.message}`; content.textContent=assistant.content; $('status').textContent='errore'; }
    save();
  } finally { controller=null; $('sendButton').disabled=false; $('stopButton').hidden=true; }
}

$('sendButton').onclick=send; $('stopButton').onclick=()=>controller?.abort();
$('prompt').onkeydown=e=>{ if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();send()} };
$('clearButton').onclick=()=>{messages=[];save();render()};
$('settingsButton').onclick=()=>{const c=settings(); for(const k of ['apiKey','model','temperature','maxTokens']) if(c[k]!==undefined) $(k).value=c[k]; $('thinking').checked=c.thinking??true; $('settings').showModal()};
$('saveSettings').onclick=()=>localStorage.setItem('ds4.settings',JSON.stringify({apiKey:$('apiKey').value,model:$('model').value,temperature:$('temperature').value,maxTokens:$('maxTokens').value,thinking:$('thinking').checked}));
fetch('/v1/models').then(r=>{ $('status').textContent=r.ok?'online':(r.status===401?'chiave richiesta':'offline') }).catch(()=>$('status').textContent='offline');
render();
