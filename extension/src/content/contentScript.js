// Minimal content script that injects an overlay and captures webcam frames when user clicks the overlay.
(function(){
  if (window.__signbridge_injected) return;
  window.__signbridge_injected = true;

  const overlay = document.createElement('div')
  overlay.id = 'signbridge-overlay'
  Object.assign(overlay.style, {
    position: 'fixed',
    right: '12px',
    bottom: '12px',
    zIndex: 2147483647,
    background: 'rgba(0,0,0,0.75)',
    color: 'white',
    padding: '10px',
    borderRadius: '10px',
    fontSize: '13px',
    width: '240px',
    boxShadow: '0 0 16px rgba(0,0,0,0.45)'
  })
  overlay.innerHTML = '<strong>SignBridge</strong><br><span id="signbridge-status">click to start</span>'
  document.body.appendChild(overlay)

  let stream = null
  let capturing = false

  const statusText = document.createElement('div')
  statusText.id = 'signbridge-status'
  statusText.style.marginTop = '6px'
  statusText.innerText = 'click to start'
  overlay.appendChild(statusText)

  overlay.addEventListener('click', async ()=>{
    if (!capturing){
      try{
        stream = await navigator.mediaDevices.getUserMedia({video: true, audio: false})
      }catch(err){
        statusText.innerText = 'Camera access denied'
        return
      }
      statusText.innerText = 'Capturing — looking for hand...'
      capturing = true
      startLoop(stream)
    } else {
      capturing = false
      statusText.innerText = 'Stopped'
      if (stream){
        stream.getTracks().forEach(t=>t.stop())
      }
    }
  })

  function startLoop(stream){
    const video = document.createElement('video')
    video.autoplay = true
    video.srcObject = stream
    video.play()
    const canvas = document.createElement('canvas')
    const ctx = canvas.getContext('2d')

    async function tick(){
      if (!capturing) return
      if (video.readyState >= 2){
        canvas.width = video.videoWidth
        canvas.height = video.videoHeight
        ctx.drawImage(video, 0, 0)
        const blob = await new Promise(res=>canvas.toBlob(res,'image/jpeg',0.7))
        if (blob){
          try{
            const backendUrl = await getBackendUrl()
            const form = new FormData()
            form.append('file', blob, 'frame.jpg')
            fetch(`${backendUrl}/api/extract`, {method:'POST', body: form})
              .then(r=>r.json()).then(j=>{
                if (j.prediction){
                  showCaption(j.prediction, j.confidence)
                }
                if (j.bbox){
                  statusText.innerText = `Capturing — hand detected (${(j.confidence*100).toFixed(0)}%)`
                  drawBox(j.bbox)
                } else {
                  statusText.innerText = 'Capturing — no hand detected'
                  hideBox()
                }
              }).catch(()=>{
                statusText.innerText = 'Backend request failed'
              })
          }catch(e){
            console.warn('SignBridge backend request failed', e)
            statusText.innerText = 'Backend request failed'
          }
        }
      }
      setTimeout(tick, 300)
    }
    tick()
  }

  let boxEl = null
  function ensureBox(){
    if (!boxEl){
      boxEl = document.createElement('div')
      Object.assign(boxEl.style, {
        position: 'absolute',
        border: '2px solid #4ade80',
        boxShadow: '0 0 0 1px rgba(74, 222, 128, 0.5)',
        pointerEvents: 'none',
        zIndex: 2147483647,
        display: 'none'
      })
      document.body.appendChild(boxEl)
    }
  }

  function getTargetVideo(){
    const videos = Array.from(document.querySelectorAll('video'))
    const visible = videos.filter(v=>v.offsetParent !== null && v.videoWidth > 0 && v.videoHeight > 0)
    if (visible.length){
      return visible.sort((a,b)=>b.clientWidth*b.clientHeight - a.clientWidth*a.clientHeight)[0]
    }
    return videos[0] || null
  }

  function drawBox(bbox){
    ensureBox()
    const video = getTargetVideo()
    const [x1, y1, x2, y2] = bbox
    let left, top, width, height
    if (video){
      const rect = video.getBoundingClientRect()
      left = rect.left + x1 * rect.width
      top = rect.top + y1 * rect.height
      width = Math.max(2, (x2 - x1) * rect.width)
      height = Math.max(2, (y2 - y1) * rect.height)
    } else {
      left = Math.max(0, x1 * window.innerWidth)
      top = Math.max(0, y1 * window.innerHeight)
      width = Math.min(window.innerWidth - left, Math.max(2, (x2 - x1) * window.innerWidth))
      height = Math.min(window.innerHeight - top, Math.max(2, (y2 - y1) * window.innerHeight))
    }
    Object.assign(boxEl.style, {
      display: 'block',
      left: `${left}px`,
      top: `${top}px`,
      width: `${width}px`,
      height: `${height}px`,
      opacity: '0.9'
    })
  }

  function hideBox(){
    if (boxEl){
      boxEl.style.display = 'none'
    }
  }

  async function getBackendUrl(){
    return new Promise((resolve) => {
      if (window.chrome?.storage?.sync){
        window.chrome.storage.sync.get({backendUrl: 'http://localhost:8000'}, (result) => {
          resolve(result.backendUrl || 'http://localhost:8000')
        })
      } else {
        resolve(localStorage.getItem('signbridge_backend_url') || 'http://localhost:8000')
      }
    })
  }

  let captionEl = null
  function showCaption(text, conf){
    if (!captionEl){
      captionEl = document.createElement('div')
      Object.assign(captionEl.style, {
        position: 'fixed',
        left: '12px',
        bottom: '80px',
        zIndex: 2147483646,
        background: 'rgba(0,0,0,0.75)',
        color: 'white',
        padding: '10px 14px',
        borderRadius: '12px',
        fontSize: '16px',
        maxWidth: '320px',
        overflowWrap: 'break-word',
        boxShadow: '0 0 16px rgba(0,0,0,0.35)'
      })
      document.body.appendChild(captionEl)
    }
    captionEl.innerText = `${text} (${(conf*100).toFixed(0)}%)`
  }

})()
