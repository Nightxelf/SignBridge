import React, { useEffect, useState } from 'react'

const DEFAULT_BACKEND_URL = 'http://localhost:8000'

function App() {
  const [backendUrl, setBackendUrl] = useState(DEFAULT_BACKEND_URL)
  const [status, setStatus] = useState('unknown')
  const [message, setMessage] = useState('Enter your backend URL and save to test connectivity.')

  useEffect(() => {
    const stored = localStorage.getItem('signbridge_backend_url')
    if (stored) {
      setBackendUrl(stored)
    }
    checkBackend(stored || DEFAULT_BACKEND_URL)
  }, [])

  const saveBackendUrl = async () => {
    localStorage.setItem('signbridge_backend_url', backendUrl)
    if (window.chrome?.storage?.sync) {
      window.chrome.storage.sync.set({backendUrl})
    }
    await checkBackend(backendUrl)
  }

  const checkBackend = async (url) => {
    try {
      const res = await fetch(`${url}/api/health`)
      const data = await res.json()
      if (res.ok) {
        setStatus('online')
        setMessage(`Backend reachable: ${data.status || 'ok'}`)
      } else {
        setStatus('error')
        setMessage(`Backend returned ${res.status}`)
      }
    } catch (err) {
      setStatus('offline')
      setMessage('Unable to reach backend. Make sure it is running.')
    }
  }

  return (
    <div style={{padding: '16px', fontFamily: 'system-ui, sans-serif', width: '320px'}}>
      <h2 style={{marginBottom: '8px'}}>SignBridge</h2>
      <p style={{margin: '8px 0'}}>Live sign language caption overlay for video calls.</p>
      <label style={{display: 'block', marginBottom: '8px'}}>
        Backend URL
        <input
          value={backendUrl}
          onChange={(event) => setBackendUrl(event.target.value)}
          style={{width: '100%', padding: '8px', marginTop: '6px', borderRadius: '4px', border: '1px solid #ccc'}}
        />
      </label>
      <button
        onClick={saveBackendUrl}
        style={{width: '100%', padding: '10px 0', background: '#2563eb', color: 'white', border: 'none', borderRadius: '6px', cursor: 'pointer'}}
      >
        Save and test
      </button>
      <div style={{marginTop: '16px'}}>
        <strong>Status:</strong> {status}
        <p style={{marginTop: '8px', fontSize: '14px', color: '#444'}}>{message}</p>
      </div>
      <hr style={{margin: '16px 0'}} />
      <p style={{fontSize: '13px', color: '#555'}}>
        Open a video call page and click the page overlay to start webcam capture. The caption overlay is handled by the content script.
      </p>
    </div>
  )
}

export default App
