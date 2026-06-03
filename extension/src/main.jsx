import React from 'react'
import { createRoot } from 'react-dom/client'
import App from './App'

const container = document.createElement('div')
container.id = 'signbridge-root'
document.body.appendChild(container)

createRoot(container).render(<App />)
