import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { createBrowserRouter, RouterProvider } from 'react-router-dom'
import './index.css'
import App from './App.tsx'
import SnrPage from './pages/SnrPage.tsx'
import SimulatorPage from './pages/SimulatorPage.tsx'
import FslPage from './pages/FslPage.tsx'
import SpikeCiPage from './pages/SpikeCiPage.tsx'

const router = createBrowserRouter([
  { path: '/', element: <App /> },
  { path: '/snr', element: <SnrPage /> },
  { path: '/sim', element: <SimulatorPage /> },
  { path: '/fsl', element: <FslPage /> },
  { path: '/spike-ci', element: <SpikeCiPage /> },
])

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <RouterProvider router={router} />
  </StrictMode>,
)
