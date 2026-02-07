import { useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'

import ThemePage from './pages/ThemePage'
import HomePage from './pages/HomePage'
import { useThemeStore } from './lib/useTheme'
import {Routes,Route} from 'react-router-dom'
import { Dot } from 'lucide-react'
import DotGridBackground from './components/DotGridBackground'

function App() {
  const {theme} = useThemeStore();

  return (
    <div className='relative h-full w-full' data-theme={theme}>
      <DotGridBackground/>
      <div className='relative z-10'>
        <Routes>
          <Route path = '/' element = {<HomePage/>}/>
          <Route path = '/theme' element={<ThemePage/>}/>
        </Routes>
      </div>
    </div>

  )
}

export default App