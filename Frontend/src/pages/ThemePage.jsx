import React from 'react'
import { useThemeStore } from '../lib/useTheme.js'
import { THEMES } from '../constants/index.js'
import { Palette, Check } from 'lucide-react'
import Navbar from '../components/Navbar.jsx'

const ThemePage = () => {
  const {theme, setTheme} = useThemeStore();

  return (
    <div className='min-h-screen bg-base-100'>
      {/* Header */}
      <Navbar/>

      {/* Theme Selection Section */}
      <div className='container mx-auto px-4 sm:px-6 py-6 sm:py-8 max-w-6xl'>
        <div className='card bg-base-200 shadow-lg mt-16'>
          <div className='card-body p-4 sm:p-6'>
            <div className='flex items-center gap-3 mb-6'>
              <div className='w-8 h-8 bg-primary/20 rounded-lg flex items-center justify-center'>
                <Palette className='w-5 h-5 text-primary' />
              </div>
              <div>
                <h2 className='text-xl font-bold text-base-content'>Theme Selection</h2>
                <p className='text-base-content/70 text-sm'>Choose a color theme for your interface</p>
              </div>
            </div>

            <div className='grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 xl:grid-cols-8 gap-3'>
              {THEMES.map((t) => (
                <button
                  key={t}
                  className={`
                    group relative flex flex-col items-center gap-2 p-3 rounded-xl transition-all duration-300 hover:scale-105 hover:shadow-md
                    ${theme === t ? "ring-2 ring-primary bg-primary/10" : "hover:bg-base-100/50"}
                  `}
                  onClick={() => setTheme(t)}
                >
                  {/* Theme preview colors */}
                  <div className='h-12 w-full relative rounded-lg overflow-hidden border border-base-300' data-theme={t}>
                    <div className='absolute inset-0 grid grid-cols-4 gap-0.5 p-1'>
                      <div className="rounded-sm bg-primary"></div>
                      <div className="rounded-sm bg-secondary"></div>
                      <div className="rounded-sm bg-accent"></div>
                      <div className="rounded-sm bg-neutral"></div>
                    </div>
                  </div>
                  
                  {/* Theme name */}
                  <span className="text-xs font-medium text-center leading-tight">
                    {t.charAt(0).toUpperCase() + t.slice(1)}
                  </span>
                  
                  {/* Selected indicator */}
                  {theme === t && (
                    <div className='absolute -top-1 -right-1 w-5 h-5 bg-primary rounded-full flex items-center justify-center'>
                      <Check className='w-3 h-3 text-primary-content' />
                    </div>
                  )}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default ThemePage