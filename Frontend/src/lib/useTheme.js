import { create } from 'zustand';

export const useThemeStore = create((set) => ({
  theme: localStorage.getItem('theme') || 'forest', // default or saved
  setTheme: (theme) => {
    localStorage.setItem('theme', theme);
    set({ theme });
    document.documentElement.setAttribute('data-theme', theme); // apply to <html>
  },
  loadTheme: () => {
    const saved = localStorage.getItem('theme');
    if (saved) {
      set({ theme: saved });
      document.documentElement.setAttribute('data-theme', saved);
    }
  }
}));