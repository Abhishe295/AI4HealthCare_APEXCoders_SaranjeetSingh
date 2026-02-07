import { Palette } from "lucide-react";

export default function Navbar() {
  return (
    <div className="navbar bg-base-200 shadow-md px-6 border-b border-base-300 z-10">
      <div className="flex-1">
        <a 
          href="/" 
          className="text-xl font-bold bg-gradient-to-r from-primary via-secondary to-accent bg-clip-text text-transparent hover:scale-105 transition-transform duration-300 cursor-pointer"
        >
          MRI-Analyzer
        </a>
      </div>

      <div className="flex-none">
        <a 
          href="/theme" 
          className="btn btn-ghost btn-circle hover:bg-primary/10 hover:text-primary hover:rotate-12 transition-all duration-300"
        >
          <Palette size={22} />
        </a>
      </div>
    </div>
  );
}