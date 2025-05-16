"use client"

import Link from "next/link"
import { ThemeToggle } from "./theme-toggle"

export function Navbar() {
  return (
    <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container flex h-16 items-center justify-between">
        <Link href="/" className="flex items-center gap-2">
          <span className="text-xl font-bold text-black dark:text-white">
            ML Caption
          </span>
        </Link>
        <nav className="flex items-center gap-6">
          <Link 
            href="/" 
            className="text-sm font-medium transition-colors hover:text-primary"
          >
            Home
          </Link>
          <Link 
            href="/indian-languages" 
            className="text-sm font-medium transition-colors hover:text-primary"
          >
            Indian Languages
          </Link>
          <Link 
            href="/foreign-languages" 
            className="text-sm font-medium transition-colors hover:text-primary"
          >
            Foreign Languages
          </Link>
          <ThemeToggle />
        </nav>
      </div>
    </header>
  )
} 