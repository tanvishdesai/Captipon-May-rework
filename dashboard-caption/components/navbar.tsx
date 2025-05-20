"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { useState } from "react"
import { ThemeToggle } from "./theme-toggle"

const navLinks = [
  { href: "/", label: "Home" },
  { href: "/indian-languages", label: "Indian Languages" },
  { href: "/foreign-languages", label: "Foreign Languages" },
]

export function Navbar() {
  const pathname = usePathname()
  const [menuOpen, setMenuOpen] = useState(false)

  return (
    <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 shadow-md">
      <div className="container flex h-16 items-center justify-between px-4 md:px-0">
        <Link href="/" className="flex items-center gap-3">
          {/* Logo Placeholder */}
          <span className="inline-block w-8 h-8 bg-primary rounded-full flex items-center justify-center">
            <svg width="20" height="20" fill="none" viewBox="0 0 24 24" stroke="white"><circle cx="12" cy="12" r="10" strokeWidth="2" /></svg>
          </span>
          <span className="text-2xl font-extrabold text-black dark:text-white tracking-tight">
            ML Caption
          </span>
        </Link>
        {/* Desktop Nav */}
        <nav className="hidden md:flex items-center gap-8">
          {navLinks.map(link => (
            <Link
              key={link.href}
              href={link.href}
              className={`text-base font-medium transition-colors hover:text-primary relative ${pathname === link.href ? "text-primary after:absolute after:-bottom-1 after:left-0 after:w-full after:h-0.5 after:bg-primary after:rounded" : "text-muted-foreground"}`}
            >
              {link.label}
            </Link>
          ))}
          <ThemeToggle />
        </nav>
        {/* Mobile Hamburger */}
        <button
          className="md:hidden flex items-center justify-center p-2 rounded focus:outline-none focus:ring-2 focus:ring-primary"
          onClick={() => setMenuOpen(!menuOpen)}
          aria-label="Toggle menu"
        >
          <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            {menuOpen ? (
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" />
            ) : (
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 6h16M4 12h16M4 18h16" />
            )}
          </svg>
        </button>
      </div>
      {/* Mobile Menu */}
      {menuOpen && (
        <nav className="md:hidden bg-background border-t shadow-lg animate-fade-in-down">
          <div className="flex flex-col gap-2 px-4 py-4">
            {navLinks.map(link => (
              <Link
                key={link.href}
                href={link.href}
                className={`text-base font-medium py-2 px-2 rounded transition-colors hover:text-primary ${pathname === link.href ? "text-primary bg-primary/10" : "text-muted-foreground"}`}
                onClick={() => setMenuOpen(false)}
              >
                {link.label}
              </Link>
            ))}
            <div className="mt-2">
              <ThemeToggle />
            </div>
          </div>
        </nav>
      )}
    </header>
  )
} 