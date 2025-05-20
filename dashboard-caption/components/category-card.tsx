"use client"

import Link from "next/link"
import { motion } from "framer-motion"
import { ArrowRight } from "lucide-react"

interface CategoryCardProps {
  title: string
  description: string
  href: string
  icon: React.ReactNode
  languages: string[]
}

export function CategoryCard({ title, description, href, icon }: CategoryCardProps) {
  return (
    <motion.div
      whileHover={{ y: -6, boxShadow: "0 8px 32px 0 rgba(124, 58, 237, 0.15)" }}
      transition={{ type: "spring", stiffness: 300, damping: 20 }}
      className="bg-white/60 dark:bg-zinc-900/60 border border-neutral-200 dark:border-zinc-800 rounded-2xl shadow-lg backdrop-blur-md p-0 flex flex-col w-full max-w-xs mx-auto min-h-[260px] relative overflow-hidden group"
    >
      <Link href={href} className="flex flex-col h-full w-full no-underline focus:outline-none">
        {/* Icon */}
        <div className="flex justify-center items-center mt-6 mb-2">
          <span className="text-4xl md:text-5xl text-purple-600 dark:text-purple-400 drop-shadow-lg">{icon}</span>
        </div>
        {/* Title */}
        <h2 className="text-center text-xl font-bold text-neutral-900 dark:text-neutral-100 mb-1 px-4">{title}</h2>
        {/* Description */}
        <p className="text-center text-sm text-neutral-600 dark:text-neutral-400 mb-6 px-6 min-h-[40px]">{description}</p>
        <div className="flex-1" />
        {/* Explore Button */}
        <div className="flex justify-end items-end px-6 pb-6 mt-auto">
          <motion.button
            whileHover={{ scale: 1.07 }}
            className="flex items-center gap-2 bg-purple-600 hover:bg-purple-700 text-white text-sm font-semibold px-4 py-2 rounded-full shadow-md transition-colors focus:outline-none focus:ring-2 focus:ring-purple-400"
            tabIndex={-1}
            aria-label={`Explore ${title}`}
          >
            Explore <ArrowRight size={18} />
          </motion.button>
        </div>
        {/* Glassy overlay for hover effect */}
        <motion.div
          aria-hidden
          className="absolute inset-0 pointer-events-none rounded-2xl group-hover:bg-purple-100/10 group-hover:ring-2 group-hover:ring-purple-400/40 transition-all"
        />
      </Link>
    </motion.div>
  )
} 