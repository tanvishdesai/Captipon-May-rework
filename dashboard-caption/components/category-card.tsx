"use client"

import Link from "next/link"
import { motion } from "framer-motion"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { ArrowRight } from "lucide-react"

interface CategoryCardProps {
  title: string
  description: string
  href: string
  icon: React.ReactNode
  languages: string[]
}

export function CategoryCard({ title, description, href, icon, languages }: CategoryCardProps) {
  // Only show the first 6 languages and indicate there are more if needed
  const displayLanguages = languages.slice(0, 6)
  const hasMoreLanguages = languages.length > 6
  
  return (
    <Link href={href} className="block h-full overflow-hidden rounded-lg">
      <motion.div
        whileHover={{ scale: 1.02 }}
        whileTap={{ scale: 0.98 }}
        className="h-full"
      >
        <Card className="h-full flex flex-col overflow-hidden border-2 border-neutral-300/50 dark:border-zinc-700/50 bg-neutral-100/50 dark:bg-neutral-900/50 backdrop-blur-md transition-all group relative shadow-lg hover:shadow-purple-500/30 dark:hover:shadow-purple-800/30">
          <div className="absolute top-0 left-0 w-full h-1.5 bg-gradient-to-r from-purple-500 via-purple-600 to-indigo-600 rounded-t-lg" />
          <CardHeader className="pb-2 flex flex-row items-center justify-between pt-4">
            <div className="flex items-center gap-2 text-purple-600 dark:text-purple-400">{icon}<span className="text-lg">🗂️</span></div>
            <motion.div 
              whileHover={{ x: 5 }}
              className="text-muted-foreground"
            >
              <ArrowRight size={18} />
            </motion.div>
          </CardHeader>
          <CardTitle className="text-xl px-6">{title}</CardTitle>
          <CardDescription className="px-6 text-sm text-neutral-600 dark:text-neutral-400 min-h-[40px]">
            {description}
          </CardDescription>
          <CardContent className="pb-2 px-6 flex-grow">
            <div className="grid grid-cols-2 gap-1.5 mt-2">
              {displayLanguages.map((language) => (
                <motion.span 
                  key={language} 
                  className="inline-flex items-center justify-center text-center rounded-md px-2 py-1 text-xs font-medium bg-neutral-200/70 dark:bg-neutral-800/60 border-neutral-300 dark:border-zinc-700 shadow-sm transition-colors hover:bg-purple-100 dark:hover:bg-purple-900/30 hover:text-purple-700 dark:hover:text-purple-300"
                  whileHover={{ scale: 1.05 }}
                >
                  {language}
                </motion.span>
              ))}
              {hasMoreLanguages && (
                <motion.span 
                  className="inline-flex items-center justify-center text-center rounded-md px-2 py-1 text-xs font-medium bg-neutral-200/70 dark:bg-neutral-800/60 border-neutral-300 dark:border-zinc-700 shadow-sm text-purple-600 dark:text-purple-400"
                  whileHover={{ scale: 1.05 }}
                >
                  +{languages.length - 6} more
                </motion.span>
              )}
            </div>
          </CardContent>
          <CardFooter className="px-6 pb-4">
            <p className="text-sm text-muted-foreground">Click to explore</p>
          </CardFooter>
          <div className="absolute inset-0 rounded-lg pointer-events-none group-hover:ring-2 group-hover:ring-purple-500/70 transition-all" />
        </Card>
      </motion.div>
    </Link>
  )
} 