"use client"

import { motion, useMotionValue, useTransform } from "framer-motion"
import { useRef } from "react"

export function InteractiveHero() {
  const ref = useRef<HTMLDivElement>(null)
  const mouseX = useMotionValue(0)
  const mouseY = useMotionValue(0)

  const handleMouseMove = (event: React.MouseEvent<HTMLDivElement>) => {
    if (ref.current) {
      const rect = ref.current.getBoundingClientRect()
      // Calculate mouse position relative to the center of the element
      mouseX.set(event.clientX - rect.left - rect.width / 2)
      mouseY.set(event.clientY - rect.top - rect.height / 2)
    }
  }

  const handleMouseLeave = () => {
    mouseX.set(0)
    mouseY.set(0)
  }

  // Reduced rotation for a more subtle effect
  const rotateX = useTransform(mouseY, [-250, 250], [5, -5]) // Max 5 degrees rotation
  const rotateY = useTransform(mouseX, [-250, 250], [-5, 5]) // Max 5 degrees rotation
  
  // For subtle gradient movement (center is 50%)
  const gradientCenterX = useTransform(mouseX, [-250, 250], [40, 60]) // Moves between 40% and 60%
  const gradientCenterY = useTransform(mouseY, [-250, 250], [40, 60]) // Moves between 40% and 60%

  return (
    <motion.section
      ref={ref}
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
      className="relative h-[calc(100vh-80px)] min-h-[500px] md:min-h-[600px] flex items-center justify-center overflow-hidden bg-neutral-100 dark:bg-neutral-950 text-neutral-800 dark:text-white select-none"
      style={{
        perspective: "1000px", // For 3D effect on the text block
      }}
    >
      {/* Clean Background Element (Light Mode) */}
      <motion.div
        className="absolute inset-0 z-0"
        style={{
          backgroundImage: `radial-gradient(circle at ${gradientCenterX}% ${gradientCenterY}%, rgba(240, 240, 240, 0.5), rgba(250, 250, 250, 0) 70%)`,
        }}
      />
      
      {/* Dark mode specific gradient overlay */}
      <motion.div
        className="absolute inset-0 z-0 dark:opacity-100 opacity-0" // Show only in dark mode
        style={{
          backgroundImage: `radial-gradient(circle at ${gradientCenterX}% ${gradientCenterY}%, rgba(50, 50, 50, 0.5), rgba(23, 23, 23, 0) 60%)`,
          transition: "background-image 0.1s ease-out, opacity 0.3s ease-in-out", // Smooth transition for gradient
        }}
      />

      {/* Light mode geometric patterns */}
      <motion.div
        className="absolute inset-0 z-0 opacity-40 dark:opacity-0"
        style={{
          backgroundImage: "radial-gradient(circle at center, rgba(126, 34, 34, 0.3) 2px, transparent 2px)",
          backgroundSize: "32px 32px",
          translateX: useTransform(mouseX, [-250, 250], [-5, 5]),
          translateY: useTransform(mouseY, [-250, 250], [-5, 5]),
        }}
      />

      {/* Dark mode constellation pattern */}
      <motion.div
        className="absolute inset-0 z-0 opacity-0 dark:opacity-30"
        style={{
          backgroundImage: "radial-gradient(circle at center, rgba(170, 170, 170, 0.8) 2px, transparent 2px)",
          backgroundSize: "40px 40px",
          translateX: useTransform(mouseX, [-250, 250], [5, -5]),
          translateY: useTransform(mouseY, [-250, 250], [5, -5]),
        }}
      />

      <motion.div
        className="z-10 text-center p-6 max-w-3xl"
        style={{
          rotateX,
          rotateY,
          transition: "transform 0.05s ease-out",
        }}
      >
        <motion.h1
          className="text-5xl sm:text-6xl md:text-7xl roboto-hero bg-clip-text text-transparent bg-gradient-to-br from-black via-black to-black dark:from-white dark:via-white dark:to-white"
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.2, ease: "easeOut" }}
        >
          AI-Powered Captions
        </motion.h1>
        <motion.p
          className="mt-6 text-lg md:text-xl text-neutral-600 dark:text-neutral-300 max-w-xl mx-auto"
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.4, ease: "easeOut" }}
        >
          Generate insightful and accurate image captions across a multitude of languages.
          Explore our advanced models and track their performance.
        </motion.p>
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.6, ease: "easeOut" }}
          className="mt-10 flex justify-center"
        >
          <motion.button
            whileHover={{ scale: 1.05, boxShadow: "0px 0px 20px rgba(120, 54, 204, 0.5)" }}
            whileTap={{ scale: 0.95 }}
            className="px-8 py-3 bg-black text-white dark:bg-white dark:text-black font-semibold rounded-lg shadow-md hover:shadow-purple-500/40 dark:hover:shadow-purple-500/50 transition-all duration-300"
            onClick={() => document.getElementById('languages')?.scrollIntoView({ behavior: 'smooth' })}
          >
            Explore Languages
          </motion.button>
        </motion.div>
      </motion.div>
    </motion.section>
  )
} 