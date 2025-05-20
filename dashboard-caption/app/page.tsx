import { Globe, Languages } from "lucide-react"
import { Navbar } from "@/components/navbar"
import { InteractiveHero } from "@/components/interactive-hero" // New hero imported
import { CategoryCard } from "@/components/category-card"
import { TaskList } from "@/components/task-list"

import languageData from "@/data/languages.json"
import taskData from "@/data/tasks.json"

export default function Home() {
  const indianLanguages = languageData.indian.map(lang => lang.name)
  const foreignLanguages = languageData.foreign.map(lang => lang.name)

  return (
    <div className="min-h-screen bg-neutral-100 dark:bg-neutral-950">
      <Navbar />
      <main>
        {/* Hero Section */}
        <section className="relative">
          <InteractiveHero />
          
        </section>

        {/* Language Categories */}
        <section id="languages" className="container mx-auto px-4 py-24 scroll-mt-20">
          <div className="text-center space-y-6 max-w-3xl mx-auto mb-16">
            <h2 className="text-4xl font-bold tracking-tight">
              <span className="bg-gradient-to-r from-neutral-900 to-neutral-700 dark:from-white dark:to-neutral-300 bg-clip-text text-transparent">
                Language Categories
              </span>
            </h2>
            <p className="text-neutral-600 dark:text-neutral-400 text-lg">
              Explore our comprehensive collection of multi-lingual caption generation models
            </p>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8 lg:gap-12">
            <CategoryCard
              title="Indian Languages"
              description="Generate captions in languages from the rich linguistic diversity of the Indian subcontinent"
              href="/indian-languages"
              icon={<Languages className="text-neutral-900 dark:text-white" size={32} />}
              languages={indianLanguages}
            />
            <CategoryCard
              title="Foreign Languages"
              description="Create captions in major world languages with our advanced AI models"
              href="/foreign-languages"
              icon={<Globe className="text-neutral-900 dark:text-white" size={32} />}
              languages={foreignLanguages}
            />
          </div>
        </section>

        {/* Execution Status */}
        <section className="container mx-auto px-4 py-24">
          <div className="bg-white/90 dark:bg-neutral-900/90 rounded-3xl shadow-2xl backdrop-blur-sm">
            <div className="p-8 lg:p-12 space-y-12">
              <div className="text-center space-y-4">
                <h2 className="text-3xl font-bold tracking-tight text-neutral-900 dark:text-white">
                  Real-time Execution Status
                </h2>
                <p className="text-neutral-600 dark:text-neutral-400 text-lg max-w-2xl mx-auto">
                  Track the progress of model training tasks and view completed operations
                </p>
              </div>
              <TaskList 
                executingTasks={taskData.executing} 
                completedTasks={taskData.completed} 
              />
            </div>
          </div>
        </section>
      </main>
    </div>
  )
}
