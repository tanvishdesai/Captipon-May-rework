import { Globe, Languages } from "lucide-react"
import { Navbar } from "@/components/navbar"
// import { Hero } from "@/components/hero" // Old hero commented out
import { InteractiveHero } from "@/components/interactive-hero" // New hero imported
import { CategoryCard } from "@/components/category-card"
import { TaskList } from "@/components/task-list"

import languageData from "@/data/languages.json"
import taskData from "@/data/tasks.json"

export default function Home() {
  const indianLanguages = languageData.indian.map(lang => lang.name)
  const foreignLanguages = languageData.foreign.map(lang => lang.name)

  return (
    <>
      <Navbar />
      <main className="bg-neutral-50 dark:bg-neutral-950">
        <InteractiveHero /> {/* New hero used here */}
        
        <section id="languages" className="container py-16 space-y-10">
          <div className="text-center space-y-4 max-w-3xl mx-auto">
            <h2 className="text-3xl font-bold tracking-tight bg-gradient-to-r from-purple-700 to-indigo-600 bg-clip-text text-transparent">Language Categories</h2>
            <p className="text-muted-foreground text-lg">
              Explore our multi-lingual caption generation models across different language groups
            </p>
          </div>
          
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mt-6">
            <CategoryCard
              title="Indian Languages"
              description="Caption generation models for languages from the Indian subcontinent"
              href="/indian-languages"
              icon={<Languages size={24} />}
              languages={indianLanguages}
            />
            <CategoryCard
              title="Foreign Languages"
              description="Caption generation models for languages from around the world"
              href="/foreign-languages"
              icon={<Globe size={24} />}
              languages={foreignLanguages}
            />
          </div>
        </section>
        
        <section className="container py-16 space-y-8 bg-gradient-to-b from-neutral-50 to-neutral-100 dark:from-neutral-950 dark:to-neutral-900 rounded-3xl my-8">
          <div className="text-center space-y-3">
            <h2 className="text-3xl font-bold">Execution Status</h2>
            <p className="text-muted-foreground">
              Monitor currently running and completed model training tasks
            </p>
          </div>
          
          <TaskList 
            executingTasks={taskData.executing} 
            completedTasks={taskData.completed} 
          />
        </section>
      </main>
    </>
  )
}
