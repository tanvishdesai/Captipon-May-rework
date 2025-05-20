import { Navbar } from "@/components/navbar"
import { LanguageCard } from "@/components/language-card"
import languageData from "@/data/languages.json"

export default function IndianLanguagesPage() {
  return (
    <div className="min-h-screen bg-neutral-100 dark:bg-neutral-950">
      <Navbar />
      <main className="container mx-auto px-4 py-24">
        <div className="text-center space-y-6 max-w-3xl mx-auto mb-16">
          <h2 className="text-4xl font-bold tracking-tight">
            <span className="bg-gradient-to-r from-neutral-900 to-neutral-700 dark:from-white dark:to-neutral-300 bg-clip-text text-transparent">
              Indian Languages
            </span>
          </h2>
          <p className="text-neutral-600 dark:text-neutral-400 text-lg">
            Caption generation models for languages from the Indian subcontinent. Select a language to view its performance metrics.
          </p>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-6">
          {languageData.indian.map((language) => (
            <LanguageCard key={language.code} language={language} />
          ))}
        </div>
      </main>
    </div>
  )
} 