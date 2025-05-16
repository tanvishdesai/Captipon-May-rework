import { Navbar } from "@/components/navbar"
import { LanguageCard } from "@/components/language-card"
import languageData from "@/data/languages.json"

export default function IndianLanguagesPage() {
  return (
    <>
      <Navbar />
      <main className="container py-12">
        <div className="space-y-6 pb-4">
          <h1 className="text-3xl font-bold">Indian Languages</h1>
          <p className="text-muted-foreground max-w-3xl">
            Caption generation models for languages from the Indian subcontinent. Select a language to view its performance metrics.
          </p>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-6">
          {languageData.indian.map((language) => (
            <LanguageCard key={language.code} language={language} />
          ))}
        </div>
      </main>
    </>
  )
} 