"use client"

import { useState } from "react"
import { motion } from "framer-motion"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog"

interface BleuScore {
  "bleu-1": number
  "bleu-2": number
  "bleu-3": number
  "bleu-4": number
}

interface ModelScores {
  "model-1": BleuScore
  "model-2": BleuScore
  "model-3": BleuScore
}

interface LanguageData {
  name: string
  code: string
  bleu_scores: {
    "8k": ModelScores
    "30k": ModelScores
  }
}

interface LanguageCardProps {
  language: LanguageData
}

function getBestScores(models: ModelScores): Record<string, string> {
  // Returns a map of BLEU key to model name with the highest score
  const best: Record<string, string> = {}
  const bleuKeys = ["bleu-1", "bleu-2", "bleu-3", "bleu-4"]
  bleuKeys.forEach((bleu) => {
    let max = -Infinity
    let bestModel = ""
    Object.entries(models).forEach(([model, scores]) => {
      if (scores[bleu as keyof BleuScore] > max) {
        max = scores[bleu as keyof BleuScore]
        bestModel = model
      }
    })
    best[bleu] = bestModel
  })
  return best
}

export function LanguageCard({ language }: LanguageCardProps) {
  const [open, setOpen] = useState(false)

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <motion.div
          whileHover={{ scale: 1.04 }}
          whileTap={{ scale: 0.97 }}
          className="cursor-pointer"
        >
          <Card className="h-full bg-gradient-to-br from-blue-50 to-violet-100 dark:from-zinc-800 dark:to-zinc-700 border-0 shadow-lg hover:shadow-2xl transition-all rounded-2xl overflow-hidden">
            <CardHeader className="flex flex-col items-center gap-2 py-6">
              <span className="text-3xl" aria-label="Language globe" role="img">🌐</span>
              <CardTitle className="text-center">
                <span className="text-2xl font-bold text-zinc-900 dark:text-white">{language.name}</span>
                <span className="ml-2 text-base text-zinc-500 dark:text-zinc-400">({language.code})</span>
              </CardTitle>
            </CardHeader>
            <CardContent className="flex flex-col items-center pb-6">
              <span className="text-sm text-zinc-600 dark:text-zinc-300">Click to compare model BLEU scores</span>
            </CardContent>
          </Card>
        </motion.div>
      </DialogTrigger>
      <DialogContent className="sm:max-w-[700px] bg-white dark:bg-zinc-800 shadow-2xl border border-zinc-200 dark:border-zinc-700 backdrop-blur-xl rounded-2xl p-0">
        <div className="w-full h-2 bg-gradient-to-r from-blue-500 via-violet-500 to-fuchsia-500 rounded-t-2xl" />
        <DialogHeader className="px-6 pt-6">
          <DialogTitle className="text-xl font-bold">{language.name} <span className="text-base text-zinc-500 dark:text-zinc-400">({language.code})</span></DialogTitle>
          <span className="text-sm text-zinc-500 dark:text-zinc-400">Model BLEU Score Comparison</span>
        </DialogHeader>
        <div className="px-6 pb-6">
          <Tabs defaultValue="8k" className="w-full">
            <TabsList className="flex gap-2 mb-4">
              <TabsTrigger value="8k">Flickr 8k</TabsTrigger>
              <TabsTrigger value="30k">Flickr 30k</TabsTrigger>
            </TabsList>
            {["8k", "30k"].map((dataset) => {
              const models = language.bleu_scores[dataset as "8k" | "30k"]
              const best = getBestScores(models)
              return (
                <TabsContent key={dataset} value={dataset} className="w-full">
                  <div className="overflow-x-auto">
                    <table className="min-w-full border-separate border-spacing-y-2">
                      <thead>
                        <tr>
                          <th className="text-left px-2 py-2 text-sm font-semibold text-zinc-700 dark:text-zinc-200">BLEU</th>
                          {Object.keys(models).map((model) => (
                            <th key={model} className="text-center px-2 py-2 text-sm font-semibold text-zinc-700 dark:text-zinc-200">
                              Model {model.split('-')[1]}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {["bleu-1", "bleu-2", "bleu-3", "bleu-4"].map((bleu) => (
                          <tr key={bleu}>
                            <td className="px-2 py-2 text-zinc-600 dark:text-zinc-300 font-medium">{bleu.toUpperCase()}</td>
                            {Object.entries(models).map(([model, scores]) => (
                              <td
                                key={model}
                                className={`text-center px-2 py-2 text-base font-semibold rounded-lg transition-colors ${best[bleu] === model ? "bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-300" : "text-zinc-800 dark:text-zinc-100"}`}
                                aria-label={best[bleu] === model ? "Best score" : undefined}
                              >
                                {scores[bleu as keyof BleuScore].toFixed(2)}
                                {best[bleu] === model && (
                                  <span className="ml-1" role="img" aria-label="Best">⭐</span>
                                )}
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </TabsContent>
              )
            })}
          </Tabs>
        </div>
      </DialogContent>
    </Dialog>
  )
} 