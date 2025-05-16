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

export function LanguageCard({ language }: LanguageCardProps) {
  const [open, setOpen] = useState(false)

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <motion.div
          whileHover={{ scale: 1.03 }}
          whileTap={{ scale: 0.97 }}
          className="cursor-pointer"
        >
          <Card className="h-full overflow-hidden hover:border-primary hover:shadow-2xl transition-all border-2 border-white/20 dark:border-zinc-800 bg-white/60 dark:bg-black/40 backdrop-blur-md group relative">
            <CardHeader className="flex flex-row items-center gap-2 pb-2">
              <span className="text-lg">🌐</span>
              <CardTitle>
                <span className="text-xl">{language.name}</span>
                <span className="ml-2 text-xs text-muted-foreground">({language.code})</span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-muted-foreground">Click to view model scores</p>
            </CardContent>
            <div className="absolute inset-0 rounded-xl pointer-events-none group-hover:ring-2 group-hover:ring-blue-400/60 dark:group-hover:ring-cyan-400/60 transition-all" />
          </Card>
        </motion.div>
      </DialogTrigger>
      <DialogContent className="sm:max-w-[600px] bg-white/80 dark:bg-black/80 shadow-2xl border border-zinc-200 dark:border-zinc-800 backdrop-blur-2xl fixed top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2">
        <div className="absolute top-0 left-0 w-full h-2 bg-gradient-to-r from-black to-violet-500 rounded-t-xl" />
        <DialogHeader>
          <DialogTitle>{language.name} ({language.code}) - Model Scores</DialogTitle>
        </DialogHeader>
        <Tabs defaultValue="8k" className="w-full">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="8k">Flickr 8k</TabsTrigger>
            <TabsTrigger value="30k">Flickr 30k</TabsTrigger>
          </TabsList>
          <TabsContent value="8k" className="pt-4">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {["model-1", "model-2", "model-3"].map((model) => (
                <Card key={model} className="overflow-hidden">
                  <CardHeader className="bg-muted py-2">
                    <CardTitle className="text-center text-sm">
                      Model Type {model.split('-')[1]}
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="p-4">
                    <div className="space-y-2">
                      {Object.entries(language.bleu_scores["8k"][model as keyof ModelScores]).map(([key, value]) => (
                        <div key={key} className="flex justify-between">
                          <span className="text-sm font-medium">{key.toUpperCase()}:</span>
                          <span className="text-sm">{value.toFixed(2)}</span>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </TabsContent>
          <TabsContent value="30k" className="pt-4">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {["model-1", "model-2", "model-3"].map((model) => (
                <Card key={model} className="overflow-hidden">
                  <CardHeader className="bg-muted py-2">
                    <CardTitle className="text-center text-sm">
                      Model Type {model.split('-')[1]}
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="p-4">
                    <div className="space-y-2">
                      {Object.entries(language.bleu_scores["30k"][model as keyof ModelScores]).map(([key, value]) => (
                        <div key={key} className="flex justify-between">
                          <span className="text-sm font-medium">{key.toUpperCase()}:</span>
                          <span className="text-sm">{value.toFixed(2)}</span>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </TabsContent>
        </Tabs>
      </DialogContent>
    </Dialog>
  )
} 