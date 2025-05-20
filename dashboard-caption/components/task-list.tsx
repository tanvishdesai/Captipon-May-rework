"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Clock, CheckCircle, Loader2, CalendarCheck2 } from "lucide-react"

interface Task {
  id: string
  description: string
  startTime: string
  estimatedEndTime?: string
  endTime?: string
  progress?: number
  result?: string
}

interface TaskListProps {
  executingTasks: Task[]
  completedTasks: Task[]
}



export function TaskList({ executingTasks, completedTasks }: TaskListProps) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6 w-full">
      {/* Active Tasks */}
      <section>
        <Card className="h-full shadow-lg border-0 bg-white/80 dark:bg-zinc-900/80">
          <CardHeader className="border-b border-neutral-200/60 dark:border-zinc-700/60 bg-gradient-to-r from-purple-500/10 to-indigo-500/10 pb-4">
            <CardTitle className="flex items-center gap-2.5 text-lg font-semibold">
              <div className="flex items-center justify-center w-8 h-8 rounded-full bg-purple-100 dark:bg-purple-900/20">
                <Loader2 className="h-5 w-5 text-purple-600 animate-spin" aria-hidden="true" />
              </div>
              <span className="text-neutral-800 dark:text-neutral-200">Active Tasks</span>
            </CardTitle>
          </CardHeader>
          <CardContent className="pt-6 px-4">
            {executingTasks.length === 0 ? (
              <div className="text-center text-neutral-500 dark:text-neutral-400 py-8">
                <span>No active tasks</span>
              </div>
            ) : (
              <ul className="space-y-4">
                {executingTasks.map((task) => (
                  <li key={task.id} className="rounded-lg bg-neutral-100/70 dark:bg-zinc-800/70 p-4 shadow-sm flex items-center gap-3 border border-neutral-200/60 dark:border-zinc-700/60">
                    <Clock className="h-5 w-5 text-purple-500 flex-shrink-0" aria-hidden="true" />
                    <span className="font-medium text-neutral-900 dark:text-neutral-100">{task.description}</span>
                  </li>
                ))}
              </ul>
            )}
          </CardContent>
        </Card>
      </section>

      {/* Completed Tasks */}
      <section>
        <Card className="h-full shadow-lg border-0 bg-white/80 dark:bg-zinc-900/80">
          <CardHeader className="border-b border-neutral-200/60 dark:border-zinc-700/60 bg-gradient-to-r from-emerald-400/10 to-yellow-300/10 pb-4">
            <CardTitle className="flex items-center gap-2.5 text-lg font-semibold">
              <div className="flex items-center justify-center w-8 h-8 rounded-full bg-emerald-100 dark:bg-emerald-900/20">
                <CalendarCheck2 className="h-5 w-5 text-emerald-600" aria-hidden="true" />
              </div>
              <span className="text-neutral-800 dark:text-neutral-200">Completed Tasks</span>
            </CardTitle>
          </CardHeader>
          <CardContent className="pt-6 px-4">
            {completedTasks.length === 0 ? (
              <div className="text-center text-neutral-500 dark:text-neutral-400 py-8">
                <span>No completed tasks</span>
              </div>
            ) : (
              <ul className="space-y-4">
                {completedTasks.map((task) => (
                  <li key={task.id} className="rounded-lg bg-neutral-100/70 dark:bg-zinc-800/70 p-4 shadow-sm flex flex-col gap-2 border border-neutral-200/60 dark:border-zinc-700/60">
                    <div className="flex items-center gap-3">
                      <CheckCircle className="h-5 w-5 text-emerald-500 flex-shrink-0" aria-hidden="true" />
                      <span className="font-medium text-neutral-900 dark:text-neutral-100">{task.description}</span>
                    </div>
                    {task.result && (
                      <div className="text-sm text-emerald-700 dark:text-emerald-400 font-semibold ml-8">{task.result}</div>
                    )}
                  </li>
                ))}
              </ul>
            )}
          </CardContent>
        </Card>
      </section>
    </div>
  )
} 