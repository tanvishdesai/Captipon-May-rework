"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { Clock, CheckCircle } from "lucide-react"

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
  const formatDate = (dateString: string) => {
    const date = new Date(dateString)
    return new Intl.DateTimeFormat("en-US", {
      month: "short",
      day: "numeric",
      hour: "numeric",
      minute: "numeric",
    }).format(date)
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
      <Card className="overflow-hidden border-2 border-neutral-300/50 dark:border-zinc-700/50 bg-neutral-100/50 dark:bg-neutral-900/50 backdrop-blur-md shadow-lg group relative transition-all hover:shadow-purple-500/30 dark:hover:shadow-purple-800/30">
        <div className="absolute top-0 left-0 w-full h-1.5 bg-gradient-to-r from-purple-500 via-purple-600 to-indigo-600 rounded-t-lg" />
        <CardHeader className="pt-4">
          <CardTitle className="flex items-center gap-2">
            <Clock className="h-5 w-5 text-purple-600 dark:text-purple-400" />
            <span>Currently Executing</span>
            <span className="text-lg">⏳</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {executingTasks.length === 0 ? (
              <p className="text-sm text-neutral-600 dark:text-neutral-400">No tasks currently running</p>
            ) : (
              executingTasks.map((task) => (
                <div key={task.id} className="space-y-2 border-b border-neutral-200/80 dark:border-zinc-700/80 pb-3 last:border-0">
                  <div className="flex justify-between">
                    <p className="font-medium text-neutral-800 dark:text-neutral-200">{task.description}</p>
                    <span className="text-xs text-purple-600 dark:text-purple-400">
                      {task.progress}%
                    </span>
                  </div>
                  <Progress value={task.progress} className="h-2 bg-gradient-to-r from-purple-500 via-purple-600 to-indigo-600 shadow-md" />
                  <div className="flex justify-between text-xs text-neutral-500 dark:text-neutral-400">
                    <span>Started: {formatDate(task.startTime)}</span>
                    {task.estimatedEndTime && (
                      <span>Est. completion: {formatDate(task.estimatedEndTime)}</span>
                    )}
                  </div>
                </div>
              ))
            )}
          </div>
        </CardContent>
        <div className="absolute inset-0 rounded-lg pointer-events-none group-hover:ring-2 group-hover:ring-purple-500/70 transition-all" />
      </Card>
      <Card className="overflow-hidden border-2 border-neutral-300/50 dark:border-zinc-700/50 bg-neutral-100/50 dark:bg-neutral-900/50 backdrop-blur-md shadow-lg group relative transition-all hover:shadow-pink-500/30 dark:hover:shadow-orange-800/30">
        <div className="absolute top-0 left-0 w-full h-1.5 bg-gradient-to-r from-pink-500 via-orange-500 to-yellow-400 rounded-t-lg" />
        <CardHeader className="pt-4">
          <CardTitle className="flex items-center gap-2">
            <CheckCircle className="h-5 w-5 text-emerald-600 dark:text-emerald-400" />
            <span>Completed Executions</span>
            <span className="text-lg">✅</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {completedTasks.length === 0 ? (
              <p className="text-sm text-neutral-600 dark:text-neutral-400">No completed tasks</p>
            ) : (
              completedTasks.map((task) => (
                <div key={task.id} className="space-y-2 border-b border-neutral-200/80 dark:border-zinc-700/80 pb-3 last:border-0">
                  <div className="flex justify-between">
                    <p className="font-medium text-neutral-800 dark:text-neutral-200">{task.description}</p>
                    <span className="text-xs text-emerald-600 dark:text-emerald-400">{task.result}</span>
                  </div>
                  <div className="flex justify-between text-xs text-neutral-500 dark:text-neutral-400">
                    <span>Started: {formatDate(task.startTime)}</span>
                    {task.endTime && (
                      <span>Completed: {formatDate(task.endTime)}</span>
                    )}
                  </div>
                </div>
              ))
            )}
          </div>
        </CardContent>
        <div className="absolute inset-0 rounded-lg pointer-events-none group-hover:ring-2 group-hover:ring-pink-500/70 dark:group-hover:ring-orange-500/70 transition-all" />
      </Card>
    </div>
  )
} 