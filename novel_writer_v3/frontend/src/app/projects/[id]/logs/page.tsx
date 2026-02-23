"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { generationApi } from "@/lib/api";
import type { AgentLog } from "@/types";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { ScrollArea } from "@/components/ui/scroll-area";
import { toast } from "sonner";
import { SidebarTrigger } from "@/components/ui/sidebar";

export default function LogsPage() {
  const params = useParams();
  const projectId = Number(params.id);
  const [logs, setLogs] = useState<AgentLog[]>([]);
  const [loading, setLoading] = useState(true);
  const [chapterFilter, setChapterFilter] = useState<string>("");
  const [expandedLog, setExpandedLog] = useState<number | null>(null);

  const loadLogs = async () => {
    setLoading(true);
    try {
      const ch = chapterFilter ? parseInt(chapterFilter) : undefined;
      const data = await generationApi.getLogs(projectId, ch);
      setLogs(data);
    } catch (err: any) {
      toast.error(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadLogs();
  }, [projectId]);

  const AGENT_COLORS: Record<string, string> = {
    Mastermind: "bg-purple-500",
    Writer: "bg-blue-500",
    ContinuityCritic: "bg-orange-500",
    StyleCritic: "bg-pink-500",
    Judge: "bg-green-500",
    MemoryAgent: "bg-teal-500",
    OutlineArchitect: "bg-indigo-500",
    StyleAnalyzer: "bg-rose-500",
  };

  return (
    <div className="p-6 max-w-5xl">
      <div className="flex items-center gap-2 mb-6">
        <SidebarTrigger />
        <h1 className="text-2xl font-bold">Agent Logs</h1>
      </div>

      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle>Activity Log</CardTitle>
            <div className="flex items-center gap-2">
              <Label>Chapter:</Label>
              <Input
                type="number"
                value={chapterFilter}
                onChange={(e) => setChapterFilter(e.target.value)}
                placeholder="All"
                className="w-20"
              />
              <Button variant="outline" size="sm" onClick={loadLogs}>
                Filter
              </Button>
              <Button variant="outline" size="sm" onClick={() => { setChapterFilter(""); loadLogs(); }}>
                Reset
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          {loading ? (
            <p className="text-muted-foreground">Loading...</p>
          ) : logs.length === 0 ? (
            <p className="text-muted-foreground">No logs yet.</p>
          ) : (
            <ScrollArea className="h-[600px]">
              <div className="space-y-2">
                {logs.map((log) => (
                  <div
                    key={log.id}
                    className="border rounded p-3 cursor-pointer hover:bg-accent/50 transition-colors"
                    onClick={() => setExpandedLog(expandedLog === log.id ? null : log.id)}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <Badge
                          className={`${AGENT_COLORS[log.agent_name] || "bg-gray-500"} text-white text-xs`}
                        >
                          {log.agent_name}
                        </Badge>
                        <span className="text-sm">{log.action}</span>
                        {log.chapter_number && (
                          <span className="text-xs text-muted-foreground">
                            Ch.{log.chapter_number}
                          </span>
                        )}
                      </div>
                      <span className="text-xs text-muted-foreground">
                        {log.elapsed_seconds.toFixed(1)}s
                      </span>
                    </div>
                    {expandedLog === log.id && (
                      <div className="mt-2 space-y-2 text-xs">
                        <div>
                          <strong>Prompt:</strong>
                          <pre className="bg-muted p-2 rounded mt-1 whitespace-pre-wrap">
                            {log.prompt_preview}
                          </pre>
                        </div>
                        <div>
                          <strong>Response:</strong>
                          <pre className="bg-muted p-2 rounded mt-1 whitespace-pre-wrap">
                            {log.response_preview}
                          </pre>
                        </div>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </ScrollArea>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
