"use client";

import { useEffect, useState, useRef, useCallback } from "react";
import { useParams } from "next/navigation";
import { generationApi, outlineApi } from "@/lib/api";
import { createProgressSocket, type ProgressMessage } from "@/lib/websocket";
import type { Chapter, ChapterPlan } from "@/types";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { ScrollArea } from "@/components/ui/scroll-area";
import { toast } from "sonner";
import { SidebarTrigger } from "@/components/ui/sidebar";

export default function GeneratePage() {
  const params = useParams();
  const projectId = Number(params.id);
  const [chapters, setChapters] = useState<Chapter[]>([]);
  const [chapterPlans, setChapterPlans] = useState<ChapterPlan[]>([]);
  const [startChapter, setStartChapter] = useState(1);
  const [endChapter, setEndChapter] = useState<number | undefined>(undefined);
  const [generating, setGenerating] = useState(false);
  const [progress, setProgress] = useState<ProgressMessage[]>([]);
  const [currentProgress, setCurrentProgress] = useState<ProgressMessage | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const progressEndRef = useRef<HTMLDivElement>(null);

  const loadData = useCallback(async () => {
    try {
      const [chs, plans] = await Promise.all([
        generationApi.listChapters(projectId),
        outlineApi.listChapterPlans(projectId),
      ]);
      setChapters(chs);
      setChapterPlans(plans);
      if (plans.length > 0 && !endChapter) {
        setEndChapter(Math.max(...plans.map((p) => p.chapter_number)));
      }
    } catch (err: any) {
      toast.error(err.message);
    }
  }, [projectId, endChapter]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  useEffect(() => {
    progressEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [progress]);

  const handleStart = () => {
    setGenerating(true);
    setProgress([]);

    const ws = createProgressSocket(
      projectId,
      (msg) => {
        setCurrentProgress(msg);
        setProgress((prev) => [...prev, msg]);
        if (msg.stage === "complete" || msg.stage === "approved") {
          loadData();
        }
      },
      () => {
        setGenerating(false);
        loadData();
      },
      () => {
        setGenerating(false);
        toast.error("WebSocket connection error");
      }
    );

    ws.onopen = () => {
      ws.send(
        JSON.stringify({
          action: "start",
          start_chapter: startChapter,
          end_chapter: endChapter,
        })
      );
    };

    wsRef.current = ws;
  };

  const handleCancel = () => {
    if (wsRef.current) {
      wsRef.current.send(JSON.stringify({ action: "cancel" }));
    }
    generationApi.cancel(projectId);
    setGenerating(false);
  };

  const totalPlans = chapterPlans.length;
  const completedChapters = chapters.filter((c) => c.status === "completed").length;
  const overallProgress = totalPlans > 0 ? (completedChapters / totalPlans) * 100 : 0;

  return (
    <div className="p-6 max-w-5xl">
      <div className="flex items-center gap-2 mb-6">
        <SidebarTrigger />
        <h1 className="text-2xl font-bold">Generate Chapters</h1>
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        <div className="lg:col-span-2 space-y-6">
          {/* Controls */}
          <Card>
            <CardHeader>
              <CardTitle>Generation Controls</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <Label>Start Chapter</Label>
                  <Input
                    type="number"
                    min={1}
                    value={startChapter}
                    onChange={(e) => setStartChapter(parseInt(e.target.value) || 1)}
                  />
                </div>
                <div>
                  <Label>End Chapter</Label>
                  <Input
                    type="number"
                    min={startChapter}
                    value={endChapter || ""}
                    onChange={(e) => setEndChapter(parseInt(e.target.value) || undefined)}
                    placeholder="All"
                  />
                </div>
              </div>
              <div className="flex gap-2">
                <Button onClick={handleStart} disabled={generating}>
                  {generating ? "Generating..." : "Start Generation"}
                </Button>
                {generating && (
                  <Button variant="destructive" onClick={handleCancel}>
                    Pause
                  </Button>
                )}
              </div>
            </CardContent>
          </Card>

          {/* Progress */}
          <Card>
            <CardHeader>
              <CardTitle>Progress</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span>Overall: {completedChapters} / {totalPlans} chapters</span>
                  <span>{Math.round(overallProgress)}%</span>
                </div>
                <Progress value={overallProgress} />
              </div>
              {currentProgress && (
                <div className="flex items-center gap-2">
                  <Badge variant="outline">{currentProgress.stage}</Badge>
                  <span className="text-sm">{currentProgress.message}</span>
                </div>
              )}
              <ScrollArea className="h-64 border rounded-lg p-3">
                <div className="space-y-1 text-xs font-mono">
                  {progress.map((p, i) => (
                    <div key={i} className="flex gap-2">
                      <Badge variant="secondary" className="text-xs shrink-0">
                        {p.stage}
                      </Badge>
                      <span>{p.message}</span>
                    </div>
                  ))}
                  <div ref={progressEndRef} />
                </div>
              </ScrollArea>
            </CardContent>
          </Card>
        </div>

        {/* Completed chapters sidebar */}
        <div>
          <Card>
            <CardHeader>
              <CardTitle>Completed Chapters</CardTitle>
            </CardHeader>
            <CardContent>
              <ScrollArea className="h-96">
                <div className="space-y-2">
                  {chapters.length === 0 ? (
                    <p className="text-sm text-muted-foreground">No chapters yet</p>
                  ) : (
                    chapters.map((ch) => (
                      <div
                        key={ch.id}
                        className="flex items-center justify-between p-2 border rounded"
                      >
                        <div>
                          <span className="font-mono text-xs text-muted-foreground mr-1">
                            {ch.chapter_number}.
                          </span>
                          <span className="text-sm">{ch.title}</span>
                        </div>
                        <Badge
                          variant={ch.final_score >= 7 ? "default" : "secondary"}
                        >
                          {ch.final_score}/10
                        </Badge>
                      </div>
                    ))
                  )}
                </div>
              </ScrollArea>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
