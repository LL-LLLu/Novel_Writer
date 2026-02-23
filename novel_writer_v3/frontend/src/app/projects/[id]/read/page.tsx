"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { generationApi, projectsApi } from "@/lib/api";
import type { Chapter, Project } from "@/types";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { toast } from "sonner";
import { SidebarTrigger } from "@/components/ui/sidebar";

export default function ReadPage() {
  const params = useParams();
  const projectId = Number(params.id);
  const [project, setProject] = useState<Project | null>(null);
  const [chapters, setChapters] = useState<Chapter[]>([]);
  const [selectedChapter, setSelectedChapter] = useState<Chapter | null>(null);

  useEffect(() => {
    const load = async () => {
      try {
        const [p, chs] = await Promise.all([
          projectsApi.get(projectId),
          generationApi.listChapters(projectId),
        ]);
        setProject(p);
        setChapters(chs);
        if (chs.length > 0) setSelectedChapter(chs[0]);
      } catch (err: any) {
        toast.error(err.message);
      }
    };
    load();
  }, [projectId]);

  const handleExport = () => {
    window.open(generationApi.exportTxt(projectId), "_blank");
  };

  const handlePrev = () => {
    if (!selectedChapter) return;
    const idx = chapters.findIndex((c) => c.id === selectedChapter.id);
    if (idx > 0) setSelectedChapter(chapters[idx - 1]);
  };

  const handleNext = () => {
    if (!selectedChapter) return;
    const idx = chapters.findIndex((c) => c.id === selectedChapter.id);
    if (idx < chapters.length - 1) setSelectedChapter(chapters[idx + 1]);
  };

  return (
    <div className="flex h-[calc(100vh-2rem)]">
      {/* Chapter list sidebar */}
      <div className="w-64 border-r flex flex-col">
        <div className="p-4 border-b flex items-center gap-2">
          <SidebarTrigger />
          <h2 className="font-semibold">Chapters</h2>
        </div>
        <ScrollArea className="flex-1">
          <div className="p-2 space-y-1">
            {chapters.map((ch) => (
              <button
                key={ch.id}
                onClick={() => setSelectedChapter(ch)}
                className={`w-full text-left p-2 rounded text-sm hover:bg-accent transition-colors ${
                  selectedChapter?.id === ch.id ? "bg-accent" : ""
                }`}
              >
                <div className="flex items-center justify-between">
                  <span className="truncate">
                    {ch.chapter_number}. {ch.title}
                  </span>
                  <Badge variant="secondary" className="text-xs ml-1 shrink-0">
                    {ch.final_score}
                  </Badge>
                </div>
              </button>
            ))}
          </div>
        </ScrollArea>
        <div className="p-2 border-t">
          <Button variant="outline" size="sm" className="w-full" onClick={handleExport}>
            Export as .txt
          </Button>
        </div>
      </div>

      {/* Reading area */}
      <div className="flex-1 flex flex-col">
        {selectedChapter ? (
          <>
            <div className="p-4 border-b flex items-center justify-between">
              <div>
                <h1 className="text-xl font-bold">
                  Chapter {selectedChapter.chapter_number}: {selectedChapter.title}
                </h1>
                <div className="flex items-center gap-2 mt-1">
                  <Badge>Score: {selectedChapter.final_score}/10</Badge>
                  <span className="text-xs text-muted-foreground">
                    {selectedChapter.text.length} characters
                  </span>
                </div>
              </div>
              <div className="flex gap-2">
                <Button variant="outline" size="sm" onClick={handlePrev}>
                  Previous
                </Button>
                <Button variant="outline" size="sm" onClick={handleNext}>
                  Next
                </Button>
              </div>
            </div>
            <ScrollArea className="flex-1 p-6">
              <div className="max-w-3xl mx-auto prose prose-sm dark:prose-invert">
                {selectedChapter.text.split("\n").map((para, i) => (
                  <p key={i}>{para}</p>
                ))}
              </div>
            </ScrollArea>
          </>
        ) : (
          <div className="flex-1 flex items-center justify-center text-muted-foreground">
            {chapters.length === 0
              ? "No chapters generated yet"
              : "Select a chapter to read"}
          </div>
        )}
      </div>
    </div>
  );
}
