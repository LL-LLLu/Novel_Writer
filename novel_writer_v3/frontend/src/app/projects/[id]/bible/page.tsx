"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { generationApi } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ScrollArea } from "@/components/ui/scroll-area";
import { toast } from "sonner";
import { SidebarTrigger } from "@/components/ui/sidebar";

interface CharacterState {
  name: string;
  description: string;
  location: string;
  emotional_state: string;
  relationships: Record<string, string>;
  knowledge: string[];
  arc_stage: string;
  last_seen_chapter: number;
}

interface PlotThread {
  name: string;
  description: string;
  status: string;
  introduced_chapter: number;
  resolved_chapter: number;
  key_events: string[];
}

export default function BiblePage() {
  const params = useParams();
  const projectId = Number(params.id);
  const [bible, setBible] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    generationApi
      .getBible(projectId)
      .then((b) => {
        try {
          setBible(JSON.parse(b.bible_json));
        } catch {
          setBible(null);
        }
      })
      .catch(() => setBible(null))
      .finally(() => setLoading(false));
  }, [projectId]);

  if (loading) return <div className="p-6">Loading...</div>;

  if (!bible) {
    return (
      <div className="p-6">
        <div className="flex items-center gap-2 mb-6">
          <SidebarTrigger />
          <h1 className="text-2xl font-bold">Story Bible</h1>
        </div>
        <p className="text-muted-foreground">
          No story bible yet. Initialize it from the Outline page.
        </p>
      </div>
    );
  }

  const characters: CharacterState[] = bible.characters || [];
  const plotThreads: PlotThread[] = bible.plot_threads || [];
  const timeline: string[] = bible.timeline || [];
  const worldNotes: string = bible.world_notes || "";

  return (
    <div className="p-6 max-w-5xl">
      <div className="flex items-center gap-2 mb-6">
        <SidebarTrigger />
        <h1 className="text-2xl font-bold">Story Bible</h1>
      </div>

      <Tabs defaultValue="characters">
        <TabsList className="mb-4">
          <TabsTrigger value="characters">Characters ({characters.length})</TabsTrigger>
          <TabsTrigger value="plot">Plot Threads ({plotThreads.length})</TabsTrigger>
          <TabsTrigger value="timeline">Timeline</TabsTrigger>
          <TabsTrigger value="world">World Notes</TabsTrigger>
        </TabsList>

        <TabsContent value="characters">
          <div className="grid gap-4 md:grid-cols-2">
            {characters.map((c, i) => (
              <Card key={i}>
                <CardHeader className="pb-2">
                  <div className="flex items-center justify-between">
                    <CardTitle className="text-base">{c.name}</CardTitle>
                    <Badge variant="outline">{c.arc_stage}</Badge>
                  </div>
                </CardHeader>
                <CardContent className="text-sm space-y-2">
                  <p>{c.description}</p>
                  <p><strong>Location:</strong> {c.location}</p>
                  <p><strong>Emotional state:</strong> {c.emotional_state}</p>
                  {Object.keys(c.relationships || {}).length > 0 && (
                    <div>
                      <strong>Relationships:</strong>
                      <ul className="list-disc list-inside">
                        {Object.entries(c.relationships).map(([name, rel]) => (
                          <li key={name}>{name}: {rel}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="plot">
          <div className="space-y-4">
            {plotThreads.map((pt, i) => (
              <Card key={i}>
                <CardHeader className="pb-2">
                  <div className="flex items-center justify-between">
                    <CardTitle className="text-base">{pt.name}</CardTitle>
                    <Badge
                      variant={pt.status === "active" ? "default" : pt.status === "resolved" ? "secondary" : "outline"}
                    >
                      {pt.status}
                    </Badge>
                  </div>
                </CardHeader>
                <CardContent className="text-sm space-y-2">
                  <p>{pt.description}</p>
                  <p><strong>Introduced:</strong> Chapter {pt.introduced_chapter}</p>
                  {pt.resolved_chapter > 0 && (
                    <p><strong>Resolved:</strong> Chapter {pt.resolved_chapter}</p>
                  )}
                  {pt.key_events?.length > 0 && (
                    <ul className="list-disc list-inside">
                      {pt.key_events.map((e, j) => (
                        <li key={j}>{e}</li>
                      ))}
                    </ul>
                  )}
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="timeline">
          <Card>
            <CardContent className="pt-6">
              {timeline.length === 0 ? (
                <p className="text-muted-foreground">No timeline entries yet.</p>
              ) : (
                <div className="space-y-2">
                  {timeline.map((entry, i) => (
                    <div key={i} className="flex gap-3 items-start">
                      <div className="w-6 h-6 rounded-full bg-primary/20 flex items-center justify-center text-xs shrink-0">
                        {i + 1}
                      </div>
                      <p className="text-sm">{entry}</p>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="world">
          <Card>
            <CardContent className="pt-6">
              <pre className="whitespace-pre-wrap text-sm">
                {worldNotes || "No world notes yet."}
              </pre>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
