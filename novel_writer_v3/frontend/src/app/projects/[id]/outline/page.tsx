"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { projectsApi, outlineApi } from "@/lib/api";
import type { Project, Section, ChapterPlan } from "@/types";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Separator } from "@/components/ui/separator";
import { toast } from "sonner";
import { SidebarTrigger } from "@/components/ui/sidebar";

export default function OutlinePage() {
  const params = useParams();
  const projectId = Number(params.id);
  const [project, setProject] = useState<Project | null>(null);
  const [outline, setOutline] = useState("");
  const [sections, setSections] = useState<Section[]>([]);
  const [chapterPlans, setChapterPlans] = useState<ChapterPlan[]>([]);
  const [numSections, setNumSections] = useState(5);
  const [loading, setLoading] = useState<string | null>(null);
  const [styleDesc, setStyleDesc] = useState("");

  const loadData = async () => {
    try {
      const p = await projectsApi.get(projectId);
      setProject(p);
      setOutline(p.outline_text);
      const s = await outlineApi.listSections(projectId);
      setSections(s);
      const cp = await outlineApi.listChapterPlans(projectId);
      setChapterPlans(cp);
    } catch (err: any) {
      toast.error(err.message);
    }
  };

  useEffect(() => {
    loadData();
  }, [projectId]);

  const handleGenerateOutline = async () => {
    setLoading("outline");
    try {
      const result = await outlineApi.generate(projectId);
      setOutline(result.outline_text);
      toast.success("Outline generated");
      loadData();
    } catch (err: any) {
      toast.error(err.message);
    } finally {
      setLoading(null);
    }
  };

  const handleSaveOutline = async () => {
    try {
      await outlineApi.save(projectId, outline);
      toast.success("Outline saved");
    } catch (err: any) {
      toast.error(err.message);
    }
  };

  const handleGenerateSections = async () => {
    setLoading("sections");
    try {
      const result = await outlineApi.generateSections(projectId, numSections);
      setSections(result);
      toast.success(`${result.length} sections created`);
      loadData();
    } catch (err: any) {
      toast.error(err.message);
    } finally {
      setLoading(null);
    }
  };

  const handleExpandSection = async (sectionId: number) => {
    setLoading(`expand-${sectionId}`);
    try {
      const result = await outlineApi.expandSection(projectId, sectionId);
      toast.success(`${result.length} chapter plans created`);
      loadData();
    } catch (err: any) {
      toast.error(err.message);
    } finally {
      setLoading(null);
    }
  };

  const handleExpandAll = async () => {
    setLoading("expand-all");
    try {
      for (const section of sections) {
        await outlineApi.expandSection(projectId, section.id);
      }
      toast.success("All sections expanded");
      loadData();
    } catch (err: any) {
      toast.error(err.message);
    } finally {
      setLoading(null);
    }
  };

  const handleGenerateGuidance = async () => {
    setLoading("guidance");
    try {
      const result = await outlineApi.generateGuidance(projectId, styleDesc);
      toast.success("Guidance generated");
      loadData();
    } catch (err: any) {
      toast.error(err.message);
    } finally {
      setLoading(null);
    }
  };

  const handleInitBible = async () => {
    setLoading("bible");
    try {
      await outlineApi.initializeBible(projectId);
      toast.success("Story bible initialized");
    } catch (err: any) {
      toast.error(err.message);
    } finally {
      setLoading(null);
    }
  };

  const handleUpdateSection = async (sectionId: number, field: string, value: string) => {
    try {
      await outlineApi.updateSection(projectId, sectionId, { [field]: value } as any);
      loadData();
    } catch (err: any) {
      toast.error(err.message);
    }
  };

  if (!project) return <div className="p-6">Loading...</div>;

  return (
    <div className="p-6 max-w-5xl">
      <div className="flex items-center gap-2 mb-6">
        <SidebarTrigger />
        <h1 className="text-2xl font-bold">Outline: {project.title}</h1>
      </div>

      <Tabs defaultValue="outline">
        <TabsList className="mb-4">
          <TabsTrigger value="outline">1. Full Outline</TabsTrigger>
          <TabsTrigger value="sections">2. Sections</TabsTrigger>
          <TabsTrigger value="chapters">3. Chapter Plans</TabsTrigger>
          <TabsTrigger value="guidance">Style Guidance</TabsTrigger>
        </TabsList>

        <TabsContent value="outline">
          <Card>
            <CardHeader>
              <CardTitle>Full Outline (Pass 1)</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <Textarea
                value={outline}
                onChange={(e) => setOutline(e.target.value)}
                rows={20}
                placeholder="Story outline will appear here..."
                className="font-mono text-sm"
              />
              <div className="flex gap-2">
                <Button onClick={handleGenerateOutline} disabled={!!loading}>
                  {loading === "outline" ? "Generating..." : "Generate Outline"}
                </Button>
                <Button variant="outline" onClick={handleSaveOutline} disabled={!!loading}>
                  Save Outline
                </Button>
                <Button variant="outline" onClick={handleInitBible} disabled={!!loading}>
                  {loading === "bible" ? "Initializing..." : "Initialize Story Bible"}
                </Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="sections">
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle>Sections / Acts (Pass 2)</CardTitle>
                <div className="flex items-center gap-2">
                  <Label>Number of sections:</Label>
                  <Input
                    type="number"
                    min={3}
                    max={8}
                    value={numSections}
                    onChange={(e) => setNumSections(parseInt(e.target.value) || 5)}
                    className="w-20"
                  />
                  <Button onClick={handleGenerateSections} disabled={!!loading}>
                    {loading === "sections" ? "Splitting..." : "Split into Sections"}
                  </Button>
                </div>
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              {sections.length === 0 ? (
                <p className="text-muted-foreground">No sections yet. Generate outline first, then split into sections.</p>
              ) : (
                sections.map((s) => (
                  <Card key={s.id}>
                    <CardHeader className="pb-2">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <Badge variant="outline">Section {s.section_number}</Badge>
                          <Input
                            value={s.title}
                            onChange={(e) => handleUpdateSection(s.id, "title", e.target.value)}
                            className="font-semibold"
                          />
                        </div>
                        <div className="flex items-center gap-2">
                          <span className="text-sm text-muted-foreground">{s.chapter_count} chapters</span>
                          <Badge variant={s.status === "expanded" ? "default" : "secondary"}>
                            {s.status}
                          </Badge>
                        </div>
                      </div>
                    </CardHeader>
                    <CardContent>
                      <Textarea
                        value={s.summary}
                        onChange={(e) => handleUpdateSection(s.id, "summary", e.target.value)}
                        rows={3}
                        className="text-sm"
                      />
                      <div className="mt-2">
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => handleExpandSection(s.id)}
                          disabled={!!loading}
                        >
                          {loading === `expand-${s.id}` ? "Expanding..." : "Expand to Chapters"}
                        </Button>
                      </div>
                    </CardContent>
                  </Card>
                ))
              )}
              {sections.length > 0 && (
                <Button onClick={handleExpandAll} disabled={!!loading} className="w-full">
                  {loading === "expand-all" ? "Expanding all..." : "Expand All Sections"}
                </Button>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="chapters">
          <Card>
            <CardHeader>
              <CardTitle>Chapter Plans (Pass 3)</CardTitle>
            </CardHeader>
            <CardContent>
              {chapterPlans.length === 0 ? (
                <p className="text-muted-foreground">No chapter plans yet. Expand sections first.</p>
              ) : (
                <div className="space-y-2">
                  {sections.map((section) => {
                    const sectionPlans = chapterPlans.filter(
                      (cp) => cp.section_id === section.id
                    );
                    if (sectionPlans.length === 0) return null;
                    return (
                      <div key={section.id}>
                        <h3 className="font-semibold text-sm text-muted-foreground mb-2">
                          Section {section.section_number}: {section.title}
                        </h3>
                        <div className="space-y-1">
                          {sectionPlans.map((cp) => {
                            let planData: any = {};
                            try {
                              planData = JSON.parse(cp.plan_json);
                            } catch {}
                            return (
                              <Card key={cp.id} className="p-3">
                                <div className="flex items-center justify-between">
                                  <div>
                                    <span className="font-mono text-xs text-muted-foreground mr-2">
                                      Ch.{cp.chapter_number}
                                    </span>
                                    <span className="font-medium">{cp.title}</span>
                                  </div>
                                  <Badge variant="secondary" className="text-xs">
                                    {cp.status}
                                  </Badge>
                                </div>
                                {planData.key_events && (
                                  <ul className="mt-1 text-xs text-muted-foreground list-disc list-inside">
                                    {planData.key_events.slice(0, 3).map((e: string, i: number) => (
                                      <li key={i}>{e}</li>
                                    ))}
                                  </ul>
                                )}
                              </Card>
                            );
                          })}
                        </div>
                        <Separator className="my-3" />
                      </div>
                    );
                  })}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="guidance">
          <Card>
            <CardHeader>
              <CardTitle>Style Guidance</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <Label>Style Description (optional)</Label>
                <Textarea
                  value={styleDesc}
                  onChange={(e) => setStyleDesc(e.target.value)}
                  rows={3}
                  placeholder="Describe the writing style you want (e.g., literary fiction, fast-paced thriller, etc.)"
                />
              </div>
              <Button onClick={handleGenerateGuidance} disabled={!!loading}>
                {loading === "guidance" ? "Generating..." : "Generate Guidance"}
              </Button>
              {project.guidance_text && (
                <div className="mt-4">
                  <Label>Generated Guidance</Label>
                  <pre className="mt-2 whitespace-pre-wrap text-sm bg-muted p-4 rounded-lg max-h-96 overflow-auto">
                    {project.guidance_text}
                  </pre>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
