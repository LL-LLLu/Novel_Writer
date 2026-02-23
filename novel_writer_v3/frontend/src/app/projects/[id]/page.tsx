"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { projectsApi } from "@/lib/api";
import type { Project } from "@/types";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { toast } from "sonner";
import { SidebarTrigger } from "@/components/ui/sidebar";

const STATUS_COLORS: Record<string, string> = {
  created: "bg-gray-500",
  outline_ready: "bg-blue-500",
  sections_ready: "bg-indigo-500",
  chapters_ready: "bg-purple-500",
  generating: "bg-yellow-500",
  completed: "bg-green-500",
  error: "bg-red-500",
};

export default function ProjectPage() {
  const params = useParams();
  const projectId = Number(params.id);
  const [project, setProject] = useState<Project | null>(null);

  useEffect(() => {
    projectsApi
      .get(projectId)
      .then(setProject)
      .catch((err) => toast.error(err.message));
  }, [projectId]);

  if (!project) return <div className="p-6">Loading...</div>;

  const links = [
    { href: `/projects/${projectId}/outline`, label: "Outline Editor", description: "Generate and edit hierarchical outline" },
    { href: `/projects/${projectId}/generate`, label: "Generate", description: "Start chapter generation with debate pipeline" },
    { href: `/projects/${projectId}/read`, label: "Read", description: "Read generated chapters" },
    { href: `/projects/${projectId}/bible`, label: "Story Bible", description: "View characters, plot threads, timeline" },
    { href: `/projects/${projectId}/logs`, label: "Agent Logs", description: "View agent activity and timing" },
  ];

  return (
    <div className="p-6 max-w-4xl">
      <div className="flex items-center gap-2 mb-6">
        <SidebarTrigger />
        <h1 className="text-2xl font-bold">{project.title}</h1>
        <Badge className={`${STATUS_COLORS[project.status] || "bg-gray-500"} text-white ml-2`}>
          {project.status}
        </Badge>
      </div>

      <Card className="mb-6">
        <CardHeader>
          <CardTitle>Project Details</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2">
          <p><strong>Premise:</strong> {project.premise || "Not set"}</p>
          <p><strong>Target Chapters:</strong> {project.target_chapters}</p>
          <p><strong>Language:</strong> {project.language === "auto" ? "Auto-detect" : project.language.toUpperCase()}</p>
          <p><strong>Created:</strong> {new Date(project.created_at).toLocaleString()}</p>
        </CardContent>
      </Card>

      <div className="grid gap-4 md:grid-cols-2">
        {links.map((link) => (
          <Link key={link.href} href={link.href}>
            <Card className="hover:border-primary transition-colors cursor-pointer h-full">
              <CardHeader>
                <CardTitle className="text-lg">{link.label}</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground">{link.description}</p>
              </CardContent>
            </Card>
          </Link>
        ))}
      </div>
    </div>
  );
}
