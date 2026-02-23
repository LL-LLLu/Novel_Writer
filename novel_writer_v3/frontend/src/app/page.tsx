"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { projectsApi } from "@/lib/api";
import type { Project } from "@/types";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
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

export default function DashboardPage() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [loading, setLoading] = useState(true);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [newProject, setNewProject] = useState({
    title: "",
    premise: "",
    language: "auto",
    target_chapters: 10,
  });

  const loadProjects = async () => {
    try {
      const data = await projectsApi.list();
      setProjects(data);
    } catch (err: any) {
      toast.error(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadProjects();
  }, []);

  const handleCreate = async () => {
    if (!newProject.title.trim()) {
      toast.error("Title is required");
      return;
    }
    try {
      await projectsApi.create(newProject);
      toast.success("Project created");
      setDialogOpen(false);
      setNewProject({ title: "", premise: "", language: "auto", target_chapters: 10 });
      loadProjects();
    } catch (err: any) {
      toast.error(err.message);
    }
  };

  const handleDelete = async (id: number, e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (!confirm("Delete this project?")) return;
    try {
      await projectsApi.delete(id);
      toast.success("Project deleted");
      loadProjects();
    } catch (err: any) {
      toast.error(err.message);
    }
  };

  return (
    <div className="p-6">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-2">
          <SidebarTrigger />
          <h1 className="text-2xl font-bold">Projects</h1>
        </div>
        <Dialog open={dialogOpen} onOpenChange={setDialogOpen}>
          <DialogTrigger asChild>
            <Button>New Project</Button>
          </DialogTrigger>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Create New Project</DialogTitle>
            </DialogHeader>
            <div className="space-y-4">
              <div>
                <Label>Title</Label>
                <Input
                  value={newProject.title}
                  onChange={(e) =>
                    setNewProject({ ...newProject, title: e.target.value })
                  }
                  placeholder="My Novel"
                />
              </div>
              <div>
                <Label>Premise</Label>
                <Textarea
                  value={newProject.premise}
                  onChange={(e) =>
                    setNewProject({ ...newProject, premise: e.target.value })
                  }
                  placeholder="A story about..."
                  rows={4}
                />
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <Label>Language</Label>
                  <Select
                    value={newProject.language}
                    onValueChange={(v) =>
                      setNewProject({ ...newProject, language: v })
                    }
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="auto">Auto-detect</SelectItem>
                      <SelectItem value="en">English</SelectItem>
                      <SelectItem value="zh">Chinese</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div>
                  <Label>Target Chapters</Label>
                  <Input
                    type="number"
                    min={1}
                    value={newProject.target_chapters}
                    onChange={(e) =>
                      setNewProject({
                        ...newProject,
                        target_chapters: parseInt(e.target.value) || 10,
                      })
                    }
                  />
                </div>
              </div>
              <Button onClick={handleCreate} className="w-full">
                Create Project
              </Button>
            </div>
          </DialogContent>
        </Dialog>
      </div>

      {loading ? (
        <p className="text-muted-foreground">Loading...</p>
      ) : projects.length === 0 ? (
        <Card>
          <CardContent className="py-10 text-center text-muted-foreground">
            No projects yet. Create one to get started.
          </CardContent>
        </Card>
      ) : (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {projects.map((p) => (
            <Link key={p.id} href={`/projects/${p.id}`}>
              <Card className="hover:border-primary transition-colors cursor-pointer">
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <CardTitle className="text-lg">{p.title}</CardTitle>
                    <Badge
                      variant="secondary"
                      className={`${STATUS_COLORS[p.status] || "bg-gray-500"} text-white`}
                    >
                      {p.status}
                    </Badge>
                  </div>
                  <CardDescription className="line-clamp-2">
                    {p.premise || "No premise"}
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="flex justify-between text-sm text-muted-foreground">
                    <span>{p.target_chapters} chapters</span>
                    <span>{p.language === "auto" ? "Auto" : p.language.toUpperCase()}</span>
                  </div>
                  <div className="mt-2 flex justify-end">
                    <Button
                      variant="destructive"
                      size="sm"
                      onClick={(e) => handleDelete(p.id, e)}
                    >
                      Delete
                    </Button>
                  </div>
                </CardContent>
              </Card>
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}
