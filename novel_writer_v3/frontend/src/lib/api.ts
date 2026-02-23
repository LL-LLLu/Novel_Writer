const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export async function apiFetch<T>(
  path: string,
  options: RequestInit = {}
): Promise<T> {
  const url = `${API_BASE}${path}`;
  const res = await fetch(url, {
    headers: {
      "Content-Type": "application/json",
      ...options.headers,
    },
    ...options,
  });

  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(error.detail || `API error: ${res.status}`);
  }

  // Handle empty responses (204, etc.)
  const text = await res.text();
  if (!text) return {} as T;
  return JSON.parse(text);
}

// Project API
export const projectsApi = {
  list: () => apiFetch<Project[]>("/api/projects"),
  get: (id: number) => apiFetch<Project>(`/api/projects/${id}`),
  create: (data: ProjectCreate) =>
    apiFetch<Project>("/api/projects", {
      method: "POST",
      body: JSON.stringify(data),
    }),
  update: (id: number, data: Partial<Project>) =>
    apiFetch<Project>(`/api/projects/${id}`, {
      method: "PUT",
      body: JSON.stringify(data),
    }),
  delete: (id: number) =>
    apiFetch<void>(`/api/projects/${id}`, { method: "DELETE" }),
};

// Settings API
export const settingsApi = {
  get: () => apiFetch<Settings>("/api/settings"),
  update: (data: Partial<Settings>) =>
    apiFetch<Settings>("/api/settings", {
      method: "PUT",
      body: JSON.stringify(data),
    }),
};

// Outline API
export const outlineApi = {
  generate: (projectId: number) =>
    apiFetch<{ outline_text: string }>(
      `/api/projects/${projectId}/outline/generate`,
      { method: "POST" }
    ),
  save: (projectId: number, outlineText: string) =>
    apiFetch<{ outline_text: string }>(
      `/api/projects/${projectId}/outline`,
      {
        method: "PUT",
        body: JSON.stringify({ outline_text: outlineText }),
      }
    ),
  generateSections: (projectId: number, numSections: number = 5) =>
    apiFetch<Section[]>(
      `/api/projects/${projectId}/sections/generate`,
      {
        method: "POST",
        body: JSON.stringify({ num_sections: numSections }),
      }
    ),
  listSections: (projectId: number) =>
    apiFetch<Section[]>(`/api/projects/${projectId}/sections`),
  updateSection: (projectId: number, sectionId: number, data: Partial<Section>) =>
    apiFetch<Section>(
      `/api/projects/${projectId}/sections/${sectionId}`,
      {
        method: "PUT",
        body: JSON.stringify(data),
      }
    ),
  expandSection: (projectId: number, sectionId: number, numChapters?: number) =>
    apiFetch<ChapterPlan[]>(
      `/api/projects/${projectId}/sections/${sectionId}/expand`,
      {
        method: "POST",
        body: JSON.stringify({ num_chapters: numChapters }),
      }
    ),
  listChapterPlans: (projectId: number) =>
    apiFetch<ChapterPlan[]>(`/api/projects/${projectId}/chapter-plans`),
  updateChapterPlan: (projectId: number, planId: number, data: Partial<ChapterPlan>) =>
    apiFetch<ChapterPlan>(
      `/api/projects/${projectId}/chapter-plans/${planId}`,
      {
        method: "PUT",
        body: JSON.stringify(data),
      }
    ),
  generateGuidance: (projectId: number, styleDescription: string = "") =>
    apiFetch<{ guidance_text: string }>(
      `/api/projects/${projectId}/guidance/generate`,
      {
        method: "POST",
        body: JSON.stringify({ style_description: styleDescription }),
      }
    ),
  initializeBible: (projectId: number) =>
    apiFetch<{ project_id: number; bible_json: string }>(
      `/api/projects/${projectId}/bible/initialize`,
      { method: "POST" }
    ),
};

// Generation API
export const generationApi = {
  start: (projectId: number, startChapter: number = 1, endChapter?: number) =>
    apiFetch<{ status: string }>(
      `/api/projects/${projectId}/generate`,
      {
        method: "POST",
        body: JSON.stringify({ start_chapter: startChapter, end_chapter: endChapter }),
      }
    ),
  cancel: (projectId: number) =>
    apiFetch<{ status: string }>(
      `/api/projects/${projectId}/generate/cancel`,
      { method: "POST" }
    ),
  listChapters: (projectId: number) =>
    apiFetch<Chapter[]>(`/api/projects/${projectId}/chapters`),
  getChapter: (projectId: number, chapterNum: number) =>
    apiFetch<Chapter>(`/api/projects/${projectId}/chapters/${chapterNum}`),
  getBible: (projectId: number) =>
    apiFetch<StoryBible>(`/api/projects/${projectId}/bible`),
  getLogs: (projectId: number, chapter?: number, limit: number = 100) => {
    let url = `/api/projects/${projectId}/logs?limit=${limit}`;
    if (chapter !== undefined) url += `&chapter=${chapter}`;
    return apiFetch<AgentLog[]>(url);
  },
  exportTxt: (projectId: number) =>
    `${API_BASE}/api/projects/${projectId}/export?format=txt`,
};

// Import types
import type {
  Project,
  ProjectCreate,
  Settings,
  Section,
  ChapterPlan,
  Chapter,
  StoryBible,
  AgentLog,
} from "@/types";
