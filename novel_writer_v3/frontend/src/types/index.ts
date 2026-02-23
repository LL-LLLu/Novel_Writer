export interface Project {
  id: number;
  title: string;
  premise: string;
  outline_text: string;
  guidance_text: string;
  style_rules_json: string;
  language: string;
  target_chapters: number;
  status: string;
  created_at: string;
  updated_at: string;
}

export interface ProjectCreate {
  title: string;
  premise?: string;
  language?: string;
  target_chapters?: number;
}

export interface Settings {
  gemini_api_key: string;
  qwen_api_key: string;
  gemini_model: string;
  qwen_model: string;
  qwen_base_url: string;
  gemini_temperature: number;
  qwen_temperature: number;
  max_output_tokens: number;
  max_debate_rounds: number;
  chapter_min_chars: number;
  chapter_max_chars: number;
}

export interface Section {
  id: number;
  project_id: number;
  section_number: number;
  title: string;
  summary: string;
  chapter_count: number;
  status: string;
  created_at: string;
}

export interface ChapterPlan {
  id: number;
  project_id: number;
  section_id: number | null;
  chapter_number: number;
  title: string;
  plan_json: string;
  status: string;
  created_at: string;
}

export interface Chapter {
  id: number;
  project_id: number;
  chapter_plan_id: number | null;
  chapter_number: number;
  title: string;
  text: string;
  final_score: number;
  debate_rounds_json: string;
  status: string;
  created_at: string;
}

export interface StoryBible {
  project_id: number;
  bible_json: string;
  updated_at: string;
}

export interface AgentLog {
  id: number;
  project_id: number;
  chapter_number: number | null;
  agent_name: string;
  action: string;
  prompt_preview: string;
  response_preview: string;
  elapsed_seconds: number;
  created_at: string;
}

export interface ProgressMessage {
  type: string;
  chapter: number;
  total: number;
  stage: string;
  debate_round: number;
  message: string;
}
