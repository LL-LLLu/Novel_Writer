"use client";

import { useEffect, useState } from "react";
import { settingsApi } from "@/lib/api";
import type { Settings } from "@/types";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { toast } from "sonner";
import { SidebarTrigger } from "@/components/ui/sidebar";

export default function SettingsPage() {
  const [settings, setSettings] = useState<Settings>({
    gemini_api_key: "",
    qwen_api_key: "",
    gemini_model: "gemini-3.1-pro-preview",
    qwen_model: "qwen3.5-plus",
    qwen_base_url: "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    gemini_temperature: 0.7,
    qwen_temperature: 0.8,
    max_output_tokens: 4096,
    max_debate_rounds: 3,
    chapter_min_chars: 5000,
    chapter_max_chars: 8000,
  });
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    settingsApi
      .get()
      .then(setSettings)
      .catch((err) => toast.error(err.message))
      .finally(() => setLoading(false));
  }, []);

  const handleSave = async () => {
    setSaving(true);
    try {
      const updated = await settingsApi.update(settings);
      setSettings(updated);
      toast.success("Settings saved");
    } catch (err: any) {
      toast.error(err.message);
    } finally {
      setSaving(false);
    }
  };

  if (loading) return <div className="p-6">Loading...</div>;

  return (
    <div className="p-6 max-w-3xl">
      <div className="flex items-center gap-2 mb-6">
        <SidebarTrigger />
        <h1 className="text-2xl font-bold">Settings</h1>
      </div>

      <div className="space-y-6">
        <Card>
          <CardHeader>
            <CardTitle>Gemini API (Google)</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <Label>API Key</Label>
              <Input
                type="password"
                value={settings.gemini_api_key}
                onChange={(e) =>
                  setSettings({ ...settings, gemini_api_key: e.target.value })
                }
                placeholder="Enter Gemini API key"
              />
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <Label>Model</Label>
                <Input
                  value={settings.gemini_model}
                  onChange={(e) =>
                    setSettings({ ...settings, gemini_model: e.target.value })
                  }
                />
              </div>
              <div>
                <Label>Temperature</Label>
                <Input
                  type="number"
                  step="0.1"
                  min="0"
                  max="2"
                  value={settings.gemini_temperature}
                  onChange={(e) =>
                    setSettings({
                      ...settings,
                      gemini_temperature: parseFloat(e.target.value),
                    })
                  }
                />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Qwen API (Alibaba DashScope)</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <Label>API Key</Label>
              <Input
                type="password"
                value={settings.qwen_api_key}
                onChange={(e) =>
                  setSettings({ ...settings, qwen_api_key: e.target.value })
                }
                placeholder="Enter Qwen API key"
              />
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <Label>Model</Label>
                <Input
                  value={settings.qwen_model}
                  onChange={(e) =>
                    setSettings({ ...settings, qwen_model: e.target.value })
                  }
                />
              </div>
              <div>
                <Label>Temperature</Label>
                <Input
                  type="number"
                  step="0.1"
                  min="0"
                  max="2"
                  value={settings.qwen_temperature}
                  onChange={(e) =>
                    setSettings({
                      ...settings,
                      qwen_temperature: parseFloat(e.target.value),
                    })
                  }
                />
              </div>
            </div>
            <div>
              <Label>Base URL</Label>
              <Input
                value={settings.qwen_base_url}
                onChange={(e) =>
                  setSettings({ ...settings, qwen_base_url: e.target.value })
                }
              />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Generation Settings</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-3 gap-4">
              <div>
                <Label>Max Output Tokens</Label>
                <Input
                  type="number"
                  value={settings.max_output_tokens}
                  onChange={(e) =>
                    setSettings({
                      ...settings,
                      max_output_tokens: parseInt(e.target.value),
                    })
                  }
                />
              </div>
              <div>
                <Label>Max Debate Rounds</Label>
                <Input
                  type="number"
                  min="1"
                  max="10"
                  value={settings.max_debate_rounds}
                  onChange={(e) =>
                    setSettings({
                      ...settings,
                      max_debate_rounds: parseInt(e.target.value),
                    })
                  }
                />
              </div>
              <div>
                <Label>Min Chapter Chars</Label>
                <Input
                  type="number"
                  value={settings.chapter_min_chars}
                  onChange={(e) =>
                    setSettings({
                      ...settings,
                      chapter_min_chars: parseInt(e.target.value),
                    })
                  }
                />
              </div>
            </div>
            <div>
              <Label>Max Chapter Chars</Label>
              <Input
                type="number"
                value={settings.chapter_max_chars}
                onChange={(e) =>
                  setSettings({
                    ...settings,
                    chapter_max_chars: parseInt(e.target.value),
                  })
                }
              />
            </div>
          </CardContent>
        </Card>

        <Button onClick={handleSave} disabled={saving} className="w-full">
          {saving ? "Saving..." : "Save Settings"}
        </Button>
      </div>
    </div>
  );
}
