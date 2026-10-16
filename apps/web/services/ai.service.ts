/**
 * Copyright (c) 2023-present Plane Software, Inc. and contributors
 * SPDX-License-Identifier: AGPL-3.0-only
 * See the LICENSE file for details.
 */

// helpers
import { API_BASE_URL } from "@plane/constants";
import type { AI_EDITOR_TASKS } from "@plane/constants";
// services
import { APIService } from "@/services/api.service";
// types
// FIXME:
// import { IGptResponse } from "@plane/types";
// helpers

export type TTaskPayload = {
  casual_score?: number;
  formal_score?: number;
  task: AI_EDITOR_TASKS;
  text_input: string;
};

export type TAIChatMessage = {
  role: "user" | "assistant";
  content: string;
};

export type TAIChatAction = {
  action: "created" | "updated" | "commented";
  id: string;
  project_id: string;
  reference: string;
  name: string;
};

export type TAIChatResponse = {
  message: string;
  actions: TAIChatAction[];
};

export type TAIChatError = {
  error?: string;
  actions?: TAIChatAction[];
};

export class AIService extends APIService {
  constructor() {
    super(API_BASE_URL);
  }

  async createGptTask(workspaceSlug: string, data: { prompt: string; task: string }): Promise<any> {
    return this.post(`/api/workspaces/${workspaceSlug}/ai-assistant/`, data)
      .then((response) => response?.data)
      .catch((error) => {
        throw error?.response;
      });
  }

  async chat(workspaceSlug: string, messages: TAIChatMessage[]): Promise<TAIChatResponse> {
    return this.post(`/api/workspaces/${workspaceSlug}/ai-chat/`, { messages })
      .then((response) => response?.data)
      .catch((error) => {
        throw (error?.response?.data ?? {}) as TAIChatError;
      });
  }

  async performEditorTask(
    workspaceSlug: string,
    data: TTaskPayload
  ): Promise<{
    response: string;
  }> {
    return this.post(`/api/workspaces/${workspaceSlug}/rephrase-grammar/`, data)
      .then((res) => res?.data)
      .catch((error) => {
        throw error?.response?.data;
      });
  }
}
