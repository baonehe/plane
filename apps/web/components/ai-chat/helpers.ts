/**
 * Copyright (c) 2023-present Plane Software, Inc. and contributors
 * SPDX-License-Identifier: AGPL-3.0-only
 * See the LICENSE file for details.
 */

import { v4 as uuidv4 } from "uuid";
import type { TAIChatAction, TAIChatMessage } from "@/services/ai.service";

export type TAIChatEntry = TAIChatMessage & {
  id: string;
  actions?: TAIChatAction[];
  // error replies are shown in the thread but never sent back to the model
  isError?: boolean;
  errorDetail?: string;
};

// The API only keeps the latest 20 turns; older ones only matter for display.
export const AI_CHAT_MAX_SENT_MESSAGES = 20;
export const AI_CHAT_MAX_STORED_ENTRIES = 100;

export const AI_CHAT_OPEN_STORAGE_KEY = "plane_ai_chat_open";
export const getAIChatHistoryStorageKey = (workspaceSlug: string) => `plane_ai_chat_history_${workspaceSlug}`;

export const toChatPayload = (entries: TAIChatEntry[]): TAIChatMessage[] =>
  entries
    .filter((entry) => !entry.isError)
    .slice(-AI_CHAT_MAX_SENT_MESSAGES)
    .map(({ role, content }) => ({ role, content }));

export const appendChatEntry = (entries: TAIChatEntry[], entry: Omit<TAIChatEntry, "id">): TAIChatEntry[] =>
  [...entries, { ...entry, id: uuidv4() }].slice(-AI_CHAT_MAX_STORED_ENTRIES);

const uniqueBy = (actions: TAIChatAction[], getKey: (action: TAIChatAction) => string) => {
  const unique = new Map<string, TAIChatAction>();
  actions.forEach((action) => unique.set(getKey(action), action));
  return [...unique.entries()].map(([key, action]) => ({ key, action }));
};

// one refresh per touched work item
export const getUniqueActionTargets = (actions: TAIChatAction[] = []) =>
  uniqueBy(actions, (action) => action.id).map(({ action }) => action);

// one chip per kind of change on each work item
export const getDisplayActions = (actions: TAIChatAction[] = []) =>
  uniqueBy(actions, (action) => `${action.action}-${action.id}`);
