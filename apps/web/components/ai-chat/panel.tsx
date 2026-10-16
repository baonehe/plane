/**
 * Copyright (c) 2023-present Plane Software, Inc. and contributors
 * SPDX-License-Identifier: AGPL-3.0-only
 * See the LICENSE file for details.
 */

import { useEffect, useRef, useState } from "react";
import type { KeyboardEvent } from "react";
import { observer } from "mobx-react";
import Link from "next/link";
// plane imports
import { Icon } from "@makeplane/propel/components/icon";
import { IconButton } from "@makeplane/propel/components/icon-button";
import { TextArea, TextAreaGroup } from "@makeplane/propel/components/text-area";
import { CloseOutline, NewChatOutline, SendOutline } from "@makeplane/propel/icons";
import { PiChatLogo } from "@plane/blocks/icons";
import { useLocalStorage } from "@plane/hooks";
import { useTranslation } from "@plane/i18n";
import { cn } from "@plane/utils";
// hooks
import { useIssueDetail } from "@/hooks/store/use-issue-detail";
// services
import type { TAIChatAction, TAIChatError } from "@/services/ai.service";
import { AIService } from "@/services/ai.service";
// local imports
import type { TAIChatEntry } from "./helpers";
import {
  appendChatEntry,
  getAIChatHistoryStorageKey,
  getDisplayActions,
  getUniqueActionTargets,
  toChatPayload,
} from "./helpers";

const aiService = new AIService();

type Props = {
  workspaceSlug: string;
  onClose: () => void;
};

export const AIChatPanel = observer(function AIChatPanel(props: Props) {
  const { workspaceSlug, onClose } = props;
  // states
  const [input, setInput] = useState("");
  const [isSending, setIsSending] = useState(false);
  // refs
  const scrollRef = useRef<HTMLDivElement>(null);
  // propel's TextArea types its ref (Base UI Field.Control) as an input; only focus() is used
  const inputRef = useRef<HTMLInputElement>(null);
  // plane hooks
  const { t } = useTranslation();
  const { storedValue, setValue } = useLocalStorage<TAIChatEntry[]>(getAIChatHistoryStorageKey(workspaceSlug), []);
  // store hooks
  const { fetchIssue } = useIssueDetail();
  // derived values
  const entries = storedValue ?? [];
  const suggestions = [t("ai_chat.suggestions.my_open"), t("ai_chat.suggestions.due_this_week")];

  // the panel only mounts when the user opens it, so move focus straight to the message box
  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [entries.length, isSending]);

  const refreshWorkItems = (actions: TAIChatAction[] = []) => {
    getUniqueActionTargets(actions).forEach((action) => {
      fetchIssue(workspaceSlug, action.project_id, action.id).catch(() => undefined);
    });
  };

  const handleSend = async (text: string) => {
    const content = text.trim();
    if (!content || isSending) return;

    const withUserMessage = appendChatEntry(entries, { role: "user", content });
    setValue(withUserMessage);
    setInput("");
    setIsSending(true);
    try {
      const response = await aiService.chat(workspaceSlug, toChatPayload(withUserMessage));
      setValue(
        appendChatEntry(withUserMessage, { role: "assistant", content: response.message, actions: response.actions })
      );
      refreshWorkItems(response.actions);
    } catch (error) {
      const { error: errorDetail, actions } = (error ?? {}) as TAIChatError;
      setValue(
        appendChatEntry(withUserMessage, {
          role: "assistant",
          content: t("ai_chat.error"),
          errorDetail,
          actions,
          isError: true,
        })
      );
      refreshWorkItems(actions);
    } finally {
      setIsSending(false);
    }
  };

  const handleKeyDown = (event: KeyboardEvent) => {
    // Let IME input (e.g. Vietnamese, Japanese) finish composing before Enter sends
    if (event.key !== "Enter" || event.shiftKey || event.nativeEvent.isComposing) return;
    event.preventDefault();
    void handleSend(input);
  };

  return (
    <div
      role="dialog"
      aria-label={t("sidebar.pi_chat")}
      className="fixed right-5 bottom-5 z-20 flex h-[min(600px,calc(100vh-2.5rem))] w-[min(400px,calc(100vw-2.5rem))] flex-col overflow-hidden rounded-lg border border-subtle bg-surface-1 shadow-raised-200"
    >
      <div className="flex items-center gap-2 border-b border-subtle px-4 py-3">
        <PiChatLogo className="size-5 flex-shrink-0 text-accent-primary" />
        <div className="min-w-0 flex-1">
          <p className="truncate text-14 font-medium text-primary">{t("sidebar.pi_chat")}</p>
          <p className="truncate text-11 text-tertiary">{t("ai_chat.subtitle")}</p>
        </div>
        <IconButton
          variant="ghost"
          size="sm"
          icon={<Icon icon={NewChatOutline} />}
          aria-label={t("ai_chat.new_chat")}
          disabled={isSending || entries.length === 0}
          onClick={() => setValue([])}
        />
        <IconButton
          variant="ghost"
          size="sm"
          icon={<Icon icon={CloseOutline} />}
          aria-label={t("close")}
          onClick={onClose}
        />
      </div>

      <div ref={scrollRef} className="flex flex-1 flex-col gap-3 overflow-y-auto px-4 py-4">
        {entries.length === 0 ? (
          <div className="my-auto flex flex-col items-center gap-3 text-center">
            <PiChatLogo className="size-8 text-accent-primary" />
            <p className="text-13 text-secondary">{t("ai_chat.empty_state")}</p>
            <div className="flex w-full flex-col gap-2">
              {suggestions.map((suggestion) => (
                <button
                  key={suggestion}
                  type="button"
                  className="rounded-md border border-subtle bg-layer-1 px-3 py-2 text-left text-13 text-secondary hover:bg-layer-1-hover hover:text-primary"
                  onClick={() => void handleSend(suggestion)}
                >
                  {suggestion}
                </button>
              ))}
            </div>
          </div>
        ) : (
          entries.map((entry) => <AIChatMessage key={entry.id} entry={entry} workspaceSlug={workspaceSlug} />)
        )}
        {isSending && (
          <div className="max-w-[85%] animate-pulse self-start rounded-lg bg-layer-1 px-3 py-2 text-13 text-tertiary">
            {t("ai_chat.thinking")}
          </div>
        )}
      </div>

      <div className="flex items-end gap-2 border-t border-subtle px-3 py-3">
        <div className="flex-1">
          <TextAreaGroup resize="none">
            <TextArea
              size="md"
              surface="field"
              autoResize
              maxRows={6}
              rows={1}
              value={input}
              placeholder={t("ai_chat.placeholder")}
              aria-label={t("ai_chat.placeholder")}
              onChange={(event) => setInput(event.target.value)}
              onKeyDown={handleKeyDown}
              ref={inputRef}
            />
          </TextAreaGroup>
        </div>
        <IconButton
          variant="primary"
          size="lg"
          icon={<Icon icon={SendOutline} />}
          aria-label={t("ai_chat.send")}
          disabled={!input.trim()}
          loading={isSending}
          onClick={() => void handleSend(input)}
        />
      </div>
    </div>
  );
});

type MessageProps = {
  entry: TAIChatEntry;
  workspaceSlug: string;
};

function AIChatMessage(props: MessageProps) {
  const { entry, workspaceSlug } = props;
  const { t } = useTranslation();
  const isUser = entry.role === "user";
  const actions = getDisplayActions(entry.actions);

  return (
    <div className={cn("flex max-w-[85%] flex-col gap-1.5", isUser ? "self-end" : "self-start")}>
      <div
        className={cn("rounded-lg px-3 py-2 text-13 break-words whitespace-pre-wrap", {
          "bg-accent-primary text-on-color": isUser,
          "bg-layer-1 text-primary": !isUser && !entry.isError,
          "bg-danger-subtle text-danger-primary": entry.isError,
        })}
      >
        {entry.content}
        {entry.errorDetail && <p className="mt-1 text-11 opacity-80">{entry.errorDetail}</p>}
      </div>
      {actions.length > 0 && (
        <div className="flex flex-col gap-1">
          {actions.map(({ key, action }) => (
            <Link
              key={key}
              href={`/${workspaceSlug}/browse/${action.reference}/`}
              className="flex items-center gap-1.5 rounded-md border border-subtle px-2 py-1 text-11 text-secondary hover:bg-layer-transparent-hover hover:text-primary"
            >
              <span className="flex-shrink-0 font-medium text-accent-primary">
                {t(`ai_chat.actions.${action.action}`)} · {action.reference}
              </span>
              <span className="truncate">{action.name}</span>
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}
