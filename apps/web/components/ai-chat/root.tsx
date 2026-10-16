/**
 * Copyright (c) 2023-present Plane Software, Inc. and contributors
 * SPDX-License-Identifier: AGPL-3.0-only
 * See the LICENSE file for details.
 */

import { observer } from "mobx-react";
import { useParams } from "next/navigation";
// plane imports
import { PiChatLogo } from "@plane/blocks/icons";
import { EUserPermissionsLevel } from "@plane/constants";
import { useLocalStorage } from "@plane/hooks";
import { useTranslation } from "@plane/i18n";
import { EUserWorkspaceRoles } from "@plane/types";
// hooks
import { useUserPermissions } from "@/hooks/store/user";
// local imports
import { AI_CHAT_OPEN_STORAGE_KEY } from "./helpers";
import { AIChatPanel } from "./panel";

export const AIChatWidget = observer(function AIChatWidget() {
  const { workspaceSlug } = useParams();
  // plane hooks
  const { t } = useTranslation();
  const { storedValue: isOpen, setValue: setIsOpen } = useLocalStorage<boolean>(AI_CHAT_OPEN_STORAGE_KEY, false);
  // store hooks
  const { allowPermissions } = useUserPermissions();

  if (!workspaceSlug) return null;
  // the chat endpoint is limited to workspace admins and members, like the tools behind it
  if (
    !allowPermissions(
      [EUserWorkspaceRoles.ADMIN, EUserWorkspaceRoles.MEMBER],
      EUserPermissionsLevel.WORKSPACE,
      workspaceSlug.toString()
    )
  )
    return null;

  if (isOpen) return <AIChatPanel workspaceSlug={workspaceSlug.toString()} onClose={() => setIsOpen(false)} />;

  return (
    <button
      type="button"
      aria-label={t("ai_chat.open")}
      title={t("ai_chat.open")}
      className="fixed right-5 bottom-5 z-20 grid size-12 place-items-center rounded-full bg-accent-primary text-on-color shadow-raised-200 transition-transform hover:scale-105 focus-visible:ring-2 focus-visible:ring-accent-strong focus-visible:outline-none"
      onClick={() => setIsOpen(true)}
    >
      <PiChatLogo className="size-6" />
    </button>
  );
});
