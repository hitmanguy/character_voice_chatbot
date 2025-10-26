import React from "react";
import { AnimatePresence } from "framer-motion";
import { Message } from "../types";
import { useAutoScroll } from "../hooks/useAutoScroll";
import { MessageBubble } from "./MessageBubble";
import { TypingIndicator } from "./TypingIndicator";

interface MessageListProps {
  messages: Message[];
  isTyping: boolean;
  isBotSpeaking: boolean;
}

export const MessageList: React.FC<MessageListProps> = ({
  messages,
  isTyping,
  isBotSpeaking,
}) => {
  const scrollRef = useAutoScroll([messages, isTyping]);

  return (
    <div
      ref={scrollRef}
      className="flex-1 overflow-y-auto p-6 space-y-4 pt-24 pb-48"
    >
      <AnimatePresence>
        {messages.map((msg, index) => (
          <MessageBubble
            key={msg.id}
            message={msg}
            isSpeaking={
              msg.sender === "bot" &&
              index === messages.length - 1 &&
              isBotSpeaking
            }
          />
        ))}
      </AnimatePresence>
      {isTyping && <TypingIndicator />}
    </div>
  );
};
