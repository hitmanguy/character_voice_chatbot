import React, { useEffect, useRef, useState } from "react";
import { Header } from "./components/Header";
import { MessageList } from "./components/MessageList";
import { VoiceInput } from "./components/VoiceInput";
import { useSpeechPipeline } from "./hooks/useSpeechPipeline";
import type { Message, Theme, VoiceState } from "./types";

const initialMessages: Message[] = [
  {
    id: 1,
    text: "Hello! I am Iron Man. Tap the microphone to start our conversation.",
    timestamp: "10:30 AM",
    sender: "bot",
  },
];

export default function App() {
  const [theme, setTheme] = useState<Theme>("dark");
  const [messages, setMessages] = useState<Message[]>(initialMessages);
  const [voiceState, setVoiceState] = useState<VoiceState>("idle");
  const [isBotSpeaking, setIsBotSpeaking] = useState(false);
  const [isTyping, setIsTyping] = useState(false);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const streamRef = useRef<MediaStream | null>(null);
  const { run } = useSpeechPipeline("http://127.0.0.1:8000");

  useEffect(() => {
    if (theme === "dark") {
      document.documentElement.classList.add("dark");
    } else {
      document.documentElement.classList.remove("dark");
    }
  }, [theme]);

  const toggleTheme = () => {
    setTheme((prev) => (prev === "light" ? "dark" : "light"));
  };

  const cleanupStream = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
  };

  const handleMicClick = async () => {
    if (voiceState === "idle") {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: true,
        });
        streamRef.current = stream;

        const options: MediaRecorderOptions = {};
        if (MediaRecorder.isTypeSupported("audio/webm"))
          options.mimeType = "audio/webm";
        const mr = new MediaRecorder(stream, options);

        audioChunksRef.current = [];

        mr.ondataavailable = (e) => {
          if (e.data && e.data.size > 0) audioChunksRef.current.push(e.data);
        };

        mr.onstop = async () => {
          const blob = new Blob(audioChunksRef.current, { type: "audio/webm" });
          setVoiceState("processing");
          setIsTyping(true);

          try {
            const { outputUrl, text, inputText } = await run(blob);

            const now = new Date().toLocaleTimeString([], {
              hour: "2-digit",
              minute: "2-digit",
            });
            const userMsg: Message = {
              id: Date.now(),
              text: inputText || "🎙️ Voice message",
              timestamp: now,
              sender: "user",
            };
            const botMsg: Message = {
              id: Date.now() + 1,
              text: text || "I generated a response.",
              timestamp: now,
              sender: "bot",
            };

            // Remove temp placeholder (-1) if present and append real messages
            setMessages((prev) => [
              ...prev.filter((m) => m.id !== -1),
              userMsg,
              botMsg,
            ]);
            setIsTyping(false);

            // Play synthesized audio
            setIsBotSpeaking(true);
            const audio = new Audio(outputUrl);
            audio.onended = () => {
              setIsBotSpeaking(false);
              setVoiceState("idle");
            };
            await audio.play();
          } catch (err) {
            console.error(err);
            setIsTyping(false);
            setVoiceState("idle");
          } finally {
            cleanupStream();
          }
        };

        mediaRecorderRef.current = mr;
        mr.start();
        setVoiceState("listening");

        // Optional temporary placeholder while recording/transcribing
        setTimeout(() => {
          if (voiceState === "listening") {
            const tempUserMessage: Message = {
              id: -1, // temporary placeholder id
              text: "I think you said...",
              timestamp: new Date().toLocaleTimeString([], {
                hour: "2-digit",
                minute: "2-digit",
              }),
              sender: "user",
            };
            setMessages((prev) => [
              ...prev.filter((m) => m.id !== -1),
              tempUserMessage,
            ]);
          }
        }, 1500);
      } catch (e) {
        console.error("Mic permission or recording failed:", e);
      }
    } else if (voiceState === "listening") {
      try {
        mediaRecorderRef.current?.stop();
      } catch {
        // ignore
      }
    }
  };

  return (
    <main
      className={`w-full min-h-screen font-sans transition-colors duration-300 ${
        theme === "light" ? "bg-gray-100" : "bg-gray-900"
      }`}
    >
      <div className="fixed inset-0 -z-0">
        <div className="absolute inset-0 bg-gradient-to-br from-blue-100 via-white to-purple-100 dark:from-gray-800 dark:via-gray-900 dark:to-indigo-900 animate-[gradient_15s_ease_infinite] bg-[length:200%_200%]"></div>
      </div>
      <div className="flex items-center justify-center min-h-screen p-2 sm:p-4">
        <div className="relative w-full max-w-4xl h-[95vh] max-h-[900px] bg-white/30 dark:bg-black/30 backdrop-blur-2xl rounded-3xl shadow-2xl flex flex-col overflow-hidden border border-white/20">
          <Header theme={theme} toggleTheme={toggleTheme} />
          <MessageList
            messages={messages}
            isTyping={isTyping}
            isBotSpeaking={isBotSpeaking}
          />
          <VoiceInput voiceState={voiceState} onMicClick={handleMicClick} />
        </div>
      </div>
    </main>
  );
}
