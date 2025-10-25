import React, { useEffect, useRef, useState } from "react";
import { Header } from "./components/Header";
import { MessageList } from "./components/MessageList";
import { VoiceInput } from "./components/VoiceInput";
import { useSpeechPipeline } from "./hooks/useSpeechPipeline";
import type { Message, Theme, VoiceState } from "./types";

const initialMessages: Message[] = [
  {
    id: 1,
    text: "Hello! Tap the microphone to start talking.",
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

            // Play synthesized audio (only if present)
            if (outputUrl) {
              setIsBotSpeaking(true);
              const audio = new Audio(outputUrl);
              audio.onended = () => {
                setIsBotSpeaking(false);
                setVoiceState("idle");
              };
              await audio.play();
            } else {
              setVoiceState("idle");
            }
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
        theme === "light" ? "bg-zinc-50" : "bg-slate-950"
      }`}
    >
      {/* Neutral animated background */}
      <div className="fixed inset-0 -z-10 overflow-hidden">
        {/* Soft gradient wash */}
        <div className="absolute inset-0 bg-gradient-to-br from-slate-200 via-white to-zinc-200 dark:from-slate-900 dark:via-slate-950 dark:to-zinc-900 animate-[gradient_18s_ease_in_out_infinite] bg-[length:200%_200%]" />
        {/* Subtle grid overlay */}
        <div
          className="absolute inset-0 opacity-[0.05] pointer-events-none mix-blend-overlay"
          style={{
            backgroundImage:
              "linear-gradient(to right, #ffffff 1px, transparent 1px), linear-gradient(to bottom, #ffffff 1px, transparent 1px)",
            backgroundSize: "44px 44px",
          }}
        />
        {/* Vignette */}
        <div
          className="absolute inset-0 pointer-events-none"
          style={{
            background:
              "radial-gradient(ellipse at center, transparent 40%, rgba(0,0,0,0.45) 95%)",
          }}
        />
      </div>

      <div className="flex items-center justify-center min-h-screen p-3 sm:p-6">
        <div
          className="
            relative w-full max-w-4xl h-[92vh] max-h-[900px]
            rounded-3xl overflow-hidden
            border border-black/5 dark:border-white/10
            backdrop-blur-2xl
            shadow-[0_40px_90px_-20px_rgba(0,0,0,0.55)]
            bg-white/60 dark:bg-neutral-900/40
          "
        >
          {/* Subtle inner ring */}
          <div className="pointer-events-none absolute inset-0 rounded-3xl ring-1 ring-black/5 dark:ring-white/10" />
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
