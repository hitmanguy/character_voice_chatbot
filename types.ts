export interface Message {
  id: number | string;
  text: string;
  timestamp: string;
  sender: 'user' | 'bot';
}

export type Theme = 'light' | 'dark';

export type VoiceState = 'idle' | 'listening' | 'processing';
