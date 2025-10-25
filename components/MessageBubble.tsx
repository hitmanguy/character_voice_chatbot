import React from 'react';
import { motion } from 'framer-motion';
import { Message } from '../types';
import { SoundWaveIcon } from './icons/SoundWaveIcon';

interface MessageBubbleProps {
  message: Message;
  isSpeaking?: boolean;
}

export const MessageBubble: React.FC<MessageBubbleProps> = ({ message, isSpeaking }) => {
  const isUser = message.sender === 'user';

  const bubbleVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0 },
  };

  return (
    <motion.div
      variants={bubbleVariants}
      initial="hidden"
      animate="visible"
      transition={{ duration: 0.3 }}
      className={`flex flex-col mb-4 ${isUser ? 'items-end' : 'items-start'}`}
    >
      <div
        className={`max-w-xs md:max-w-md lg:max-w-lg px-4 py-3 rounded-3xl shadow-md flex items-center ${
          isUser
            ? 'bg-gradient-to-br from-blue-500 to-indigo-600 text-white rounded-br-lg'
            : 'bg-gradient-to-br from-teal-400 to-green-500 text-white rounded-bl-lg'
        }`}
      >
        <p className="text-sm">{message.text}</p>
        {!isUser && isSpeaking && <SoundWaveIcon />}
      </div>
      <span className="text-xs text-gray-400 dark:text-gray-500 mt-1.5 px-2">
        {message.timestamp}
      </span>
    </motion.div>
  );
};
