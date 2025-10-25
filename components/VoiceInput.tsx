import React from 'react';
import { motion } from 'framer-motion';
import { VoiceState } from '../types';
import { MicIcon } from './icons/MicIcon';

interface VoiceInputProps {
  voiceState: VoiceState;
  onMicClick: () => void;
}

const stateHints: Record<VoiceState, string> = {
  idle: 'Tap the mic to speak',
  listening: 'Listening... Tap again to stop',
  processing: 'Processing your request...',
};

export const VoiceInput: React.FC<VoiceInputProps> = ({ voiceState, onMicClick }) => {
  return (
    <div className="absolute bottom-0 left-0 right-0 p-4 bg-white/10 backdrop-blur-md rounded-b-3xl border-t border-white/10 flex flex-col items-center justify-center pt-6 pb-8">
       <div className="relative flex items-center justify-center w-20 h-20">
         {voiceState === 'listening' && (
             <motion.div
                 className="absolute inset-0 bg-blue-500/30 rounded-full"
                 animate={{ scale: [1, 1.5, 1], opacity: [1, 0, 1] }}
                 transition={{ duration: 1.5, repeat: Infinity, ease: 'easeOut' }}
             />
         )}
         {voiceState === 'processing' && (
             <motion.div
                 className="absolute inset-0 border-4 border-blue-500 rounded-full"
                 style={{ borderTopColor: 'transparent', borderBottomColor: 'transparent' }}
                 animate={{ rotate: 360 }}
                 transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
             />
         )}
        <motion.button
            onClick={onMicClick}
            className="relative w-16 h-16 rounded-full bg-gradient-to-br from-blue-500 to-indigo-600 text-white flex items-center justify-center shadow-lg"
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
        >
          <MicIcon className="w-7 h-7" />
        </motion.button>
       </div>
      <p className="mt-4 text-sm text-gray-600 dark:text-gray-300 transition-opacity duration-300">
        {stateHints[voiceState]}
      </p>
    </div>
  );
};
