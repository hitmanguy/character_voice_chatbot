import React from 'react';
import { motion } from 'framer-motion';
import { Theme } from '../types';
import { SunIcon } from './icons/SunIcon';
import { MoonIcon } from './icons/MoonIcon';

interface HeaderProps {
  theme: Theme;
  toggleTheme: () => void;
}

export const Header: React.FC<HeaderProps> = ({ theme, toggleTheme }) => {
  return (
    <div className="absolute top-0 left-0 right-0 p-4 bg-white/10 backdrop-blur-md rounded-t-3xl border-b border-white/10 z-10">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <img
            src="https://picsum.photos/seed/ai-assistant/40/40"
            alt="AI Assistant Avatar"
            className="w-10 h-10 rounded-full border-2 border-white/30"
          />
          <div>
            <h1 className="font-semibold text-gray-800 dark:text-gray-100">Voice Assistant</h1>
            <div className="flex items-center space-x-1.5">
              <span className="h-2 w-2 rounded-full bg-green-400 animate-pulse"></span>
              <p className="text-xs text-gray-500 dark:text-gray-400">Online</p>
            </div>
          </div>
        </div>
        <motion.button
          onClick={toggleTheme}
          className="p-2 rounded-full text-gray-500 dark:text-gray-400 hover:bg-black/10 dark:hover:bg-white/10"
          whileHover={{ scale: 1.1 }}
          whileTap={{ scale: 0.9 }}
        >
          {theme === 'light' ? <MoonIcon className="w-5 h-5" /> : <SunIcon className="w-5 h-5" />}
        </motion.button>
      </div>
    </div>
  );
};
