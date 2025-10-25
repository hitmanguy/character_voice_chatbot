
import React from 'react';
import { motion } from 'framer-motion';

export const TypingIndicator: React.FC = () => {
  const dotVariants = {
    initial: { y: 0 },
    animate: {
      y: -5,
      transition: {
        duration: 0.4,
        ease: "easeInOut",
        repeat: Infinity,
        repeatType: "reverse",
      },
    },
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0 }}
      className="flex items-center space-x-1.5 p-3"
    >
      <motion.span
        className="h-2 w-2 rounded-full bg-gray-400 dark:bg-gray-500"
        variants={dotVariants}
        initial="initial"
        animate="animate"
      />
      <motion.span
        className="h-2 w-2 rounded-full bg-gray-400 dark:bg-gray-500"
        variants={dotVariants}
        initial="initial"
        animate="animate"
        style={{ animationDelay: '0.2s' }}
      />
      <motion.span
        className="h-2 w-2 rounded-full bg-gray-400 dark:bg-gray-500"
        variants={dotVariants}
        initial="initial"
        animate="animate"
        style={{ animationDelay: '0.4s' }}
      />
    </motion.div>
  );
};
