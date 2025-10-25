import React from 'react';
import { motion } from 'framer-motion';

export const SoundWaveIcon: React.FC = () => {
  const waveVariants = {
    animate: (i: number) => ({
      scaleY: [1, 1.5, 1],
      transition: {
        duration: 0.8,
        repeat: Infinity,
        delay: i * 0.2,
        ease: 'easeInOut',
      },
    }),
  };

  return (
    <motion.div className="flex items-center justify-center space-x-0.5 w-4 h-4 ml-2">
      {[...Array(3)].map((_, i) => (
        <motion.div
          key={i}
          custom={i}
          variants={waveVariants}
          animate="animate"
          className="w-0.5 h-full bg-current origin-bottom"
        />
      ))}
    </motion.div>
  );
};
